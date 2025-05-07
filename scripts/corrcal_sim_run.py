from astropy.utils import iers
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "warn"

import warnings
warnings.filterwarnings("ignore", "hera_cal")
warnings.filterwarnings("ignore", "pyfof")
warnings.filterwarnings("ignore", "Matplotlib")
warnings.filterwarnings("ignore", "Casting")
warnings.filterwarnings("ignore", "WARNING")

import argparse
import corrcal
import healpy
import hera_sim
import numpy as np
import sys
import yaml

from astropy import constants, units
from astropy.coordinates import Latitude, Longitude, AltAz
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pathlib import Path
from pyradiosky import SkyModel
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AiryBeam
from scipy.optimize import minimize
from time import time

parser = argparse.ArgumentParser("CLI for CorrCal tests.")

parser.add_argument("config", type=str, help="Path to config file.")
parser.add_argument(
    "-o",
    "--outdir",
    type=str,
    default=".",
    help="Where to save test results.",
)
parser.add_argument(
    "--src_dir",
    type=str,
    default="/project/s/sievers/rpascua/corrcal_tests/source_catalogs",
    help="Where the sky model files are saved.",
)
parser.add_argument(
    "--pix_wgts",
    type=str,
    default=None,
    help="Path to directory containing SPHT pixel weights.",
)
parser.add_argument(
    "--maxiter",
    type=int,
    default=None,
    help="Maximum number of iterations to let the solver run.",
)
parser.add_argument(
    "--flat_sky",
    action="store_true",
    default=False,
    help="Whether to compute the diffuse matrix with in the flat sky limit.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Hack for skipping files that have already been completed.
    if len(list(Path(args.outdir).glob("*.npz"))):
        sys.exit()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    array_layout = config.get("array_layout", "HERA-19")
    if array_layout.lower().startswith("hera"):
        diameter = 14
        sep = 14.6
        hex_num = {
            7: 2, 19: 3, 37: 4, 61: 5
        }[int(array_layout.split("-")[1])]
        array_layout = hera_sim.antpos.hex_array(
            hex_num=hex_num, sep=sep, split_core=False, outriggers=0
        )
    elif array_layout.lower().startswith("chord"):
        diameter = 6
        sep_e = 6.5
        sep_n = 9
        array_layout = {}
        # Hard-code 8x8 grid for now.
        n_rows = 8
        n_cols = 8
        ant = 0
        for row in range(n_rows):
            for col in range(n_cols):
                array_layout[ant] = np.array([row*sep_e, col*sep_n, 0])
                ant += 1
    elif isinstance(array_layout, str):
        raise ValueError("Array is unknown.")
    else:
        if not isinstance(array_layout, dict):
            raise ValueError("Array layout not understood.")
        diameter = config.get("diameter", 14)

    # Keep track of this for later.
    ideal_array = {ant: np.array(pos) for ant, pos in array_layout.items()}

    # Extract other information from the config file.
    # TODO: more general beam support
    beam = AiryBeam(diameter=diameter)
    beam_ids = [0,] * len(array_layout)
    freq = config.get("freq", 150e6)
    obstime = config.get("obstime", 2459917.737631866)
    latitude = config.get("latitude", -30.721526120689507)
    longitude = config.get("longitude", 21.428303826863015)
    altitude = config.get("altitude", 1051.690000018105)

    # Add jitter to antenna positions if requested
    jitter = config.get("jitter", 0)
    if jitter > 0:
        for ant, pos in array_layout.items():
            dpos = np.random.normal(loc=0, scale=jitter*diameter, size=3)
            dpos[-1] = 0  # Only perturb in EW/NS directions.
            array_layout[ant] = pos + dpos

    # Make the UVData object.
    uvdata = hera_sim.io.empty_uvdata(
        Ntimes=1,
        start_time=obstime,
        integration_time=1.0,
        array_layout=array_layout,
        Nfreqs=1,
        start_freq=freq,
        channel_width=1.0,
        telescope_location=np.array([latitude, longitude, altitude]),
        polarization_array=["xx",],
    )
    diff_uvdata = uvdata.copy()

    # Generate source catalog.
    src_dir = Path(args.src_dir)
    field_type = config.get("field_type", "calib")
    realization = config.get("realization", "00")
    src_file = src_dir / f"{field_type}_fields/catalog_{realization}.npz"
    src_info = dict(np.load(src_file))
    ra = Longitude(
        (uvdata.lst_array[0] + src_info["dra"]) * units.rad
    ).value
    dec = Latitude(
        (uvdata.telescope_location_lat_lon_alt[0] + src_info["ddec"]) * units.rad
    ).value
    fluxes = src_info["fluxes"]  # Jy

    # Simulate the visibilities.
    stokes = np.zeros((4, 1, fluxes.size), dtype=float)
    stokes[0,0] = fluxes
    src_model = SkyModel(
        name=np.arange(fluxes.size).astype(str),
        ra=Longitude(ra*units.rad),
        dec=Latitude(dec*units.rad),
        stokes=stokes*units.Jy,
        spectral_type="flat",
        component_type="point",
        frame="icrs",
    )
    data_model = hera_sim.visibilities.ModelData(
        uvdata=uvdata,
        sky_model=src_model,
        beam_ids=beam_ids,
        beams=[beam,],
    )
    simulation = hera_sim.visibilities.VisibilitySimulation(
        data_model=data_model,
        simulator=hera_sim.visibilities.MatVis(),
    )
    simulation.simulate()

    # Generate diffuse sky model.
    max_Tsky = config.get("max_Tsky", 500)
    nside = config.get("nside", 64)
    ell_max = 3*nside - 1
    ells = np.arange(ell_max + 1, dtype=float)
    sky_pspec = (1+ells)**config.get("pspec_idx", -2)
    Tsky = healpy.synfast(sky_pspec, nside)

    # Rescale the sky to the desired amplitude.
    rescaling = max_Tsky / Tsky.max()
    Tsky *= rescaling

    # Rescale the power spectrum appropriately.
    # (Additional factor of 2 is fftvis Stokes convention.)
    K_to_Jy = 1 / hera_sim.utils.jansky_to_kelvin(freq/1e9, 1)
    sky_pspec *= (K_to_Jy * rescaling.value / 2)**2

    # Prepare the diffuse sky model.
    stokes = np.zeros((4, 1, Tsky.size), dtype=float)
    stokes[0,0] = Tsky
    diff_model = SkyModel(
        stokes=stokes*units.K,
        spectral_type="flat",
        component_type="healpix",
        nside=nside,
        frame="icrs",
        hpx_inds=np.arange(Tsky.size),
        freq_array=np.array([freq]) * units.Hz,
    )

    # Manually convert the diffuse model into a point source model.
    diffuse_model.kelvin_to_jansky()
    diffuse_model.stokes *= healpy.nside2pixarea(nside) * units.sr
    npix = healpy.nside2npix(nside)
    pix_dec, pix_ra = healpy.pix2ang(nside, np.arange(npix))
    pix_dec = np.pi/2 - pix_dec  # pix2ang outputs colatitude
    diffuse_model = SkyModel(
        name=np.arange(npix).astype(str),
        stokes=diffuse_model.stokes,
        ra=Longitude(pix_ra*units.rad),
        dec=Latitude(pix_dec*units.rad),
        spectral_type="flat",
        component_type="point",
        frame="icrs",
        freq_array=np.array([freq])*units.Hz,
    )
    
    # Simulate the visibilities.
    data_model = hera_sim.visibilities.ModelData(
        uvdata=diff_uvdata,
        sky_model=diff_model,
        beam_ids=beam_ids,
        beams=[beam,],
    )
    simulation = hera_sim.visibilities.VisibilitySimulation(
        data_model=data_model,
        simulator=hera_sim.visibilities.FFTVis(),
    )
    simulation.simulate()

    # Sum the data together.
    uvdata.data_array += diff_uvdata.data_array


    # Delete things we won't need anymore.
    del data_model, diff_model, diff_uvdata, src_model, simulation, stokes

    # Compute some helpful things.
    ideal_uvdata = hera_sim.io.empty_uvdata(
        Ntimes=1,
        start_time=obstime,
        integration_time=1.0,
        Nfreqs=1,
        start_freq=freq,
        channel_width=1.0,
        array_layout=ideal_array,
    )

    # Get the baselines we'll use for calibration.
    min_length = config.get("min_length", np.sqrt(2) * diameter)
    min_group_size = max(config.get("min_group_size", 5), config["n_eig"])
    ant_1_array, ant_2_array, edges = corrcal.gridding.make_groups_from_uvdata(
        ideal_uvdata, min_bl_len=min_length, min_group_size=min_group_size
    )

    # Compute the uvws; apparently fftvis uses b_ij = x_i - x_j convention
    uvws = freq * (
        enu_antpos[ant_1_array] - enu_antpos[ant_2_array]
    ) / constants.c.si.value

    # Compute diffuse matrix. First, some auxiliary things.
    enu_antpos = uvdata.get_ENU_antpos()[0]
    za, az = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))

    # Construct the AltAz frame for coordinate transformations.
    observatory = EarthLocation(longitude, latitude, altitude)
    local_frame = AltAz(location=observatory, obstime=Time(obstime, format="jd"))
    
    if args.flat_sky:
        from astropy_healpix import HEALPix
        from scipy.stats import binned_statistic
        from scipy.interpolate import interp1d
        from scipy.special import spherical_jn, sph_harm_y

        # Prepare some auxiliary parameters for computing covariance elements.
        nside = 64
        lmax = 10  # We won't need more Bessel functions than this.
        ells, emms = healpy.Alm.getlm(lmax)
        ells = ells[:,None,None]
        emms = emms[:,None,None]  # Conform shapes for later calculations.
        zero_m = emms[:,0,0] == 0
        pixels = np.arange(healpy.nside2npix(nside))
        bm_thetas, bm_phis = healpy.pix2ang(nside, pixels)
        hpx_bsq_vals = beam.power_eval(
            az_array=bm_phis,
            za_array=bm_thetas,
            freq_array=np.array([freq]),
        )[0][0,0] ** 2
        if hpx_bsq_vals.ndim == 3:
            hpx_bsq_vals = hpx_bsq_vals[0,0]
        hpx_bsq_vals[bm_thetas>np.pi/2] = 0  # Horizon cut.
        Blm = healpy.map2alm(
            hpx_bsq_vals,
            use_pixel_weights=True,
            lmax=lmax,
            datapath=args.pix_wgts,
        )[:,None,None]
        sky_ells = np.arange(sky_pspec.size)

        # Now compute the diffuse matrix.
        n_eig = config["n_eig"]
        diff_mat = np.zeros((edges[-1], 2*n_eig), dtype=float)
        for start, stop in zip(edges, edges[1:]):
            # Compute amplitude of baseline deviations within the group.
            uvw_here = uvws[start//2:stop//2]
            uvw_diffs = uvw_here[:,None] - uvw_here[None]
            uvw_diff_mags = np.linalg.norm(uvw_diffs, axis=2, keepdims=True)

            # Get orientations of baseline differences.
            normed_uvw_diffs = uvw_diffs / np.where(
                uvw_diff_mags > 0, uvw_diff_mags, 1
            )
            thetas = np.arccos(normed_uvw_diffs[:,:,2])[None,:,:]
            phis = np.arctan2(
                normed_uvw_diffs[:,:,1], normed_uvw_diffs[:,:,0]
            )[None,:,:]

            # Compute the bandpower measured by this group.
            avg_uvw_mag = np.linalg.norm(uvw_here.mean(axis=0))
            pspec_bessels = special.spherical_jn(sky_ells, 2*np.pi*avg_uvw_mag)
            band_power = 4 * np.pi * np.sum(
                sky_pspec * pspec_bessels**2 * (2*sky_ells + 1)
            )

            # Now compute the covariance for this block.
            bessels = spherical_jn(ells, 2*np.pi*uvw_diff_mags[None,:,:])
            Ylms = sph_harm_y(ells, emms, thetas, phis)
            flm = 1j**ells * bessels * Ylms
            offset = np.sum(flm[zero_m] * Blm[zero_m], axis=0)
            B_kk = np.sum(
                flm*Blm + (-1)**ells * flm.conj() * Blm.conj(), axis=0
            ) - offset

            # Fill in the real/imaginary block covariance.
            block = np.zeros(2*np.array(B_kk.shape), dtype=float)
            block[::2,::2] = B_kk.real
            block[1::2,1::2] = B_kk.real
            block[1::2,::2] = B_kk.imag
            block[::2,1::2] = -B_kk.imag
            block *= 0.5 * band_power

            # Now take the eigendecomposition.
            eigvals, eigvecs = np.linalg.eigh(block)
            sort = np.argsort(eigvals)[::-1]
            eigvals = eigvals[sort][:2*n_eig]
            eigvecs = eigvecs[:,sort][:,:2*n_eig]
            block = np.sqrt(eigvals)[None,:] * eigvecs

            # This should hopefully just be a no-op, but it's possible that
            # for very redundant cases and many eigenmodes, some of the modes
            # are negative (since they don't actually contribute meaningfully).
            diff_mat[start:stop] = np.where(np.isnan(block), 0, block)
    else:
        # For now, removing this code and just adding a NotImplementedError.
        # When this is implemented, some care will be needed since healpy only
        # computes spherical harmonics for real-valued fields. A treatment
        # in harmonics will probably also need to use Wigner 3-j matrices and
        # so on when we're allowing for non-redundancy, so... much work TBD.
        # The flat sky approach above converges very nicely to the sample
        # covariance *as long as the array is planar*. For arrays that have
        # substantial w components, we may need to do things in harmonic space.
        # This is still TBD, requiring a more careful systematic analysis of
        # how various covariance modeling approaches compare to whatever
        # the empirical covariance converges to.
        raise NotImplementedError("Please set the --flat_sky flag.")

    # Compute source matrix. First, figure out local source positions.
    source_positions = SkyCoord(
        Longitude(ra*units.rad), Latitude(dec*units.rad), frame="icrs"
    ).transform_to(local_frame)

    # Now downselect to calibration candidates.
    above_horizon = source_positions.alt > 0
    near_zenith = source_positions.alt.deg > 80
    is_bright = fluxes > 100  # Fluxes are assumed to be in Jy.
    select = near_zenith | (is_bright & above_horizon)
    src_az = Longitude(
        np.pi/2*units.rad - source_positions.az
    )[select].to(units.rad).value
    src_za = np.pi/2 - source_positions.alt.rad[select]
    src_flux = fluxes[select]

    # Compute the beam at each source position.
    src_beam_vals = beam.power_eval(
        az_array=src_az,
        za_array=src_za,
        freq_array=np.array([freq]),
    )[0][0,0]
    if src_beam_vals.ndim == 2:
        src_beam_vals = src_beam_vals[0]

    # Optionally add some flux errors.
    flux_err = np.random.normal(
        loc=1, scale=config.get("flux_err", 0), size=src_flux.size
    )
    # Rescale the fluxes to agree with matvis/fftvis convention.
    src_fluxes = 0.5 * src_flux * src_beam_vals * flux_err

    # Now figure out which sources to keep for calibration.
    n_src = config.get("n_src", 1)
    sort = np.argsort(src_fluxes)[::-1]  # Sort from highest to lowest flux.
    src_az = src_az[sort][:n_src]
    src_za = src_za[sort][:n_src]
    src_flux = src_fluxes[sort][:n_src]

    # Finally, actually compute the source matrix.
    src_nhat = np.array(
        [
            np.sin(src_za)*np.cos(src_az),
            np.sin(src_za)*np.sin(src_az),
            np.cos(src_za),
        ]
    )
    src_fringe = np.exp(-2j * np.pi * uvws @ src_nhat)
    src_vis = src_flux[None,:] * src_fringe
    src_mat = np.zeros((edges[-1], n_src), dtype=float)
    src_mat[::2] = src_vis.real
    src_mat[1::2] = src_vis.imag

    # Simulate gains.
    gain_amp = config.get("gain_amp", 1)
    amp_jitter = config.get("amp_jitter", 0.1)
    n_ants = len(array_layout)
    gain_amplitudes = np.random.normal(
        size=n_ants, loc=gain_amp, scale=amp_jitter
    )
    gain_phases = np.random.uniform(0, 2*np.pi, n_ants)
    true_gains = gain_amplitudes * np.exp(1j*gain_phases)

    # Generate initial guess.
    gain_amp_err = config.get("gain_amp_err", 0.05)
    gain_phs_err = config.get("gain_phs_err", 0.02)
    init_amp = gain_amplitudes * np.random.normal(
        loc=1, size=n_ants, scale=gain_amp_err
    )
    init_phs = gain_phases + np.random.normal(
        loc=0, size=n_ants, scale=gain_phs_err
    )
    init_gains = init_amp * np.exp(1j*init_phs)
    split_gains = np.zeros(2*n_ants, dtype=float)
    split_gains[::2] = init_gains.real
    split_gains[1::2] = init_gains.imag

    # Compute noise-related stuff. Anecdotally, SNR on the crosses is about
    # 5-20% of SNR on the autos. The SNR we're setting in the config is the
    # desired SNR in the crosses.
    snr = config.get("snr", 20)
    #fudge_factor = 8 / 0.07
    fudge_factor = 500  # It looks like this is roughly a factor of 10 too big
    snr = snr * fudge_factor
    integration_time = 1
    channel_width = snr ** 2
    omega_p = np.array([1])
    autocorr = uvdata.get_data(0, 0, "xx")
    noise_amp = np.abs(autocorr)[0,0] / snr
    noise = 0.5 * np.ones(2*ant_1_array.size, dtype=float) * noise_amp**2

    # Now initialize the sparse covariance.
    cov = corrcal.sparse.SparseCov(
        noise=noise,
        src_mat=src_mat,
        diff_mat=diff_mat,
        edges=edges,
        n_eig=2*n_eig,
        isinv=False,
    )
    gain_scale = 1
    phs_norm = 1

    # Initialize arrays for storing useful data and metadata.
    n_realizations = config.get("n_noise_realizations", 1)
    gain_solutions = np.zeros((n_realizations, n_ants), dtype=complex)
    noisy_visibilities = np.zeros(
        (n_realizations, ant_1_array.size), dtype=complex
    )
    all_visibilities = np.zeros(
        (n_realizations, uvdata.Nbls), dtype=complex
    )
    solver_stats = {"nit": [], "nfev": [], "njev": [], "walltime": []}

    # Loop over noise realizations, calibrating and taking note of results
    for i in range(n_realizations):
        # Simulate some noise.
        sim = hera_sim.Simulator(data=ideal_uvdata.copy())
        sim.data.data_array = uvdata.data_array.copy()
        sim.add(
            "thermal_noise",
            integration_time=integration_time,
            channel_width=channel_width,
            omega_p=omega_p,
            seed="initial",
            autovis=autocorr,
        )

        # Record the full set of visibilities.
        all_visibilities[i] = sim.data_array.flatten()

        # Sort the data by redundancy.
        data = np.array(
            [
                sim.get_data(ai,aj,"xx")[0,0]
                for ai, aj in zip(ant_1_array, ant_2_array)
            ]
        ).flatten()
        
        # Now let's keep track of the true visibilities, then apply gains.
        noisy_visibilities[i,:] = data.copy()
        data = data * true_gains[ant_1_array] * true_gains[ant_2_array].conj()
        split_data = np.zeros(2*data.size, dtype=float)
        split_data[::2] = data.real
        split_data[1::2] = data.imag
        
        # Run the minimizer, and track how long it takes.
        opt_args = (
            cov, split_data, ant_1_array, ant_2_array, gain_scale, phs_norm
        )
        t1 = time()
        try:
            result = minimize(
                corrcal.optimize.nll,
                gain_scale*split_gains,
                args=opt_args,
                method="CG",
                jac=corrcal.optimize.grad_nll,
                options={"maxiter": args.maxiter},
            )
        except np.linalg.LinAlgError:
            # This should never happen, but just in case.
            class Failure:
                def __init__(self):
                    self.x = np.ones_like(split_gains)
                    self.nit = 0
                    self.nfev = 0
                    self.njev = 0
            result = Failure()
        t2 = time()

        # Record the gain solutions and the solver stats.
        gain_solutions[i,:] = corrcal.utils.rephase_to_ant(
            result.x[::2] + 1j*result.x[1::2], 0
        ) / gain_scale
        solver_stats["nit"].append(result.nit)
        solver_stats["nfev"].append(result.nfev)
        solver_stats["njev"].append(result.njev)
        solver_stats["walltime"].append(t2-t1)

    # Write visibilities, solutions, and stats to disk
    outdir = Path(args.outdir)
    basename = Path(args.config).stem
    np.savez(
        outdir / f"{basename}_results.npz",
        vis=noisy_visibilities,
        all_vis=all_visibilities,
        init_gains=init_gains,
        true_gains=true_gains,
        gain_sols=gain_solutions,
        ant_1_array=ant_1_array,
        ant_2_array=ant_2_array,
        full_ant_1_array=uvdata.ant_1_array,
        full_ant_2_array=uvdata.ant_2_array,
        edges=edges,
        enu_antpos=enu_antpos,
        src_mat=src_mat,
        diff_mat=diff_mat,
        noise=noise,
    )
    with open(outdir/f"{basename}_results.yaml", "w") as f:
        yaml.dump(solver_stats, f)
