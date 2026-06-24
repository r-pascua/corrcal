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
import hera_cal
import hera_sim
import numpy as np
import sys
import matvis
import yaml

from astropy import constants, units
from astropy.coordinates import Latitude, Longitude, AltAz
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pathlib import Path
from pyradiosky import SkyModel
from pyuvdata import UVBeam, AiryBeam, GaussianBeam
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

if __name__ == "__main__":
    args = parser.parse_args()

    # Hack for skipping files that have already been completed.
    if len(list(Path(args.outdir).glob("*.npz"))):
        sys.exit()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Going to define things in terms of wavelength.
    freq = config.get("freq", 150e6)
    wavelength = constants.c.si.value / freq
    diameter = 14 * wavelength
    sep = 20 * wavelength
    jitter = 0.04 * wavelength

    # Keep track of this for later.
    ideal_array = {}
    perturbed_array = {}
    ant = 0
    for row in range(8):
        for col in range(8):
            dpos = np.random.normal(size=3, scale=jitter)
            dpos[-1] = 0
            ideal_array[ant] = np.array([col*sep, row*sep, 0])
            perturbed_array[ant] = ideal_array[ant] + dpos
            ant += 1

    # Extract other information from the config file.
    # TODO: more general beam support
    # beam = GaussianBeam(diameter=diameter, reference_frequency=freq)
    beam = AiryBeam(diameter=diameter)
    beam_ids = [0,] * len(ideal_array)
    obstime = config.get("obstime", 2459917.737631866)
    latitude = config.get("latitude", -30.721526120689507)
    longitude = config.get("longitude", 21.428303826863015)
    altitude = config.get("altitude", 1051.690000018105)

    # Make the UVData object.
    uvdata = hera_sim.io.empty_uvdata(
        Ntimes=1,
        start_time=obstime,
        integration_time=1.0,
        array_layout=perturbed_array,
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
    offset = np.abs(Tsky.min())
    Tsky += offset  # Make sure map temperature is non-negative.
    rescaling = max_Tsky / Tsky.max()
    Tsky *= rescaling

    # Update the sky power spectrum appropriately.
    K_to_Jy = 1 / hera_sim.utils.jansky_to_kelvin(freq/1e9, 1)
    sky_pspec[0] += offset**2
    sky_pspec *= (rescaling * K_to_Jy / 2) ** 2

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

    # Manually convert from K to Jy
    diff_model.kelvin_to_jansky()
    diff_model.stokes *= healpy.nside2pixarea(nside) * units.sr
    npix = healpy.nside2npix(nside)
    pix_dec, pix_ra = healpy.pix2ang(nside, np.arange(npix))
    pix_dec = np.pi/2 - pix_dec  # healpy outputs co-latitude.
    diff_model = SkyModel(
        name=np.arange(npix).astype(str),
        stokes=diff_model.stokes,
        ra=Longitude(pix_ra*units.rad),
        dec=Latitude(pix_dec*units.rad),
        spectral_type="flat",
        component_type="point",
        frame="icrs",
        freq_array=np.array([freq])*units.Hz,
    )

    # Now actually simulate the visibilities.
    data_model = hera_sim.visibilities.ModelData(
        uvdata=diff_uvdata,
        sky_model=diff_model,
        beam_ids=beam_ids,
        beams=[beam,],
    )
    simulation = hera_sim.visibilities.VisibilitySimulation(
        data_model=data_model,
        simulator=hera_sim.visibilities.MatVis(),
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
        start_freq=150e6,
        channel_width=1.0,
        array_layout=ideal_array,
    )

    # Get the baselines we'll use for calibration.
    min_length = config.get("min_length", np.sqrt(2) * diameter)
    min_group_size = max(
        config.get("min_group_size", 5),
        config.get("n_eig", 1),
    )
    ant_1_array, ant_2_array, edges = corrcal.gridding.make_groups_from_uvdata(
        ideal_uvdata, min_bl_len=min_length, min_group_size=min_group_size
    )

    # Compute diffuse matrix. First, some auxiliary things.
    enu_antpos = uvdata.get_ENU_antpos()[0]
    ideal_antpos = ideal_uvdata.get_ENU_antpos()[0]
    baselines = enu_antpos[ant_1_array] - enu_antpos[ant_2_array]
    uvws = baselines / wavelength

    # Construct the AltAz frame for coordinate transformations.
    observatory = EarthLocation(longitude, latitude, altitude)
    local_frame = AltAz(location=observatory, obstime=Time(obstime, format="jd"))
    
    # Prepare to compute diffuse matrices.
    n_eig = 3
    ideal_n_eig = 1

    # Now compute the diffuse matrices.
    diff_mat = corrcal.models._compute_diffuse_matrix_from_flat_sky(
        sky_pspec,
        nside,
        beam,
        freq,
        enu_antpos,
        ant_1_array,
        ant_2_array,
        edges,
        n_eig=n_eig,
        pixwgt_datapath=args.pix_wgts,
    )
    ideal_diff_mat = corrcal.models._compute_diffuse_matrix_from_flat_sky(
        sky_pspec,
        nside,
        beam,
        freq,
        ideal_antpos,
        ant_1_array,
        ant_2_array,
        edges,
        n_eig=ideal_n_eig,
        pixwgt_datapath=args.pix_wgts,
    )

    # Rescale the diffuse matrix amplitude to mimic redcal.
    # ideal_diff_mat *= 1e5

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

    # Now convert that to an observed flux with matvis Stokes convention.
    flux_err = np.random.normal(
        loc=1, scale=config.get("flux_err", 0), size=src_flux.size
    )
    src_fluxes = 0.5 * src_flux * src_beam_vals * flux_err

    # Now figure out which sources to keep for calibration.
    n_src = config.get("n_src", 1)
    sort = np.argsort(src_fluxes)[::-1]  # Sort from highest to lowest flux.
    src_az = src_az[sort][:n_src]
    src_za = src_za[sort][:n_src]
    src_flux = src_fluxes[sort][:n_src]

    # Finally, actually compute the source matrix.
    uvws = freq * (
        enu_antpos[ant_2_array] - enu_antpos[ant_1_array]
    ) / constants.c.si.value
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
    src_mat[1::2] = -src_vis.imag

    # Simulate gains.
    gain_amp = config.get("gain_amp", 1)
    amp_jitter = config.get("amp_jitter", 0.1)
    n_ants = enu_antpos.shape[0]
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

    # Compute noise-related stuff.
    snr = config.get("snr", 20)
    fudge_factor = 25000
    snr = snr * fudge_factor
    integration_time = 1
    channel_width = snr ** 2
    omega_p = np.array([1])
    autocorr = uvdata.get_data(0, 0, "xx")
    noise_amp = np.abs(autocorr)[0,0] / snr
    noise = 0.5 * np.ones(2*ant_1_array.size, dtype=float) * noise_amp**2

    # Apply the noise to the data.
    sim = hera_sim.Simulator(data=uvdata.copy())
    sim.add(
        "thermal_noise",
        autovis=autocorr,
        integration_time=integration_time,
        channel_width=channel_width,
        Trx=0,
        omega_p=1,
    )

    # Now initialize the sparse covariance.
    full_cov = corrcal.sparse.SparseCov(
        noise=noise,
        src_mat=src_mat,
        diff_mat=diff_mat,
        edges=edges,
        n_eig=2*n_eig,
        isinv=False,
    )
    alt_cov = full_cov.copy()
    alt_cov.src_mat = np.zeros((edges[-1], 1), dtype=float)
    redcal_cov = alt_cov.copy()
    redcal_cov.diff_mat = ideal_diff_mat
    redcal_cov.n_eig = 2 * ideal_n_eig

    # Extra optimization parameters.
    gain_scale = 1
    phs_norm = 1

    # Retrieve the data and apply gains.
    ref_data = np.array(
        [
            uvdata.get_data(ai,aj,'xx')[0,0]
            for ai, aj in zip(ant_1_array, ant_2_array)
        ]
    )
    noisy_data = np.array(
        [
            sim.get_data(ai,aj,"xx")[0,0]
            for ai, aj in zip(ant_1_array, ant_2_array)
        ]
    ).flatten()
    data = noisy_data * true_gains[ant_1_array] * true_gains[ant_2_array].conj()

    # Separate the real and imaginary parts.
    split_data = np.zeros(2*data.size, dtype=float)
    split_data[::2] = data.real
    split_data[1::2] = data.imag
    
    # Run the minimizer for each scenario.
    opt_args = (
        full_cov, split_data, ant_1_array, ant_2_array, gain_scale, phs_norm
    )
    alt_args = (alt_cov,) + opt_args[1:]
    redcal_args = (redcal_cov,) + opt_args[1:]

    result = minimize(
        corrcal.optimize.nll,
        gain_scale*split_gains,
        args=opt_args,
        method="CG",
        jac=corrcal.optimize.grad_nll,
        options={"maxiter": args.maxiter},
    )
    alt_result = minimize(
        corrcal.optimize.nll,
        gain_scale*split_gains,
        args=alt_args,
        method="CG",
        jac=corrcal.optimize.grad_nll,
        options={"maxiter": args.maxiter},
    )
    redcal_result = minimize(
        corrcal.optimize.nll,
        gain_scale*split_gains,
        args=redcal_args,
        method="CG",
        jac=corrcal.optimize.grad_nll,
        options={"maxiter": args.maxiter},
    )

    # Now run HERA's redundant calibration.
    data_container = {}
    for ai, aj, vis in zip(ant_1_array, ant_2_array, data):
        data_container[(ai,aj,'nn')] = np.atleast_2d(vis)

    reds = [
        [
            (ai,aj,'nn')
            for ai, aj in zip(ant_1_array[start:stop], ant_2_array[start:stop])
        ] for start, stop in zip(edges//2, edges[1:]//2)
    ]

    g0 = {}
    for ant, gain in enumerate(init_gains):
        g0[(ant,'Jnn')] = np.atleast_2d(gain)

    v0 = {}
    for grp, (start, stop) in enumerate(zip(edges//2, edges[1:]//2)):
        _v0 = ref_data[start:stop].mean()
        v0[reds[grp][0]] = np.atleast_2d(_v0)

    rc = hera_cal.redcal.RedundantCalibrator(reds)
    sol0 = hera_cal.redcal.RedSol(reds, gains=g0, vis=v0)
    sol1 = rc.logcal(data_container, sol0=sol0)[1]
    redcal_meta, sol2 = rc.omnical(data_container, sol1)
    redcal_gains = np.zeros(n_ants, dtype=complex)
    for ant in range(n_ants):
        redcal_gains[ant] = sol2[(ant,'Jnn')][0,0]
    

    # Write visibilities, solutions, and stats to disk
    outdir = Path(args.outdir)
    basename = Path(args.config).stem
    np.savez(
        outdir / f"{basename}_results.npz",
        noisy_vis=noisy_data,
        ref_vis=ref_data,
        init_gains=init_gains,
        true_gains=true_gains,
        ref_soln=result.x[::2]+1j*result.x[1::2],
        alt_soln=alt_result.x[::2]+1j*alt_result.x[1::2],
        redcal_soln=redcal_result.x[::2]+1j*redcal_result.x[1::2],
        hera_redcal_soln=redcal_gains,
        ref_stats=np.array([result.nit, result.nfev, result.njev]),
        alt_stats=np.array([alt_result.nit, alt_result.nfev, alt_result.njev]),
        redcal_stats=np.array(
            [redcal_result.nit, redcal_result.nfev, redcal_result.njev]
        ),
        hera_redcal_info=np.array(list(redcal_meta.values())),
        ant_1_array=ant_1_array,
        ant_2_array=ant_2_array,
        edges=edges,
        enu_antpos=enu_antpos,
        src_mat=src_mat,
        diff_mat=diff_mat,
        ideal_diff_mat=ideal_diff_mat,
        noise=noise,
    )
