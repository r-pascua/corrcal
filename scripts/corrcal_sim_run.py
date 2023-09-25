import argparse
import corrcal
import healpy
import hera_sim
import numpy as np
import vis_cpu
import yaml

from astropy import constants, units
from astropy.coordinates import Longitude, Latitude
from pathlib import Path
from pyradiosky import SkyModel
from pyuvdata import UVBeam
from pyuvsim import AnalyticBeam
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

if __name__ == "__main__":
    args = parser.parse_args()
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
        ant = 0
        raise NotImplementedError("too tired")
    elif isinstance(array_layout, str):
        raise ValueError("Array is unknown.")
    else:
        if not isinstance(array_layout, dict):
            raise ValueError("Array layout not understood.")
        diameter = config.get("diameter", 14)

    # Keep track of this for later.
    ideal_array = array_layout.copy()

    # Add jitter to antenna positions if requested
    jitter = config.get("jitter", 0)
    if jitter > 0:
        wavelength = constants.c.si.value / freq
        for ant, pos in array_layout.items():
            dpos = np.random.normal(loc=0, scale=jitter*wavelength, size=3)
            dpos[-1] = 0
            array_layout[ant] = pos + dpos

    # Extract other information from the config file.
    beam = AnalyticBeam(config.get("beam", "airy"), diameter=diameter)
    beam_ids = [0,] * len(array_layout)
    freq = config.get("freq", 150e6)
    obstime = config.get("obstime", 2459917.737631866)
    latitude = config.get("latitude", -30.721526120689507)
    longitude = config.get("longitude", 21.428303826863015)
    altitude = config.get("altitude", 1051.690000018105)

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
        (latitude*units.deg.to(units.rad) + src_info["ddec"]) * units.rad
    ).value
    fluxes = src_info["fluxes"]  # Jy

    # Simulate the visibilities.
    stokes = np.zeros((4, 1, fluxes.size), dtype=float)
    src_model = SkyModel(
        name=np.arange(fluxes.size).astype(str),
        ra=Longitude(ra*units.rad),
        dec=Latitude(dec*units.rad),
        stokes=stokes*units.Jy,
        spectral_type="flat",
        component_type="point",
    )
    data_model = hera_sim.visibilities.ModelData(
        uvdata=uvdata,
        sky_model=src_model,
        beam_ids=beam_ids,
        beams=[beam,],
    )
    simulation = hera_sim.visibilities.VisibilitySimulation(
        data_model=data_model,
        simulator=hera_sim.visibilities.VisCPU(),
    )
    simulation.simulate()

    # Generate diffuse sky model.
    # This scaling makes it so the maximum flux in any pixel of the diffuse
    # map will have a flux density of ~90 K at 150 MHz for nside=64.
    max_Tsky = config.get("max_Tsky", 500)
    nside = config.get("nside", 64)
    ell_max = 3*nside - 1
    ells = np.arange(ell_max + 1, dtype=float)
    sky_pspec = 1 / (1+ells)**config.get("pspec_idx", -2)
    Tsky = healpy.synfast(sky_pspec, nside)
    Tsky += np.abs(Tsky.min())  # Make sure map temperature is non-negative.
    Tsky *= max_Tsky / Tsky.max()

    # Simulate the visibilities.
    stokes = np.zeros((4, 1, Tsky.size), dtype=float)
    stokes[0,0] = Tsky
    diff_model = SkyModel(
        stokes=stokes*units.Jy/units.sr,
        spectral_type="flat",
        component_type="healpix",
        nside=nside,
        hpx_inds=np.arange(Tsky.size),
    )
    data_model = hera_sim.visibilities.ModelData(
        uvdata=diff_uvdata,
        sky_model=diff_model,
        beam_ids=beam_ids,
        beams=[beam,],
    )
    simulation = hera_sim.visibilities.VisibilitySimulation(
        data_model=data_model,
        simulator=hera_sim.visibilities.VisCPU(),
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
    reds, _, lens, conj = uvdata.get_redundancies(include_conjugates=True)
    conj = set(conj)
    autocorr = uvdata.get_data(0, 0, "xx")
    del ideal_uvdata

    # Get the baselines we'll use for calibration.
    min_length = config.get("min_length", np.sqrt(2) * diameter)
    min_group_size = max(config.get("min_group_size", 5), config["n_eig"])
    ant_1_array = []
    ant_2_array = []
    edges = [0,]
    idx = 0
    for group, length in zip(reds, lens):
        if (length <= min_length) or (len(group) < min_group_size):
            continue
        for bl in group:
            ai, aj = uvdata.baseline_to_antnums(bl)
            if bl in conj:
                ai, aj = aj, ai
            ant_1_array.append(ai)
            ant_2_array.append(aj)
            idx += 1
        edges.append(idx)
    ant_1_array = np.asarray(ant_1_array)
    ant_2_array = np.asarray(ant_2_array)
    edges = np.asarray(edges)

    # Compute diffuse matrix. First, some auxiliary things.
    beam.efield_to_power()
    enu_antpos = uvdata.get_ENU_antpos()[0]
    za, az = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
    
    # Now estimate the angular power spectrum from the diffuse sky model.
    sky_power = healpy.anafast(Tsky, use_pixel_weights=True)

    # Evaluate the beam at every pixel on the sky.
    beam_vals = beam.interp(
        az_array=Longitude((az - np.pi/2)*units.rad).value,
        za_array=za,
        freq_array=uvdata.freq_array.flatten(),
    )[0][0,0,0,0]

    # Enforce a horizon cut, then rotate to ECEF frame.
    beam_vals[za>np.pi/2] = 0
    lst = uvdata.lst_array[0]
    beam_alms = healpy.map2alm(beam_vals, use_pixel_weights=True)
    healpy.rotate_alm(beam_alms, 0, np.pi/2-latitude, lst)
    beam_vals = healpy.alm2map(beam_alms, nside)

    # Now compute the fringe pattern on the sky for every baseline.
    sky_crd = np.array(
        [np.sin(za)*np.cos(az), np.sin(za)*np.sin(az), np.cos(za)]
    )  # Unit vector pointing to each pixel on the sky.
    rotation = vis_cpu.conversions.enu_to_eci_matrix(lst, latitude)
    ecef_antpos = (rotation @ enu_antpos.T).T
    uvws = freq * (
        ecef_antpos[ant_2_array] - ecef_antpos[ant_1_array]
    ) / constants.c.si.value
    fringe = np.exp(-2j * np.pi * uvws @ sky_crd)

    # Actually compute the diffuse matrix now.
    n_eig = config["n_eig"]
    diff_mat = np.zeros((edges[-1], n_eig), dtype=complex)
    if (jitter == 0) and (n_eig == 1):
        scaling = 2*ells + 1

    for grp in range(edges.size-1):
        start, stop = edges[grp:grp+2]
        if (jitter == 0) and (n_eig == 1):
            # There's no non-redundancy, so we don't need to waste time
            # with eigenvalue decomposition.
            fringed_beam = beam_vals * fringe[start]
            beam_spectrum = healpy.anafast(
                fringed_beam, use_pixel_weights=True
            )
            cov_amp = np.sum(sky_power * scaling * beam_spectrum)
            diff_mat[start:stop,0] = cov_amp
        else:
            # We have some non-redundancy, so we'll fill out the upper-
            # triangular region of each quasi-redundant block, then take
            # the eigendecomposition and keep only n_eig modes.
            block_size = stop - start
            block = np.zeros((block_size, block_size), dtype=complex)
            for m, i in enumerate(range(start, stop)):
                b_i = healpy.map2alm(
                    beam_vals*fringe[i],
                    use_pixel_weights=True,
                    lmax=sky_power.size-1,
                )
                for n, j in enumerate(range(i, stop)):
                    b_j = healpy.map2alm(
                        beam_vals*fringe[j],
                        use_pixel_weights=True,
                        lmax=sky_power.size-1,
                    )
                    block[m,m+n] = healpy.almxfl(
                        b_i*b_j.conj(), sky_power
                    ).sum()
            
            # Compute the eigenvalues/vectors and sort in decreasing order.
            eigvals, eigvecs = np.linalg.eigh(block, "U")
            if eigvals[0] < eigvals[-1]:
                eigvals = eigvals[::-1]
                eigvecs = eigvecs[:,::-1]
            diff_mat[start:stop] = eigvecs[:,:n_eig] * np.sqrt(
                eigvals[None,:n_eig]
            )

    del fringe, uvws, sky_crd

    # Compute source matrix. First, figure out "local" source positions.
    src_xyz = vis_cpu.conversions.point_source_crd_eq(ra, dec)
    rotation = vis_cpu.conversions.eci_to_enu_matrix(lst, latitude)
    src_enu = rotation @ src_xyz
    az, za = vis_cpu.conversions.enu_to_az_za(
        src_enu[0], src_enu[1], orientation="uvbeam"
    )

    # Now make a horizon cut.
    above_horizon = za < np.pi/2
    if above_horizon.sum() < above_horizon.size:
        az = az[above_horizon]
        za = za[above_horizon]
        src_enu = src_enu[:,above_horizon]
        fluxes = fluxes[above_horizon]
        ra = ra[above_horizon]
        dec = dec[above_horizon]

    # Compute the beam at each source position.
    uvws = freq * (
        enu_antpos[ant_2_array] - enu_antpos[ant_1_array]
    ) / constants.c.si.value
    src_beam_vals = beam.interp(
        az_array=az,
        za_array=za,
        freq_array=np.array([freq]),
    )[0][0,0,0,0]

    # Now convert that to an observed flux with vis_cpu Stokes convention.
    src_fluxes = 0.5 * fluxes * src_beam_vals

    # Now figure out which sources to keep for calibration.
    n_src = config.get("n_src", 1)
    sort = np.argsort(src_fluxes)[::-1]  # Sort from highest to lowest flux.
    select = sort[:n_src]

    # Finally, actually compute the source matrix.
    flux_err = np.random.normal(
        loc=1, scale=config.get("flux_err", 0), size=n_src
    )
    phases = 2 * np.pi * uvws @ src_enu[:,select]
    src_mat = (flux_err*src_fluxes)[None,select] * np.exp(1j * phases)

    # Simulate gains and generate initial guess.
    gain_amp_err = config.get("gain_amp_err", 0.05)
    gain_phs_err = config.get("gain_phs_err", 0.02)
    n_ants = len(array_layout)
    gain_amp = np.random.normal(loc=1, size=n_ants, scale=gain_amp_err)
    gain_phs = np.random.normal(loc=0, size=n_ants, scale=gain_phs_err)
    phases = np.random.uniform(0, 2*np.pi, n_ants)
    true_gains = np.exp(1j * phases)
    fit_gains = gain_amp * true_gains * np.exp(1j * gain_phs)
    split_gains = np.zeros(2*n_ants, dtype=float)
    split_gains[::2] = fit_gains.real
    split_gains[1::2] = fit_gains.imag

    # Compute noise-related stuff. Anecdotally, SNR on the crosses is about
    # 5-20% of SNR on the autos. The SNR we're setting in the config is the
    # desired SNR in the crosses.
    snr = config.get("snr", 20)
    fudge_factor = 1 / 0.07
    snr *= fudge_factor
    integration_time = 1
    channel_width = snr ** 2
    omega_p = np.array([1])
    noise_amp = np.abs(autocorr)[0,0] / snr
    noise = np.ones(ant_1_array.size, dtype=complex) * noise_amp

    # Now initialize the sparse covariance.
    cov = corrcal.sparse.SparseCov(
        noise=noise,
        src_mat=src_mat,
        diff_mat=diff_mat,
        edges=edges,
        n_eig=n_eig,
        isinv=False,
    )
    gain_scale = 1
    phs_norm = 0.1

    # Initialize arrays for storing useful data and metadata.
    n_realizations = config.get("n_noise_realizations", 1)
    gain_solutions = np.zeros((n_realizations, n_ants), dtype=complex)
    noisy_visibilities = np.zeros(
        (n_realizations, ant_1_array.size), dtype=complex
    )
    solver_stats = {"nit": [], "nfev": [], "njev": [], "walltime": []}

    # Loop over noise realizations, calibrating and taking note of results
    for i in range(n_realizations):
        # Simulate some noise.
        sim = hera_sim.Simulator(data=uvdata.copy())
        sim.add(
            "thermal_noise",
            integration_time=integration_time,
            channel_width=channel_width,
            omega_p=omega_p,
            seed="initial",
            autovis=autocorr,
        )

        # Sort the data by redundancy.
        data = np.array(
            [
                sim.get_data(ai,aj,"xx")[0,0]
                for ai, aj in zip(ant_1_array, ant_2_array)
            ]
        )
        
        # Now let's keep track of the true visibilities, then apply gains.
        noisy_visibilities[i,:] = data.copy()
        data = data * true_gains[ant_1_array] * true_gains[ant_2_array].conj()
        
        # Run the minimizer, and track how long it takes.
        opt_args = (
            cov, data, ant_1_array, ant_2_array, gain_scale, phs_norm
        )
        t1 = time()
        result = minimize(
            corrcal.optimize.nll,
            gain_scale*split_gains,
            args=opt_args,
            method="CG",
            jac=corrcal.optimize.grad_nll,
        )
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
        init_gains=fit_gains,
        true_gains=true_gains,
        gain_sols=gain_solutions,
    )
    with open(outdir/f"{basename}_results.yaml", "w") as f:
        yaml.dump(solver_stats, f)
