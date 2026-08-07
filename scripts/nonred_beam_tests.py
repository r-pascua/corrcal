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
import matvis
import yaml

from astropy import constants, units
from astropy.coordinates import Latitude, Longitude, AltAz
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pathlib import Path
from pyradiosky import SkyModel
from scipy.optimize import minimize
from time import time
from corrcal.beams import TiltedGaussianBeam

def rectangular_array(n_row, n_col, ew_sep=6.3, ns_sep=8.5):
    array_layout = {}
    ant_num = 0
    for row in range(n_row):
        for col in range(n_col):
            array_layout[ant_num] = np.array([col*ew_sep, row*ns_sep, 0])
            ant_num += 1
    return array_layout

def prepare_split_data(uvdata: UVData, gains: np.ndarray, ant_1_array: np.ndarray, ant_2_array: np.ndarray,  pol: str | int,)-> np.ndarray:
    noisy_data = np.array(
        [
            uvdata.get_data(ai,aj,pol)[0,0]
            for ai, aj in zip(ant_1_array, ant_2_array)
        ]
    ).flatten()
    data = noisy_data * gains[ant_1_array] * gains[ant_2_array].conj()
    split_data = np.zeros(edges[-1], dtype=float)
    split_data[::2] = data.real 
    split_data[1::2] = data.imag
    return noisy_data, data, split_data

parser = argparse.ArgumentParser("CLI for CorrCal beam nonredundancy tests.")
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
    default="/home/wendy/research/chord_project/source_catalogs",
    help="Where the sky model files are saved.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader = yaml.SafeLoader) 

    # Sky settings
    freq = config.get("freq", 300e6)
    wavelength = constants.c.si.value / freq
    diameter = config.get("diameter", 6)

    # Configure observatory parameters and observing time.
    obstime = config.get("obstime", 2459917.737631866) 
    latitude = config.get("latitude", 49.3210)
    longitude = config.get("longitude", -119.6218)
    altitude = config.get("altitude", 545.671)
    n_row = config.get("n_row", 4)
    n_col = config.get("n_col", 4)
    array_layout = rectangular_array(n_row, n_col, ew_sep=6.3, ns_sep=8.5)
    
    # Initialize empty UVData objects for storing simulated data.
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

    # Create point source sky model.
    src_dir = Path(args.src_dir)
    field_type = config.get("field_type", "calib")
    realization = config.get("realization", "00")
    src_file = src_dir / f"{field_type}_fields/catalog_{realization}.npz"
    src_info = dict(np.load(src_file))
    ra = Longitude(
        (uvdata.lst_array[0] + src_info["dra"]) * units.rad
    ).value
    dec = Latitude(
        (np.radians(latitude) + src_info["ddec"]) * units.rad
    ).value
    fluxes = src_info["fluxes"]  # Jy

    stokes = np.zeros((4, 1, fluxes.size), dtype=float)
    # TODO: graduate source model to skyh5 file
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
    # Generate diffuse sky model from a power spectrum  
    diffuse_sky_random_seed = config.get(
        "diffuse_sky_random_seed",
        None,
    )
    max_Tsky = config.get("max_Tsky", 500)
    nside = config.get("nside", 64)
    
    ell_max = 3*nside - 1
    ells = np.arange(ell_max + 1, dtype=float)
    sky_pspec = (1+ells)**config.get("pspec_idx", -2)
    
    state = np.random.get_state()
    np.random.seed(diffuse_sky_random_seed)
    Tsky = healpy.synfast(sky_pspec, nside)
    np.random.set_state(state)
    
    monopole = np.abs(Tsky.min())
    Tsky += monopole  # Make the diffuse signal nonnegative.
    rescaling = max_Tsky / Tsky.max()
    Tsky *= rescaling 

    K_to_Jy = 1 / hera_sim.utils.jansky_to_kelvin(freq/1e9, 1) 
    sky_pspec[0] += monopole**2 
    sky_pspec *= (rescaling * K_to_Jy / 2) ** 2 

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
    
    # Generate the perturbed beams for simulating visibilities.
    beam_param_names = [
        "sig_x",
        "sig_y",
        "ref_freq",
        "spec_ind",
        "feed_rotation_angle",
        "inclination_angle",
        "elevation_axis_rotation",
        "theta_x_offset",
        "theta_y_offset",
    ]
    
    conversions = {
        name: 1.0 if name in ("ref_freq", "spec_ind") 
        else units.deg.to(units.rad)
        for name in beam_param_names
    }
    beam_params = {
        "ideal": {
            "sig_x": (
                config["beam_params"].get("sig_x_mean", 1.0)
                * units.deg.to(units.rad)
            ),
            "sig_y": (
                config["beam_params"].get("sig_y_mean", 1.0)
                * units.deg.to(units.rad)
            ),
            "ref_freq": config["beam_params"].get(
                "ref_freq_mean", 300e6
            ),
            "spec_ind": config["beam_params"].get(
            "spec_ind_mean", 0.0
            ),
            "feed_rotation_angle": (
                config["beam_params"].get(
                    "feed_rotation_angle_mean", 0.0
                )
                * units.deg.to(units.rad)
            ),
            "inclination_angle": (
                config["beam_params"].get(
                    "inclination_angle_mean", 5.0
                )
                * units.deg.to(units.rad)
            ),
            "elevation_axis_rotation": (
                config["beam_params"].get(
                    "elevation_axis_rotation_mean", 0.0
                )
                * units.deg.to(units.rad)
            ),
            "theta_x_offset": (
                config["beam_params"].get(
                    "theta_x_offset_mean", 0.0
                )
                * units.deg.to(units.rad)
            ),
            "theta_y_offset": (
                config["beam_params"].get(
                    "theta_y_offset_mean", 0.0
                )
                * units.deg.to(units.rad)
            ),
        }
    }
    for ant in array_layout.keys():
        beam_params[ant] = {
            param: np.random.normal(
                loc=beam_params["ideal"][param],
                scale=config["beam_params"].get(f"{param}_std", 0) * conv,
            ) for param, conv in conversions.items()
        }
    
    model_beam = TiltedGaussianBeam(**beam_params["ideal"])
    data_beams = [
        TiltedGaussianBeam(**params)
        for name, params in beam_params.items()
        if name != "ideal"
    ]

    beam_ids = {str(ant): idx for idx, ant in enumerate(array_layout.keys())} 
    ideal_beam_ids = [0,] * len(array_layout)
    
    #Simulate point source visibilities with perturbed beam 
    data_model = hera_sim.visibilities.ModelData(
        uvdata=uvdata,
        sky_model=src_model,
        beam_ids=beam_ids,
        beams=data_beams,
    )
    simulation = hera_sim.visibilities.VisibilitySimulation(
        data_model=data_model,
        simulator=hera_sim.visibilities.MatVis(),
    )
    simulation.simulate()
    
    #Simulate diffuse visibilities with perturbed beam
    data_model = hera_sim.visibilities.ModelData(
        uvdata=diff_uvdata,
        sky_model=diff_model,
        beam_ids=beam_ids,
        beams=data_beams,
    )
    simulation = hera_sim.visibilities.VisibilitySimulation(
        data_model=data_model,
        simulator=hera_sim.visibilities.MatVis(),
    )
    simulation.simulate()

    # Sum the data together.
    uvdata.data_array += diff_uvdata.data_array

    # Simulate unperturbed visibilities.
    ideal_uvdata = hera_sim.io.empty_uvdata(
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
    ideal_diff_uvdata = ideal_uvdata.copy()

    ideal_data_model = hera_sim.visibilities.ModelData(
        uvdata=ideal_uvdata,
        sky_model=src_model,
        beam_ids=ideal_beam_ids,
        beams=[model_beam,],
    )
    simulation = hera_sim.visibilities.VisibilitySimulation(
        data_model=ideal_data_model,
        simulator=hera_sim.visibilities.MatVis(),
    )
    simulation.simulate()

    ideal_data_model = hera_sim.visibilities.ModelData(
        uvdata=ideal_diff_uvdata,
        sky_model=diff_model, 
        beam_ids=ideal_beam_ids,
        beams=[model_beam,],
    )
    simulation = hera_sim.visibilities.VisibilitySimulation(
        data_model=ideal_data_model,
        simulator=hera_sim.visibilities.MatVis(),
    )
    simulation.simulate()
    ideal_uvdata.data_array += ideal_diff_uvdata.data_array

    # Prepare to compute model covariance.
    min_length = config.get("min_length", np.sqrt(2) * diameter)
    min_group_size = max(
        config.get("min_group_size", 5),
        config.get("n_eig", 1),
    )

    ant_1_array, ant_2_array, edges = corrcal.gridding.make_groups_from_uvdata(
        uvdata, min_bl_len=min_length, min_group_size=min_group_size
    )
    
    enu_antpos = uvdata.get_enu_data_ants()[0]
    baselines = enu_antpos[ant_2_array] - enu_antpos[ant_1_array]

    observatory = EarthLocation(uvdata.telescope.location)
    local_frame = AltAz(
        location=observatory, obstime=Time(obstime, format="jd")
    )

    n_eig = config.get("n_eig", 1)

    diff_mat = corrcal.models._compute_diffuse_matrix_from_flat_sky(
        sky_pspec,  
        nside,
        model_beam,
        freq,
        enu_antpos,
        ant_1_array,
        ant_2_array,
        edges,
        n_eig=n_eig,
    )

    source_positions = SkyCoord(
        Longitude(ra*units.rad), Latitude(dec*units.rad), frame="icrs"
    ).transform_to(local_frame)

    above_horizon = source_positions.alt > 0
    dish_pointing = SkyCoord(
        az=(
            beam_params["ideal"]["elevation_axis_rotation"]
            * units.rad
        ),
        alt=(
            np.pi / 2
            - beam_params["ideal"]["inclination_angle"]
        ) * units.rad,
        frame=local_frame,
    )
    source_separation = source_positions.separation(dish_pointing)
    near_dish_pointing = (source_separation < 10 * units.deg)
    # TODO: set brightness threshold from config.
    is_bright = fluxes > 100 
    select = near_dish_pointing | (is_bright & above_horizon)
    src_az = Longitude(np.pi/2*units.rad - source_positions.az)[select].to(units.rad).value
    src_za = np.pi/2 - source_positions.alt.rad[select]
    src_flux = fluxes[select]

    beam_vals = model_beam.power_eval(
        az_array=src_az, za_array=src_za, freq_array=np.array([freq]),
    )[0][0,0]
    weighted_fluxes = beam_vals * src_flux
    
   # Now pick the n_src brightest observed sources for calibration.
    n_src = config.get("n_src", 1)
    sort = np.argsort(weighted_fluxes)[::-1]
    src_az = src_az[sort][:n_src]
    src_za = src_za[sort][:n_src]

    src_flux = 0.5 * weighted_fluxes[sort][:n_src]

    src_nhat = np.array(
        [np.sin(src_za)*np.cos(src_az), np.sin(src_za)*np.sin(src_az), np.cos(src_za)]
    )

    uvws = baselines / wavelength
    src_fringe = np.exp(-2j * np.pi * uvws @ src_nhat)

    src_vis = src_flux[None,:] * src_fringe
    src_mat = np.zeros((edges[-1], n_src), dtype=float)
    src_mat[::2] = src_vis.real
    src_mat[1::2] = src_vis.imag

    # Generate the true gains.
    gain_amp = config.get("gain_amp", 1) 
    amp_jitter = config.get("amp_jitter", 0.1) 
    n_ants = uvdata.Nants_data 
    gain_amplitudes = np.random.normal(size=n_ants, loc=gain_amp, scale=amp_jitter)
    gain_phases = np.random.uniform(0, 2*np.pi, n_ants)
    true_gains = gain_amplitudes * np.exp(1j*gain_phases)

    # Generate initial guess of true gains and split into real and imaginary.
    amp_error = config.get("amp_error", 0.1) 
    phs_error = config.get("phs_error", 0.1)
    init_amp = gain_amplitudes * np.random.normal(size=n_ants, loc=1, scale=amp_error)
    init_phs = gain_phases + np.random.normal(size=n_ants, loc=0, scale=phs_error)
    init_gains = init_amp * np.exp(1j*init_phs)
    split_gains = np.zeros(2*n_ants, dtype=float) 
    split_gains[::2] = init_gains.real
    split_gains[1::2] = init_gains.imag

    # Setting up noise parameters
    snr = config.get("snr", 20)
    fudge_factor = 25000
    snr = snr * fudge_factor
    integration_time = 1 
    channel_width = snr ** 2 
    omega_p = np.array([1])
    autocorr = uvdata.get_data(0, 0, "xx") 
    noise_amp = np.abs(autocorr)[0,0] / snr
    noise = 0.5 * np.ones(2*ant_1_array.size, dtype=float) * noise_amp**2

    # Add noise realization.
    sim = hera_sim.Simulator(data=uvdata.copy())
    sim.add(
        "thermal_noise",
        autovis=autocorr,
        integration_time=integration_time,
        channel_width=channel_width,
        Trx=0,
        omega_p=1,
        seed="once",
    )

    ideal_sim = hera_sim.Simulator(data=ideal_uvdata.copy())
    ideal_sim._seeds = sim._seeds.copy()
    ideal_sim.add(
        "thermal_noise",
        autovis=autocorr,
        integration_time=integration_time,
        channel_width=channel_width,
        Trx=0,
        omega_p=1,
        seed="once"
    )
    
    # Create model covariance object.
    cov = corrcal.sparse.SparseCov(
        noise=noise,
        diff_mat=diff_mat,
        src_mat=src_mat,
        edges=edges,
        n_eig=2*n_eig,
        isinv=False,
    )

    # Split the data into alternating real/imaginary components 
    noisy_data, data, split_data = prepare_split_data(sim, true_gains, ant_1_array, ant_2_array, "xx")
    
    ideal_noisy_data, ideal_data, ideal_split_data = prepare_split_data(ideal_sim, true_gains, ant_1_array, ant_2_array, "xx")
    
    # Collect extra arguments required for optimization
    gain_scale = config.get("gain_scale", 1)  
    phs_norm = config.get("phs_norm", 1)
    
    opt_args = (
        cov, split_data, ant_1_array, ant_2_array, gain_scale, phs_norm
    )

    ideal_opt_args = (
        cov, ideal_split_data, ant_1_array, ant_2_array, gain_scale, phs_norm
    )

    # Run calibration
    result = minimize(
        corrcal.optimize.nll,
        gain_scale*split_gains,
        args=opt_args,
        method="CG",  # Use the conjugate-gradient solver
        jac=corrcal.optimize.grad_nll,
    )

    ideal_result = minimize(
        corrcal.optimize.nll,
        gain_scale*split_gains,
        args=ideal_opt_args,
        method="CG",  # Use the conjugate-gradient solver
        jac=corrcal.optimize.grad_nll,
    )

    gain_soln = corrcal.utils.rephase_to_ant(
        result.x[::2] + 1j*result.x[1::2], 0
    )

    ideal_gain_soln = corrcal.utils.rephase_to_ant(
        ideal_result.x[::2] + 1j*ideal_result.x[1::2], 0
    )

    true_vis = np.array(
        [
            uvdata.get_data(ai,aj,'xx')[0,0]
            for ai, aj in zip(ant_1_array, ant_2_array)
        ]
    )
    ideal_vis = np.array(
        [
            ideal_uvdata.get_data(ai,aj,'xx')[0,0]
            for ai, aj in zip(ant_1_array, ant_2_array)
        ]
    )
    
    # Write calibration solutions, simulated data, and metadata to disk.
    outdir = Path(args.outdir)
    basename = Path(args.config).stem
    with open(outdir / f"{basename}_beam_params.yaml", "w") as f:
        yaml.dump(beam_params, f)
        
    np.savez(
        outdir / f"{basename}_results.npz",
        noisy_vis=noisy_data,
        ideal_noisy_data=ideal_noisy_data,
        ideal_vis=ideal_vis,
        ref_vis=true_vis,
        init_gains=init_gains,
        true_gains=true_gains,
        ref_soln=result.x[::2]+1j*result.x[1::2],
        ideal_gain_soln=ideal_result.x[::2]+1j*ideal_result.x[1::2],
        ref_stats=np.array([result.nit, result.nfev, result.njev]),
        ideal_ref_stats=np.array([ideal_result.nit, ideal_result.nfev, ideal_result.njev]),
        ant_1_array=ant_1_array,
        ant_2_array=ant_2_array,
        edges=edges,
        enu_antpos=enu_antpos,
        src_mat=src_mat,
        diff_mat=diff_mat,
        noise=noise,
        array_layout=array_layout
    )
