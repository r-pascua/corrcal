import numpy as np
from pyuvdata.utils.types import FloatArray, IntArray, StrArray
from dataclasses import dataclass
from pyuvdata.analytic_beam import AnalyticBeam

@dataclass(kw_only=True)
class TiltedGaussianBeam(AnalyticBeam):
    """
    A polarized Gaussian beam tilted away from zenith along an elevation axis.

    Attributes
    ----------
    sig_x : float
        Standard deviation parameter for the beam along the "x" feed's axis, in
        radians.
    sig_y : float
        Standard deviation parameter for the beam along the "y" feed's axis, in
        radians.
    ref_freq : float
        Frequency, in Hz, at which the beam has the provided standard deviation
        parameters.
    spec_ind : float
        Spectral index governing the evolution of the beam size with frequency.
    feed_rotation_angle : float
        How much the feed blades are rotated from being coincident with the
        cardinal directions, in radians and measured counter-clockwise when
        viewing the feed from boresight.
    inclination_angle : float
        How far away from zenith the dish has been tilted, in radians. Positive
        values indicate a northward tilt.
    elevation_axis_rotation : float
        Angle between the elevation axis orientation and ICRS north, in radians
        and measured counter-clockwise when viewed from above.
    theta_x_offset : float
        Offset between dish boresight and actual boresight in the image plane
        x-direction, in radians.
    theta_y_offset : float
        Offset between dish boresight and actual boresight in the image plane
        y-direction, in radians.


    Parameters
    ----------
    sig_x : float
        Standard deviation parameter for the beam along the "x" feed's axis, in
        radians.
    sig_y : float
        Standard deviation parameter for the beam along the "y" feed's axis, in
        radians.
    ref_freq : float
        Frequency, in Hz, at which the beam has the provided standard deviation
        parameters.
    spec_ind : float
        Spectral index governing the evolution of the beam size with frequency.
    feed_rotation_angle : float
        How much the feed blades are rotated from being coincident with the
        cardinal directions, in radians and measured counter-clockwise when
        viewing the feed from boresight.
    inclination_angle : float
        How far away from zenith the dish has been tilted, in radians. Positive
        values indicate a northward tilt.
    elevation_axis_rotation : float
        Angle between the elevation axis orientation and ICRS north, in radians
        and measured counter-clockwise when viewed from above.
    theta_x_offset : float
        Offset between dish boresight and actual boresight in the image plane
        x-direction, in radians.
    theta_y_offset : float
        Offset between dish boresight and actual boresight in the image plane
        y-direction, in radians.
    """

    sig_x: float = np.pi / 180
    sig_y: float = np.pi / 180
    ref_freq: float = 150e6
    spec_ind: float = 0
    feed_rotation_angle: float = 0
    inclination_angle: float = 0
    elevation_axis_rotation: float = 0
    theta_x_offset: float = 0
    theta_y_offset: float = 0

    def _efield_eval(
        self, *, az_grid: FloatArray, za_grid: FloatArray, f_grid:FloatArray
    ) -> FloatArray:
        data_array = self._get_empty_data_array(az_grid.shape, beam_type="efield")

        # Rotation sending north -> elevation axis orientation.
        cos_gamma = np.cos(self.elevation_axis_rotation)
        sin_gamma = np.sin(self.elevation_axis_rotation)
        R1 = np.array(
            [
                [cos_gamma, sin_gamma, 0],
                [-sin_gamma, cos_gamma, 0],
                [0, 0, 1]
            ]
        )

        # Rotation sending up -> dish boresight
        cos_i = np.cos(self.inclination_angle)
        sin_i = np.sin(self.inclination_angle)
        R2 = np.array(
            [
                [1, 0, 0],
                [0, cos_i, -sin_i],
                [0, sin_i, cos_i]
            ]
        )
        
        # Rotation from dish frame to feed frame.
        cos_psi = np.cos(self.feed_rotation_angle)
        sin_psi = np.sin(self.feed_rotation_angle)
        R3 = np.array(
            [
                [cos_psi, sin_psi],
                [-sin_psi, cos_psi],
            ]
        )

        # Construct the rotation from local enu to dish xyz.
        R = R2 @ R1

        # Define the inverse covariance for each polarization.
        C_inv = np.array([[1/self.sig_x**2, 0],[0, 1/self.sig_y**2]])
        C_inv_X = R3.T @ C_inv @ R3
        C_inv_Y = R3.T @ np.diag(np.diag(C_inv)[::-1]) @ R3

        # Compute the beam argument at the reference frequency, then broadcast.
        az = az_grid[0]
        za = za_grid[0]
        enu_rhat = np.array(
            [np.sin(za)*np.cos(az), np.sin(za)*np.sin(az), np.cos(za)]
        )

        # Convert enu direction vectors to dish-frame xyz direction vectors.
        xyz_rhat = R @ enu_rhat
        theta = np.array(
            [
                np.arcsin(xyz_rhat[0])-self.theta_x_offset,
                np.arcsin(xyz_rhat[1])-self.theta_y_offset,
            ]
        )

        # Populate the data array.
        for feed in range(self.Nfeeds):
            Cinv = (C_inv_X, C_inv_Y)[feed]
            beam_arg = -0.5 * np.einsum("xp,xy,yp->p", theta, Cinv, theta)
            data_array[:,feed,:,:] = np.exp(
                beam_arg[np.newaxis,:] * (f_grid/self.ref_freq)**(-2*self.spec_ind)
            )[np.newaxis] / np.sqrt(2)  # Peak-normalized power beam

        return data_array
