import numpy as np
import numpy.typing as npt
import matplotlib.axes as mpl_axes
import typing

import input_checking as ic

def spherical_to_cartesian(r: typing.Union[float, npt.NDArray],
                           theta: typing.Union[float, npt.NDArray],
                           phi: typing.Union[float, npt.NDArray]):
    '''
    Convert spherical coordinate(s), with angles in degrees, to their Cartesian representation.
    For spherical coordiantes we assume that theta originates from the x-axis, and phi from the x-y plane.

    :param r: the range value(s) in meters
    :type r: float or 1d numpy array
    :param theta: the azimuth, or theta, angle(s) in degrees
    :type theta: float or 1d numpy array
    :param phi: the elevation, or phi, angle(s) in degrees
    :type phi: float or 1d numpy array
    :returns: an array with the Cartesian coordinates, with the x values in the first row, y in the second row, and z in the third row
    :rtype: 3xN numpy array
    '''
    # Run the input checks to catch any erroneous input values
    r, theta, phi = ic.spherical_to_cartesian_checks(r, theta, phi)
    # We use degrees as the input for angles, as we believe it's easier to interpret those numbers for humans, but need to use radians for calculations, so we convert the angles to radians
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    # Convert the spherical coordinate(s) to Cartesian coordinate(s)
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    # Put everything into a numpy array of the desired output dimensions (3 x N, where N is the number of points)
    cartesian_coords = np.block([[x], [y], [z]])
    # Return the output
    return cartesian_coords

def cartesian_to_spherical(x: typing.Union[float, npt.NDArray],
                           y: typing.Union[float, npt.NDArray],
                           z: typing.Union[float, npt.NDArray]):
    '''
    Convert Cartesian coordinate(s) to their spherical coordinate representation, with angles in degrees.
    For the spherical coordiante output we assume that theta originates from the x-axis, and phi from the x-y plane.

    :param x: the x coordinate(s)
    :type x: float or 1d numpy array
    :param y: the y coordinate(s)
    :type y: float or 1d numpy array
    :param z: the z coordinate(s)
    :type z: float or 1d numpy array
    :returns: an array with the spherical coordinates, with the range value(s) in the first row, theta angle(s) in the second row, and phi angle(s) in the third row
    :rtype: 3xN numpy array
    '''
    # Run input checks
    x, y, z = ic.cartesian_to_spherical_checks(x, y, z)
    # Calculate the range value(s)
    r = np.sqrt(x**2 + y**2 + z**2)
    # Calculate the theta value(s) -- azimuth angle(s)
    theta = np.arctan2(y, x)
    # Calculate the phi value(s) -- elevation angle(s), though if we have any range values of 0 it will throw an error, so we ensure any range values of are set to 1e-10 so that the calculations run without errors
    if isinstance(r, np.ndarray):
        rtemp = r.copy()
        rtemp[np.where(rtemp == 0)] = 1e-10
    else:
        if r == 0:
            rtemp = 1e-10
        else:
            rtemp = r
    phi = np.arcsin(z / rtemp)
    # Convert the angles from radians to degrees
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)
    # Set up the output as a numpy array with the desired dimensions (3 x N, where N is the number of points)
    spherical_coords = np.block([[r], [theta], [phi]])
    # Return the result
    return spherical_coords

def rot_mtx_3d(rx: float, 
               ry: float, 
               rz: float, 
               active_rotation: bool=False, 
               check_angle_bounds: bool=True):
    '''
    Constructs a 3D rotation matrix in the rotation order of yaw, then pitch, then roll.
    The input angles should all be in degrees, and the output can be for an active or passive rotation, depending on the purpose.

    :param rx: the roll, or rotation about the x-axis, in degrees
    :type rx: float
    :param ry: the pitch, or rotation about the y-axis, in degrees
    :type ry: float
    :param rz: the yaw, or rotation about the z-axis, in degrees
    :type rz: float
    :param active_rotation: whether or not the rotation is active or passive. Defaults to False, which corresponds with a passive rotation
    :type active_rotation: bool
    :param check_angle_bounds: whether or not to bound the angles, which should be done most of the time, but in some cases may be useful to disable
    :type check_angle_bounds: bool
    :returns: A 3x3 numpy array with the rotation matrix that encodes the roll, pitch, and yaw
    :rtype: numpy array
    '''
    # Run input checks
    rx, ry, rz, active_rotation, check_angle_bounds = ic.rot_mtx_3d_checks(rx, ry, rz, active_rotation, check_angle_bounds)
    # When enabled, ensure that the angles are within the desired bounds
    if check_angle_bounds:
        if not -180 < rx <= 180:
            raise ValueError("utils.rot_mtx_3d: input parameter 'rx' must be within value range of (-180, 180] degrees")
        if not -90 < ry <= 90:
            raise ValueError("utils.rot_mtx_3d: input parameter 'ry' must be within value range of (-90, 90] degrees")
        if not -180 < rz <= 180:
            raise ValueError("utils.rot_mtx_3d: input parameter 'rz' must be within value range of (-180, 180] degrees")
    # Convert the angles to radians 
    rx_rad = np.deg2rad(rx)
    ry_rad = np.deg2rad(ry)
    rz_rad = np.deg2rad(rz)
    # Construct the individual (active) rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx_rad), -np.sin(rx_rad)],
                   [0, np.sin(rx_rad), np.cos(rx_rad)]], float)
    Ry = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)],
                   [0, 1, 0],
                   [-np.sin(ry_rad), 0, np.cos(ry_rad)]], float)
    Rz = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0], 
                   [np.sin(rz_rad), np.cos(rz_rad), 0], 
                   [0, 0, 1]], float)
    # Compose into the final (active) rotation matrix
    R = Rz @ Ry @ Rx
    # If the passive rotation is desired, convert from active to passive (via transpose)
    if not active_rotation:
        R = R.T
    # Return the 3x3 rotation matrix
    return R

def calculate_translation_from_p1_p2_and_rot(p1: npt.NDArray, 
                                             p2: npt.NDArray, 
                                             R: npt.NDArray):
    '''
    Calculates the relative translation given:
    1. The desired coordinate in the first reference frame
    2. The desired coordinate in the second reference frame
    3. The relative passive rotation between the two reference frames 

    :param p1: the point, in spherical coordinates, in the first reference frame
    :type p1: numpy array
    :param p2: the point, in spherical coordinates, in the second reference frame
    :type p2: numpy array
    :param R: the passive rotation matrix from the first to the second reference frames
    :typs R: numpy array
    :returns: a 3x1 numpy array, or vector, with the relative translation between the two frames
    :rtype: numpy array
    '''
    # Run input checks
    p1, p2, R = ic.calculate_translation_from_p1_p2_and_rot_checks(p1, p2, R)
    # Convert the first and second points to Cartesian coordinates
    p1_cartesian = spherical_to_cartesian(*p1)
    p2_cartesian = spherical_to_cartesian(*p2)
    # Calculate the translation
    t = p1_cartesian - R.T @ p2_cartesian
    # Return the translation vector
    return t

def transform_reference_frame_spherical_pts(pts: npt.NDArray,
                                            R: npt.NDArray,
                                            t: npt.NDArray):
    '''
    Takes a spherical coordinate and transforms it to a spherical coordinate in a new reference frame

    :param pt: The 3D spherical coordinate(s) to be transformed, input as a 3xN array, where N is the number of points
    :type pt: numpy array
    :param R: the 3x3 rotation matrix
    :type R: numpy array
    :param t: the 3x1 translation vector
    :type t: numpy array
    :returns: a 3xN array of the transformed point(s) in spherical coordinates
    :rtype: numpy array
    '''
    # Run input checks
    pts, R, t = ic.transform_reference_frame_spherical_pts_checks(pts, R, t)
    # Convert from spherical to Cartesian coordinates
    p1_cartesian = spherical_to_cartesian(*pts)
    # Transform to the new reference frame
    p2_cartesian = R @ p1_cartesian + t
    # Convert from Cartesian to spherical coordinates
    p2_spherical = cartesian_to_spherical(*p2_cartesian)
    # Return the transformed spherical coordinates
    return p2_spherical

def plot_polar_fov_2d(ax: mpl_axes.Axes,
                      r_min: float,
                      r_max: float,
                      theta_aperture: float,
                      num_range_lines: int=0,
                      num_theta_lines: int=0,
                      rotate_north: bool=True):
    '''
    Draws the field of view (FOV) of a sonar sensor in the 2D (polar) plane.

    :param ax: the matplotlib axes on which to plot the field-of-view
    :type ax: matplotlib axes
    :param r_min: the minimum range for the sonar field-of-view, in meters
    :type r_min: float
    :param r_max: the maximum range for the sonar field-of-view, in meters
    :type r_max: float
    :param theta_aperture: the azimuth aperture, also represented by the angle theta, of the sensor, in degrees
    :type theta_aperture: float
    :param num_range_lines: the number of lines (arcs) to draw within the field-of-view, visually indicating different range values
    :type num_range_lines: int
    :param num_theta_lines: the number of lines to draw within the field-of-view, visually indicating different theta, or azimuth, values
    :type num_theta_lines: int
    :param rotate_north: whether or not to rotate the visual to be oriented with range in the north direction, rather than the matplotlib default of range oriented to the east
    :type rotate_north: bool
    '''
    # Run input check
    ax, r_min, r_max, theta_aperture, num_range_lines, num_theta_lines, rotate_north = ic.plot_polar_fov_2d_checks(ax, r_min, r_max, theta_aperture, num_range_lines, num_theta_lines, rotate_north)
    if rotate_north:
        R = np.array([[0, -1], [1, 0]])
    else:
        R = np.eye(2)
    r_vals = np.array([r_min, r_max])
    theta_max = theta_aperture / 2
    theta_min = -theta_max
    theta_vals = np.linspace(theta_min, theta_max, 100)
    # Draw the minimum and maximum range arcs
    min_range_arc = R @ spherical_to_cartesian(r_vals[0]*np.ones_like(theta_vals), theta_vals, np.zeros_like(theta_vals))[:2]
    ax.plot(*min_range_arc, c='k')
    max_range_arc = R @ spherical_to_cartesian(r_vals[1]*np.ones_like(theta_vals), theta_vals, np.zeros_like(theta_vals))[:2]
    ax.plot(*max_range_arc, c='k')
    # Draw the minimum and maximum theta lines
    min_theta_line = R @ spherical_to_cartesian(r_vals, theta_min*np.ones_like(r_vals), np.zeros_like(r_vals))[:2]
    ax.plot(*min_theta_line, c='k')
    max_theta_line = R @ spherical_to_cartesian(r_vals, theta_max*np.ones_like(r_vals), np.zeros_like(r_vals))[:2]
    ax.plot(*max_theta_line, c='k')
    # If appropriate, draw the intermediate range arcs and theta lines
    if num_range_lines > 0:
        itrm_range_vals = np.linspace(r_min, r_max, num_range_lines+2)[1:-1]
        for rval in itrm_range_vals:
            itrm_range_arc = R @ spherical_to_cartesian(rval*np.ones_like(theta_vals), theta_vals, np.zeros_like(theta_vals))[:2]
            ax.plot(*itrm_range_arc, c='k', ls='dashed')
    if num_theta_lines > 0:
        itrm_theta_vals = np.linspace(theta_min, theta_max, num_theta_lines+2)[1:-1]
        for tval in itrm_theta_vals:
            itrm_theta_line = R @ spherical_to_cartesian(r_vals, tval*np.ones_like(r_vals), np.zeros_like(r_vals))[:2]
            ax.plot(*itrm_theta_line, c='k', ls='dashed')
    return

if __name__ == "__main__":
    print("This file has no main function, and instead consists of many functions that can be used in other scripts")