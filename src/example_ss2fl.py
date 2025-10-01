import numpy as np
import matplotlib.pyplot as plt

import utils as ut
import input_checking as ic

def main():
    # Set up the configuration
    config = {
        # Sidescan sonar config
        "ss_min_range": 0.1, # meters
        "ss_max_range": 30, # meters
        "ss_azimuth_aperture": 130, # degrees
        "ss_azimuth_resolution": 0.1, # degrees
        "ss_elevation_aperture": 0.3, # degrees
        "ss_elevation_resolution": 0.1, # degrees
        # Forward-looking sonar config
        "fl_min_range": 0.1, # meters
        "fl_max_range": 10, # meters
        "fl_azimuth_aperture": 60, # degrees
        "fl_elevation_aperture": 10, # degrees
        # Feature config (values are with respect to the coordinate frame of the forward-looking sonar)
        "feat_range": 15, # meters (this must be within [ss_min_range, ss_max_range])
        "feat_azimuth": 0, # degrees (this must be within [-ss_azimuth_aperture/2, ss_azimuth_aperture/2])
        "feat_elevation": 0, # degrees (this must be within [-ss_elevation_aperture/2, ss_elevation_aperture/2])
        # Relative pose config (to simplify the use for the user, in this example we restrict the user-set config to be limited to the distance of the second sonar from the feature, as well as the relative orientation between the two sensors, and from there the relative translation is calculated and the feature is kept in the center of the field of view of the sensor sensor)
        "rp_range": 5, # meters (this must be within [fl_min_range, fl_max_range])
        "rp_azimuth": 0, # degrees (this must be within [-fl_azimuth_aperture/2, fl_azimuth_aperture/2])
        "rp_elevation": 0, # degrees (this must be within [-fl_elevation_aperture/2, fl_elevation_aperture/2])
        "rp_roll": 45, # degrees
        "rp_pitch": 0, # degrees
        "rp_yaw": 45, # degrees
    }
    # Run input checks on the configuration values
    config = ic.example_ss2fl_config_checks(config)
    # Calculate other config values that will be useful later
    config["ss_azimuth_min"] = -config["ss_azimuth_aperture"] / 2
    config["ss_azimuth_max"] = config["ss_azimuth_aperture"] / 2
    config["ss_elevation_min"] = -config["ss_elevation_aperture"] / 2
    config["ss_elevation_max"] = config["ss_elevation_aperture"] / 2
    config["fl_azimuth_min"] = -config["fl_azimuth_aperture"] / 2
    config["fl_azimuth_max"] = config["fl_azimuth_aperture"] / 2
    config["fl_elevation_min"] = -config["fl_elevation_aperture"] / 2
    config["fl_elevation_max"] = config["fl_elevation_aperture"] / 2
    # Calculate config values that will be used for area approximation of projection in forward-looking sonar image
    config["area_ymax"] = float(ut.spherical_to_cartesian(config["fl_max_range"], config["fl_azimuth_max"], 0)[1,0])
    config["area_ymin"] = -config["area_ymax"]
    config["area_xmax"] = config["fl_max_range"]
    config["area_xmin"] = float(ut.spherical_to_cartesian(config["fl_min_range"], config["fl_azimuth_max"], 0)[0,0])
    config["area_spacing"] = 0.08
    config["area_square"] = config["area_spacing"]**2
    # Set up arrays, one that holds the azimuth values, and the other for the elevation values, of the sidescan sonar aperture
    num_ss_azimuth_vals = round(config["ss_azimuth_aperture"] / config["ss_azimuth_resolution"]) # Get integer number of values
    if not num_ss_azimuth_vals % 2:
        num_ss_azimuth_vals += 1 # Ensures that the number is odd so that we include the value of 0 as an azimuth angle in the array (this is somewhat arbitrary, but is how we chose to set up the code that was used in the paper)
    ss_azimuth_vals = np.linspace(config["ss_azimuth_min"], config["ss_azimuth_max"], num_ss_azimuth_vals)
    num_ss_elevation_vals = round(config["ss_elevation_aperture"] / config["ss_elevation_resolution"]) # Get integer number of values
    if not num_ss_elevation_vals % 2:
        num_ss_elevation_vals += 1 # Ensures that the number is odd so that we include the value of 0 as an elevation angle in the array (this is somewhat arbitrary, but is how we chose to set up the code that was used in the paper)
    ss_elevation_vals = np.linspace(config["ss_elevation_min"], config["ss_elevation_max"], num_ss_elevation_vals)
    # Set up arrays to hold the x and y values used to set up a grid that will be used to approximate the projection area later
    area_yvals = np.arange(config["area_ymin"], config["area_ymax"], config["area_spacing"])
    area_xvals = np.arange(config["area_xmin"], config["area_xmax"], config["area_spacing"])
    # Calculate the rotation matrix and translation vector for the relative pose between sensors
    R = ut.rot_mtx_3d(config["rp_roll"], config["rp_pitch"], config["rp_yaw"], False, True)
    feature_point_fl = np.array([config["feat_range"], config["feat_azimuth"], config["feat_elevation"]])
    feature_point_ss = np.array([config["rp_range"], config["rp_azimuth"], config["rp_elevation"]])
    t = ut.calculate_translation_from_p1_p2_and_rot(feature_point_fl, feature_point_ss, R)
    t = -R @ t # adjust translation to be w.r.t. the second reference frame
    # Go through the process for projecting a feature    
    # Back project to 3D surface in sidescan field of view
    ss_frame_feat_surface_pts = []
    for ss_azimuth_val in ss_azimuth_vals:
        for ss_elevation_val in ss_elevation_vals:
            ss_frame_feat_surface_pts.append([config["feat_range"], ss_azimuth_val, ss_elevation_val])
    ss_frame_feat_surface_pts = np.asarray(ss_frame_feat_surface_pts).T # Convert to a numpy array and orient as 3xN
    # Transform the spherical coordinates into the reference frame of the forward-looking sonar, with the output also in spherical coordinates
    fl_frame_feat_surface_pts = ut.transform_reference_frame_spherical_pts(ss_frame_feat_surface_pts, R, t)
    # Clip to the points that are within the field-of-view of the forward-looking sonar
    fl_frame_feat_surface_pts_in_view = []
    for fl_frame_surface_pt in fl_frame_feat_surface_pts.T:
        temp_range, temp_azimuth, temp_elevation = fl_frame_surface_pt
        if not config["fl_min_range"] <= temp_range <= config["fl_max_range"]:
            continue
        elif not abs(temp_azimuth) <= config["fl_azimuth_max"]:
            continue
        elif not abs(temp_elevation) <= config["fl_elevation_max"]:
            continue
        else:
            fl_frame_feat_surface_pts_in_view.append(fl_frame_surface_pt)
    fl_frame_feat_surface_pts_in_view = np.asarray(fl_frame_feat_surface_pts_in_view).T
    # Project the points to the forward-looking measurement (range and azimuth)
    fl_meas_spherical = fl_frame_feat_surface_pts_in_view[:2]
    fl_meas_cartesian = ut.spherical_to_cartesian(*fl_meas_spherical, np.zeros_like(fl_meas_spherical[0]))[:2]
    # Set up a plot of the result
    fig, ax = plt.subplots()
    ut.plot_polar_fov_2d(ax, config["fl_min_range"], config["fl_max_range"], config["fl_azimuth_aperture"], 5, 5, True)
    ax.scatter(fl_meas_cartesian[1], fl_meas_cartesian[0], c='r', zorder=10)
    ax.set_aspect("equal")
    # Show the plot
    plt.show()
    return

if __name__ == "__main__":
    main()