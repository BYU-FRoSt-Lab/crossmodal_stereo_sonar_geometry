import numpy as np
import matplotlib.pyplot as plt

import utils as ut
import input_checking as ic

def main():
    # Set up the configuration
    config = {
        # Forward-looking sonar config
        "fl_min_range": 0.1, # meters
        "fl_max_range": 10, # meters
        "fl_azimuth_aperture": 60, # degrees
        "fl_elevation_aperture": 10, # degrees
        "fl_elevation_resolution": 0.1, # degrees
        # Sidescan sonar config
        "ss_min_range": 0.1, # meters
        "ss_max_range": 30, # meters
        "ss_azimuth_aperture": 130, # degrees
        "ss_elevation_aperture": 0.3, # degrees
        # Feature config (values are with respect to the coordinate frame of the forward-looking sonar)
        "feat_range": 5, # meters (this must be within [fl_min_range, fl_max_range])
        "feat_azimuth": 0, # degrees (this must be within [-fl_azimuth_aperture/2, fl_azimuth_aperture/2])
        "feat_elevation": 0, # degrees (this must be within [-fl_elevation_aperture/2, fl_elevation_aperture/2])
        # Relative pose config (to simplify the use for the user, in this example we restrict the user-set config to be limited to the distance of the second sonar from the feature, as well as the relative orientation between the two sensors, and from there the relative translation is calculated and the feature is kept in the center of the field of view of the sensor sensor)
        "rp_range": 15, # meters (this must be within [ss_min_range, ss_max_range])
        "rp_azimuth": 0, # degrees (this must be within [-ss_azimuth_aperture/2, ss_azimuth_aperture/2])
        "rp_elevation": 0, # degrees (this must be within [-ss_elevation_aperture/2, ss_elevation_aperture/2])
        "rp_roll": 0, # degrees
        "rp_pitch": 90, # degrees
        "rp_yaw": 0, # degrees
    }
    # Run input checks on the configuration values
    config = ic.example_fl2ss_config_checks(config)
    # Calculate other config values that will be useful later
    config["fl_azimuth_min"] = -config["fl_azimuth_aperture"] / 2
    config["fl_azimuth_max"] = config["fl_azimuth_aperture"] / 2
    config["fl_elevation_min"] = -config["fl_elevation_aperture"] / 2
    config["fl_elevation_max"] = config["fl_elevation_aperture"] / 2
    config["ss_azimuth_min"] = -config["ss_azimuth_aperture"] / 2
    config["ss_azimuth_max"] = config["ss_azimuth_aperture"] / 2
    config["ss_elevation_min"] = -config["ss_elevation_aperture"] / 2
    config["ss_elevation_max"] = config["ss_elevation_aperture"] / 2
    # Set up an array that will hold the elevation values for the forward-looking sonar aperture at the user-specified resolution
    num_fl_elevation_vals = round(config["fl_elevation_aperture"] / config["fl_elevation_resolution"]) # Get integer number of values
    if not num_fl_elevation_vals % 2:
        num_fl_elevation_vals += 1 # Ensures that the number is odd so that we include the value of 0 as an elevation angle in the array (this is somewhat arbitrary, but is how we chose to set up the code that was used in the paper)
    fl_elevation_vals = np.linspace(config["fl_elevation_min"], config["fl_elevation_max"], num_fl_elevation_vals)
    # Calculate the rotation matrix and translation vector for the relative pose between sensors
    R = ut.rot_mtx_3d(config["rp_roll"], config["rp_pitch"], config["rp_yaw"], False, True)
    feature_point_fl = np.array([config["feat_range"], config["feat_azimuth"], config["feat_elevation"]])
    feature_point_ss = np.array([config["rp_range"], config["rp_azimuth"], config["rp_elevation"]])
    t = ut.calculate_translation_from_p1_p2_and_rot(feature_point_fl, feature_point_ss, R)
    t = -R @ t # adjust translation to be w.r.t. the second reference frame
    # Go through the process for projecting a feature    
    # Back project to 3D elevation arc in forward-looking field of view
    fl_frame_arc_pts = []
    for fl_elevation_val in fl_elevation_vals:
        fl_frame_arc_pts.append([config["feat_range"], config["feat_elevation"], fl_elevation_val])
    fl_frame_arc_pts = np.asarray(fl_frame_arc_pts).T # Convert to a numpy array and orient as 3xN
    # Transform the spherical coordinates into the reference frame of the sidescan sonar, with the output also in spherical coordinates
    ss_frame_arc_pts = ut.transform_reference_frame_spherical_pts(fl_frame_arc_pts, R, t)
    # Clip to the points that are within the field-of-view of the sidescan sonar
    ss_frame_arc_pts_in_view = []
    for ss_frame_arc_pt in ss_frame_arc_pts.T:
        temp_range, temp_azimuth, temp_elevation = ss_frame_arc_pt
        if not config["ss_min_range"] <= temp_range <= config["ss_max_range"]:
            continue
        elif not abs(temp_azimuth) <= config["ss_azimuth_max"]:
            continue
        elif not abs(temp_elevation) <= config["ss_elevation_max"]:
            continue
        else:
            ss_frame_arc_pts_in_view.append(ss_frame_arc_pt)
    ss_frame_arc_pts_in_view = np.asarray(ss_frame_arc_pts_in_view).T
    # Project the points to the sidescan measurement (only retain range)
    ss_meas = ss_frame_arc_pts_in_view[0]
    # Set up a plot of the result
    fig, ax = plt.subplots()
    ax.plot([config["ss_min_range"], config["ss_max_range"]], [0, 0], c='gray')
    ax.scatter(ss_meas, np.zeros_like(ss_meas), c='r', zorder=10)
    # Show the plot
    plt.show()
    return

if __name__ == "__main__":
    main()