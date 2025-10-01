import numpy as np
import numpy.typing as npt
import matplotlib.axes as mpl_axes
import typing

def spherical_to_cartesian_checks(r: typing.Union[float, npt.NDArray],
                                  theta: typing.Union[float, npt.NDArray],
                                  phi: typing.Union[float, npt.NDArray]):
    # Check the range input(s)
    if isinstance(r, (int, float, np.integer, np.floating)):
        # Ensure that the range value is not negative
        if r < 0:
            raise ValueError("utils.spherical_to_cartesian: input parameter 'r' must be >= 0")
    elif isinstance(r, (list, np.ndarray)):
        r = np.asarray(r) # Ensure the input is an array (just in case a list was passed in)
        # Ensure that the dimensions of the input are appropriate (we want a 1d array)
        if len(r.shape) > 2:
            raise ValueError("utils.spherical_to_cartesian: input parameter 'r' has too many dimensions")
        elif len(r.shape) == 2:
            if np.any(np.asarray(r.shape) == 1):
                r = r.flatten() # if we have a 1xN array or an Nx1 array, but with two dimensions, flatten it to a 1d array
            else:
                raise ValueError("utils.spherical_to_cartesian: input parameter 'r' has too many dimensions")
        # Ensure that no negative range values are present
        if np.any(r < 0):
            raise ValueError("utils.spherical_to_cartesian: input parameter 'r' contains a negative value")
    else:
        raise TypeError("utils.spherical_to_cartesian: input parameter 'r' is of an invalid type")
    # Then check the theta input(s)
    if isinstance(theta, (list, np.ndarray)):
        theta = np.asarray(theta) # Ensure the input is an array (just in case a list was passed in)
        # Ensure that the dimensions of the input are appropriate (we want a 1d array)
        if len(theta.shape) > 2:
            raise ValueError("utils.spherical_to_cartesian: input parameter 'theta' has too many dimensions")
        elif len(theta.shape) == 2:
            if np.any(np.asarray(theta.shape) == 1):
                theta = theta.flatten() # if we have a 1xN array or an Nx1 array, but with two dimensions, flatten it to a 1d array
            else:
                raise ValueError("utils.spherical_to_cartesian: input parameter 'theta' has too many dimensions")
    elif not isinstance(theta, (int, float, np.integer, np.floating)):
        raise TypeError("utils.spherical_to_cartesian: input parameter 'theta' is of an invalid type")
    # Then check the phi input(s)
    if isinstance(phi, (int, float, np.integer, np.floating)):
        # Ensure that the phi value is within the necessary bounds
        if phi < -90 or phi > 90:
            raise ValueError("utils.spherical_to_cartesian: input parameter 'phi' must be within range [-90, 90] degrees")
    elif isinstance(phi, (list, np.ndarray)):
        phi = np.asarray(phi) # Ensure the input is an array (just in case a list was passed in)
        # Ensure that the dimensions of the input are appropriate (we want a 1d array)
        if len(phi.shape) > 2:
            raise ValueError("utils.spherical_to_cartesian: input parameter 'phi' has too many dimensions")
        elif len(phi.shape) == 2:
            if np.any(np.asarray(phi.shape) == 1):
                phi = phi.flatten() # if we have a 1xN array or an Nx1 array, but with two dimensions, flatten it to a 1d array
            else:
                raise ValueError("utils.spherical_to_cartesian: input parameter 'phi' has too many dimensions")
        # Ensure that all values are within the necessary bounds
        if np.any(phi < -90) or np.any(phi > 90):
            raise ValueError("utils.spherical_to_cartesian: input parameter 'phi' contains at least one value outside of the range [-90, 90] degrees")
    else:
        raise TypeError("utils.spherical_to_cartesian: input parameter 'phi' is of an invalid type")
    # Lastly, make sure that the dimensions and types between the three inputs are consistent
    if isinstance(r, (int, float, np.integer, np.floating)) and isinstance(theta, (int, float, np.integer, np.floating)) and isinstance(phi, (int, float, np.integer, np.floating)):
        return r, theta, phi
    elif isinstance(r, np.ndarray) and isinstance(theta, np.ndarray) and isinstance(phi, np.ndarray):
        if len(r) == len(theta) == len(phi):
            return r, theta, phi
        else:
            raise ValueError("utils.spherical_to_cartesian: dimensions of input parameters 'r', 'theta', and 'phi', do not match")
    else:
        raise TypeError("utils.spherical_to_cartesian: input types for input parameters 'r', 'theta', and 'phi' are valid, but mismatched")
    
def cartesian_to_spherical_checks(x: typing.Union[float, npt.NDArray],
                                  y: typing.Union[float, npt.NDArray],
                                  z: typing.Union[float, npt.NDArray]):
    # Check that each input is of a valid type and dimension
    if isinstance(x, (list, np.ndarray)):
        x = np.asarray(x) # Ensure the input is an array (just in case a list was passed in)
        # Ensure that the dimensions of the input are appropriate (we want a 1d array)
        if len(x.shape) > 2:
            raise ValueError("utils.cartesian_to_spherical: input parameter 'x' has too many dimensions")
        elif len(x.shape) == 2:
            if np.any(np.asarray(x.shape) == 1):
                x = x.flatten() # if we have a 1xN array or an Nx1 array, but with two dimensions, flatten it to a 1d array
            else:
                raise ValueError("utils.cartesian_to_spherical: input parameter 'x' has too many dimensions")
    elif not isinstance(x, (int, float, np.integer, np.floating)):
        raise TypeError("utils.cartesian_to_spherical: input parameter 'x' is of an invalid type")
    if isinstance(y, (list, np.ndarray)):
        y = np.asarray(y) # Ensure the input is an array (just in case a list was passed in)
        # Ensure that the dimensions of the input are appropriate (we want a 1d array)
        if len(y.shape) > 2:
            raise ValueError("utils.cartesian_to_spherical: input parameter 'y' has too many dimensions")
        elif len(y.shape) == 2:
            if np.any(np.asarray(y.shape) == 1):
                y = y.flatten() # if we have a 1xN array or an Nx1 array, but with two dimensions, flatten it to a 1d array
            else:
                raise ValueError("utils.cartesian_to_spherical: input parameter 'y' has too many dimensions")
    elif not isinstance(y, (int, float, np.integer, np.floating)):
        raise TypeError("utils.cartesian_to_spherical: input parameter 'y' is of an invalid type")
    if isinstance(z, (list, np.ndarray)):
        z = np.asarray(z) # Ensure the input is an array (just in case a list was passed in)
        # Ensure that the dimensions of the input are appropriate (we want a 1d array)
        if len(z.shape) > 2:
            raise ValueError("utils.cartesian_to_spherical: input parameter 'z' has too many dimensions")
        elif len(z.shape) == 2:
            if np.any(np.asarray(z.shape) == 1):
                z = z.flatten() # if we have a 1xN array or an Nx1 array, but with two dimensions, flatten it to a 1d array
            else:
                raise ValueError("utils.cartesian_to_spherical: input parameter 'z' has too many dimensions")
    elif not isinstance(z, (int, float, np.integer, np.floating)):
        raise TypeError("utils.cartesian_to_spherical: input parameter 'z' is of an invalid type")
    # Then ensure that the types are all consistent, and if they are arrays that the dimensions match
    if isinstance(x, (int, float, np.integer, np.floating)) and isinstance(y, (int, float, np.integer, np.floating)) and isinstance(z, (int, float, np.integer, np.floating)):
        return x, y, z
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray):
        if len(x) == len(y) == len(z):
            return x, y, z
        else:
            raise ValueError("utils.cartesian_to_spherical: dimensions of input parameters 'x', 'y', and 'z', do not match")
    else:
        raise TypeError("utils.cartesian_to_spherical: input types for input parameters 'x', 'y', and 'y' are valid, but mismatched")
    
def rot_mtx_3d_checks(rx: typing.Union[float, np.number], 
                      ry: typing.Union[float, np.number], 
                      rz: typing.Union[float, np.number], 
                      active_rotation: bool, 
                      check_angle_bounds: bool) -> typing.Tuple[float, float, float, bool, bool]:
    '''
    Performs type checking for the input parameters of a 3D rotation matrix function.

    Ensures that the roll (rx), pitch (ry), and yaw (rz) angles are valid 
    numeric types (float, int, or numpy numeric), and that the flags 
    (active_rotation, check_angle_bounds) are booleans.

    :param rx: The roll angle (x-axis rotation).
    :type rx: float or numpy numeric type
    :param ry: The pitch angle (y-axis rotation).
    :type ry: float or numpy numeric type
    :param rz: The yaw angle (z-axis rotation).
    :type rz: float or numpy numeric type
    :param active_rotation: Flag indicating active or passive rotation.
    :type active_rotation: bool
    :param check_angle_bounds: Flag to enable or disable angle bounding checks.
    :type check_angle_bounds: bool
    :returns: The validated input parameters, with numeric types converted to standard floats.
    :rtype: Tuple[float, float, float, bool, bool]
    '''
    
    func_name = "utils.rot_mtx_3d"
    # Define valid numeric types (standard Python int/float and any NumPy numeric type)
    valid_numeric_types = (int, float, np.number)
    
    # --- Check Numeric Angles (rx, ry, rz) ---
    
    # 1. Check rx
    if not isinstance(rx, valid_numeric_types):
        raise TypeError(f"{func_name}: input parameter 'rx' is of an invalid type. Expected float or numeric.")
    
    # 2. Check ry
    if not isinstance(ry, valid_numeric_types):
        raise TypeError(f"{func_name}: input parameter 'ry' is of an invalid type. Expected float or numeric.")

    # 3. Check rz
    if not isinstance(rz, valid_numeric_types):
        raise TypeError(f"{func_name}: input parameter 'rz' is of an invalid type. Expected float or numeric.")
    
    # --- Check Boolean Flags (active_rotation, check_angle_bounds) ---

    # 4. Check active_rotation
    # Explicitly checking for 'bool' avoids issues where a non-zero integer 
    # might pass an 'isinstance(x, (bool, int))' check but is not intended.
    if not isinstance(active_rotation, bool):
        raise TypeError(f"{func_name}: input parameter 'active_rotation' is of an invalid type. Expected bool.")
        
    # 5. Check check_angle_bounds
    if not isinstance(check_angle_bounds, bool):
        raise TypeError(f"{func_name}: input parameter 'check_angle_bounds' is of an invalid type. Expected bool.")

    # --- Type Conversion for Consistency ---
    # Convert any NumPy numeric types back to standard Python floats for cleaner
    # downstream use and adherence to the function's float type hint.
    if isinstance(rx, np.number):
        rx = float(rx)
    if isinstance(ry, np.number):
        ry = float(ry)
    if isinstance(rz, np.number):
        rz = float(rz)
        
    return rx, ry, rz, active_rotation, check_angle_bounds

def calculate_translation_from_p1_p2_and_rot_checks(p1: typing.Union[list, npt.NDArray], 
                                                    p2: typing.Union[list, npt.NDArray], 
                                                    R: typing.Union[list, npt.NDArray]) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Performs type and dimension checking for the input parameters of 
    calculate_translation_from_p1_p2_and_rot.

    Ensures p1 and p2 are convertible to 3x1 vectors and R is a 3x3 matrix.

    :param p1: The point in the first reference frame.
    :type p1: list or numpy array (must contain 3 elements)
    :param p2: The point in the second reference frame.
    :type p2: list or numpy array (must contain 3 elements)
    :param R: The passive rotation matrix.
    :type R: list or numpy array (must be 3x3)
    :returns: The validated input parameters as numpy arrays (p1 and p2 are 3x1, R is 3x3).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    '''
    
    func_name = "utils.calculate_translation_from_p1_p2_and_rot"
    valid_array_types = (list, np.ndarray)

    def _validate_vector_input(arr, param_name):
        if not isinstance(arr, valid_array_types):
            raise TypeError(f"{func_name}: input parameter '{param_name}' is of an invalid type. Expected list or numpy array.")
        
        arr_np = np.asarray(arr, dtype=float)

        # Check for number of elements (must be exactly 3)
        if arr_np.size != 3:
            raise ValueError(f"{func_name}: input parameter '{param_name}' must contain exactly 3 numerical elements. Found {arr_np.size}.")
        
        # Reshape to (3, 1) vector for consistent linear algebra operations
        if arr_np.shape != (3, 1):
            try:
                arr_np = arr_np.reshape((3, 1))
            except ValueError:
                # This should not happen if arr_np.size == 3, but is a robust guardrail
                raise ValueError(f"{func_name}: input parameter '{param_name}' cannot be reshaped to a (3, 1) vector.")
                
        return arr_np

    # 1. Check p1
    p1_validated = _validate_vector_input(p1, 'p1')

    # 2. Check p2
    p2_validated = _validate_vector_input(p2, 'p2')

    # 3. Check R (Rotation Matrix)
    if not isinstance(R, valid_array_types):
        raise TypeError(f"{func_name}: input parameter 'R' is of an invalid type. Expected list or numpy array.")
        
    R_validated = np.asarray(R, dtype=float)

    # Check for 3x3 shape
    if R_validated.shape != (3, 3):
        # A 1D list/array of 9 elements is technically valid input for np.asarray
        # but if it was not 3x3 originally, we check if it can be reshaped to 3x3
        if R_validated.size == 9:
             R_validated = R_validated.reshape((3, 3))
        else:
             raise ValueError(f"{func_name}: input parameter 'R' must be a 3x3 rotation matrix. Found shape {R_validated.shape}.")

    return p1_validated, p2_validated, R_validated

def transform_reference_frame_spherical_pts_checks(pts: typing.Union[list, npt.NDArray],
                                                   R: typing.Union[list, npt.NDArray],
                                                   t: typing.Union[list, npt.NDArray]) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Performs type and dimension checking for the input parameters of 
    transform_reference_frame_spherical_pts.

    Ensures pts is convertible to a 3xN matrix, R is a 3x3 matrix, and t is a 3x1 vector.

    :param pts: The 3D spherical coordinate(s) (3xN).
    :type pts: list or numpy array
    :param R: The 3x3 rotation matrix.
    :type R: list or numpy array
    :param t: The 3x1 translation vector.
    :type t: list or numpy array
    :returns: The validated input parameters as numpy arrays (pts is 3xN, R is 3x3, t is 3x1).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    '''
    
    func_name = "utils.transform_reference_frame_spherical_pts"
    valid_array_types = (list, np.ndarray)
    
    # Helper to validate and reshape 3x1 vector (used for t)
    def _validate_3x1_vector(arr, param_name):
        if not isinstance(arr, valid_array_types):
            raise TypeError(f"{func_name}: input parameter '{param_name}' is of an invalid type. Expected list or numpy array.")
        
        arr_np = np.asarray(arr, dtype=float)

        # Check for number of elements (must be exactly 3)
        if arr_np.size != 3:
            raise ValueError(f"{func_name}: input parameter '{param_name}' must contain exactly 3 numerical elements. Found {arr_np.size}.")
        
        # Reshape to (3, 1) vector
        if arr_np.shape != (3, 1):
            arr_np = arr_np.reshape((3, 1))
                
        return arr_np
        
    # --- 1. Check pts (3xN matrix) ---
    if not isinstance(pts, valid_array_types):
        raise TypeError(f"{func_name}: input parameter 'pts' is of an invalid type. Expected list or numpy array.")
        
    pts_validated = np.asarray(pts, dtype=float)

    if pts_validated.size == 0:
        raise ValueError(f"{func_name}: input parameter 'pts' cannot be empty.")
    
    if pts_validated.size % 3 != 0:
        raise ValueError(f"{func_name}: input parameter 'pts' must contain a multiple of 3 elements (3xN points). Found {pts_validated.size} elements.")
        
    num_points = pts_validated.size // 3
    target_shape = (3, num_points)

    if pts_validated.ndim == 1:
        # 1D array, reshape to (3, N)
        pts_validated = pts_validated.reshape(target_shape)
    elif pts_validated.ndim == 2:
        # 2D array, check shapes (3, N) or (N, 3)
        if pts_validated.shape == target_shape:
            # Already (3, N). Good.
            pass
        elif pts_validated.shape == (num_points, 3):
            # Is (N, 3), need to transpose to (3, N)
            pts_validated = pts_validated.T
        else:
            raise ValueError(f"{func_name}: input parameter 'pts' has an invalid 2D shape {pts_validated.shape}. Expected (3, N) or (N, 3).")
    else:
        raise ValueError(f"{func_name}: input parameter 'pts' has too many dimensions. Expected 1D or 2D array.")
        
    # --- 2. Check R (3x3 matrix) ---
    if not isinstance(R, valid_array_types):
        raise TypeError(f"{func_name}: input parameter 'R' is of an invalid type. Expected list or numpy array.")
        
    R_validated = np.asarray(R, dtype=float)

    # Check for 3x3 shape
    if R_validated.shape != (3, 3):
        # If the size is 9, attempt to reshape it (handles 1D list/array of 9 elements)
        if R_validated.size == 9:
             R_validated = R_validated.reshape((3, 3))
        else:
             raise ValueError(f"{func_name}: input parameter 'R' must be a 3x3 rotation matrix. Found shape {R_validated.shape}.")

    # --- 3. Check t (3x1 vector) ---
    t_validated = _validate_3x1_vector(t, 't')

    return pts_validated, R_validated, t_validated

def _validate_non_negative_numeric(val: typing.Union[float, np.number], 
                                   param_name: str, 
                                   func_name: str, 
                                   allow_zero: bool = True) -> float:
    '''Helper function to validate and convert a numeric value to float, ensuring it's non-negative.'''
    valid_numeric_types = (int, float, np.number)
    if not isinstance(val, valid_numeric_types):
        raise TypeError(f"{func_name}: input parameter '{param_name}' is of an invalid type. Expected float or numeric.")
    
    val_float = float(val)
    
    if not allow_zero and val_float <= 0:
        raise ValueError(f"{func_name}: input parameter '{param_name}' must be > 0. Found {val_float}.")
    elif val_float < 0:
        raise ValueError(f"{func_name}: input parameter '{param_name}' must be >= 0. Found {val_float}.")
        
    return val_float

def _validate_non_negative_integer(val: typing.Union[int, np.integer], 
                                   param_name: str, 
                                   func_name: str) -> int:
    '''Helper function to validate and convert a value to integer, ensuring it's non-negative.'''
    valid_integer_types = (int, np.integer)
    # Check if it's an integer type (disallows float, even if it's 5.0)
    if not isinstance(val, valid_integer_types) or isinstance(val, bool): # bool is subclass of int, so we exclude it
        raise TypeError(f"{func_name}: input parameter '{param_name}' is of an invalid type. Expected integer.")
        
    val_int = int(val)
    
    if val_int < 0:
        raise ValueError(f"{func_name}: input parameter '{param_name}' must be >= 0. Found {val_int}.")
        
    return val_int

def plot_polar_fov_2d_checks(ax: object, # mpl_axes.Axes expected, using object for self-contained check file
                             r_min: typing.Union[float, np.number],
                             r_max: typing.Union[float, np.number],
                             theta_aperture: typing.Union[float, np.number],
                             num_range_lines: typing.Union[int, np.integer],
                             num_theta_lines: typing.Union[int, np.integer],
                             rotate_north: bool) -> typing.Tuple[object, float, float, float, int, int, bool]:
    '''
    Performs type and value checking for the input parameters of 
    plot_polar_fov_2d.

    :param ax: The matplotlib axes object. (Type check skipped for file portability)
    :type ax: matplotlib axes
    :param r_min: The minimum range (>= 0).
    :type r_min: float
    :param r_max: The maximum range (> r_min, >= 0).
    :type r_max: float
    :param theta_aperture: The azimuth aperture (> 0).
    :type theta_aperture: float
    :param num_range_lines: The number of range lines (int, >= 0).
    :type num_range_lines: int
    :param num_theta_lines: The number of theta lines (int, >= 0).
    :type num_theta_lines: int
    :param rotate_north: Whether to orient North upwards (bool).
    :type rotate_north: boolean
    :returns: The validated parameters.
    :rtype: Tuple[matplotlib.axes, float, float, float, int, int, bool]
    '''
    func_name = "utils.plot_polar_fov_2d"
    
    # 1. Check ax (Matplotlib Axes object)
    if not isinstance(ax, mpl_axes.Axes):
        raise TypeError(f"{func_name}: input parameter 'ax' is of an invalid type. Expected matplotlib axes, but got {type(ax)}.")
    ax_validated = ax
    
    # 2. Check r_min, r_max, theta_aperture (non-negative floats with constraints)
    r_min_validated = _validate_non_negative_numeric(r_min, 'r_min', func_name, allow_zero=True)
    r_max_validated = _validate_non_negative_numeric(r_max, 'r_max', func_name, allow_zero=True)
    theta_aperture_validated = _validate_non_negative_numeric(theta_aperture, 'theta_aperture', func_name, allow_zero=False)

    # 3. Check Range Constraints (r_min < r_max)
    if r_min_validated >= r_max_validated:
        raise ValueError(f"{func_name}: input parameter 'r_min' ({r_min_validated}) must be strictly less than 'r_max' ({r_max_validated}).")

    # 4. Check num_range_lines, num_theta_lines (non-negative integers)
    num_range_lines_validated = _validate_non_negative_integer(num_range_lines, 'num_range_lines', func_name)
    num_theta_lines_validated = _validate_non_negative_integer(num_theta_lines, 'num_theta_lines', func_name)

    # 5. Check rotate_north (boolean)
    if not isinstance(rotate_north, bool):
        raise TypeError(f"{func_name}: input parameter 'rotate_north' is of an invalid type. Expected bool.")
        
    return (ax_validated, r_min_validated, r_max_validated, theta_aperture_validated, 
            num_range_lines_validated, num_theta_lines_validated, rotate_north)

def example_ss2fl_config_checks(config: dict):
    # Define valid numeric types (standard Python int/float and any NumPy numeric type)
    VALID_NUMERIC_TYPES = (int, float, np.number)

    def _get_key_float(config: dict, key: str, func_name: str) -> float:
        '''
        Helper to check if a key exists in the config, is numeric, and returns its value as a float.
        Raises ValueError or TypeError if invalid.
        '''
        if key not in config:
            raise ValueError(f"{func_name}: Missing required configuration key '{key}'.")
        
        val = config[key]
        if not isinstance(val, VALID_NUMERIC_TYPES):
            raise TypeError(f"{func_name}: Value for key '{key}' is of invalid type. Expected numeric, found {type(val)}.")
            
        return float(val)

    def _validate_range_value(config: dict, key: str, func_name: str) -> float:
        '''
        Checks that a numeric value is non-negative (>= 0).
        '''
        val = _get_key_float(config, key, func_name)
        if val < 0:
            raise ValueError(f"{func_name}: Value for key '{key}' must be non-negative (>= 0). Found {val}.")
        return val

    def _validate_aperture_or_resolution(config: dict, key: str, func_name: str) -> float:
        '''
        Checks that a numeric value is strictly positive (> 0).
        '''
        val = _get_key_float(config, key, func_name)
        if val <= 0:
            raise ValueError(f"{func_name}: Value for key '{key}' must be strictly positive (> 0). Found {val}.")
        return val

    '''
    Validates all constraints within the sonar and pose configuration dictionary for sidescan to forward-looking sonar

    The validated dictionary, with all values cast to float, is returned.

    :param config: The dictionary containing all sensor and pose parameters.
    :type config: dict
    :returns: The validated configuration dictionary with values cast to float.
    :rtype: dict
    :raises TypeError: If any value is not numeric or not a bool (for flags).
    :raises ValueError: If any value violates constraints (e.g., negative range, max < min).
    '''
    func_name = "example_ss2fl.main()"
    
    # We will build a validated configuration dictionary
    validated_config = {}
    
    # --- 1. Sidescan Sonar (SS) Checks ---
    
    # 1a. Ranges (>= 0)
    ss_min_range = _validate_range_value(config, "ss_min_range", func_name)
    ss_max_range = _validate_range_value(config, "ss_max_range", func_name)
    
    # 1b. Range Constraint (min < max)
    if ss_min_range >= ss_max_range:
        raise ValueError(f"{func_name}: 'ss_min_range' ({ss_min_range}) must be strictly less than 'ss_max_range' ({ss_max_range}).")
        
    # 1c. Apertures and Resolutions (> 0)
    ss_azimuth_aperture = _validate_aperture_or_resolution(config, "ss_azimuth_aperture", func_name)
    ss_azimuth_resolution = _validate_aperture_or_resolution(config, "ss_azimuth_resolution", func_name)
    ss_elevation_aperture = _validate_aperture_or_resolution(config, "ss_elevation_aperture", func_name)
    ss_elevation_resolution = _validate_aperture_or_resolution(config, "ss_elevation_resolution", func_name)

    # --- 2. Forward-Looking Sonar (FL) Checks ---
    
    # 2a. Ranges (>= 0)
    fl_min_range = _validate_range_value(config, "fl_min_range", func_name)
    fl_max_range = _validate_range_value(config, "fl_max_range", func_name)

    # 2b. Range Constraint (min < max)
    if fl_min_range >= fl_max_range:
        raise ValueError(f"{func_name}: 'fl_min_range' ({fl_min_range}) must be strictly less than 'fl_max_range' ({fl_max_range}).")

    # 2c. Apertures (> 0)
    fl_azimuth_aperture = _validate_aperture_or_resolution(config, "fl_azimuth_aperture", func_name)
    fl_elevation_aperture = _validate_aperture_or_resolution(config, "fl_elevation_aperture", func_name)

    # --- 3. Feature Configuration (feat_) Checks ---
    
    feat_range = _validate_range_value(config, "feat_range", func_name)
    feat_azimuth = _get_key_float(config, "feat_azimuth", func_name) # Can be negative/zero
    feat_elevation = _get_key_float(config, "feat_elevation", func_name) # Can be negative/zero

    # 3a. Feature Range must be within SS FOV [min, max]
    if not (ss_min_range <= feat_range <= ss_max_range):
        raise ValueError(f"{func_name}: 'feat_range' ({feat_range}) must be within SS range [{ss_min_range}, {ss_max_range}].")

    # 3b. Feature Azimuth must be within SS FOV aperture
    ss_az_bound = ss_azimuth_aperture / 2.0
    if not (-ss_az_bound <= feat_azimuth <= ss_az_bound):
        raise ValueError(f"{func_name}: 'feat_azimuth' ({feat_azimuth}) must be within SS azimuth FOV bounds [{-ss_az_bound}, {ss_az_bound}] degrees.")
        
    # 3c. Feature Elevation must be within SS FOV aperture
    ss_el_bound = ss_elevation_aperture / 2.0
    if not (-ss_el_bound <= feat_elevation <= ss_el_bound):
        raise ValueError(f"{func_name}: 'feat_elevation' ({feat_elevation}) must be within SS elevation FOV bounds [{-ss_el_bound}, {ss_el_bound}] degrees.")

    # --- 4. Relative Pose Configuration (rp_) Checks ---

    rp_range = _validate_range_value(config, "rp_range", func_name)
    rp_azimuth = _get_key_float(config, "rp_azimuth", func_name)
    rp_elevation = _get_key_float(config, "rp_elevation", func_name)
    
    # 4a. Relative Pose Range must be within FL FOV [min, max]
    if not (fl_min_range <= rp_range <= fl_max_range):
        raise ValueError(f"{func_name}: 'rp_range' ({rp_range}) must be within FL range [{fl_min_range}, {fl_max_range}].")

    # 4b. Relative Pose Azimuth must be within FL FOV aperture
    fl_az_bound = fl_azimuth_aperture / 2.0
    if not (-fl_az_bound <= rp_azimuth <= fl_az_bound):
        raise ValueError(f"{func_name}: 'rp_azimuth' ({rp_azimuth}) must be within FL azimuth FOV bounds [{-fl_az_bound}, {fl_az_bound}] degrees.")
        
    # 4c. Relative Pose Elevation must be within FL FOV aperture
    fl_el_bound = fl_elevation_aperture / 2.0
    if not (-fl_el_bound <= rp_elevation <= fl_el_bound):
        raise ValueError(f"{func_name}: 'rp_elevation' ({rp_elevation}) must be within FL elevation FOV bounds [{-fl_el_bound}, {fl_el_bound}] degrees.")
        
    # 4d. Check Roll, Pitch, Yaw (Must be numeric, no specific bounds required by prompt)
    rp_roll = _get_key_float(config, "rp_roll", func_name)
    rp_pitch = _get_key_float(config, "rp_pitch", func_name)
    rp_yaw = _get_key_float(config, "rp_yaw", func_name)

    # --- 5. Return Validated Config ---
    # Since we used _get_key_float and validation functions which cast to float, 
    # we can rebuild the dictionary using the validated values.
    
    validated_config = {
        # Sidescan sonar config
        "ss_min_range": ss_min_range,
        "ss_max_range": ss_max_range,
        "ss_azimuth_aperture": ss_azimuth_aperture,
        "ss_azimuth_resolution": ss_azimuth_resolution,
        "ss_elevation_aperture": ss_elevation_aperture,
        "ss_elevation_resolution": ss_elevation_resolution,
        # Forward-looking sonar config
        "fl_min_range": fl_min_range,
        "fl_max_range": fl_max_range,
        "fl_azimuth_aperture": fl_azimuth_aperture,
        "fl_elevation_aperture": fl_elevation_aperture,
        # Feature config
        "feat_range": feat_range,
        "feat_azimuth": feat_azimuth,
        "feat_elevation": feat_elevation,
        # Relative pose config
        "rp_range": rp_range,
        "rp_azimuth": rp_azimuth,
        "rp_elevation": rp_elevation,
        "rp_roll": rp_roll,
        "rp_pitch": rp_pitch,
        "rp_yaw": rp_yaw,
    }

    return validated_config

def example_fl2ss_config_checks(config: dict):
    # Define valid numeric types (standard Python int/float and any NumPy numeric type)
    VALID_NUMERIC_TYPES = (int, float, np.number)

    def _get_key_float(config: dict, key: str, func_name: str) -> float:
        '''
        Helper to check if a key exists in the config, is numeric, and returns its value as a float.
        Raises ValueError or TypeError if invalid.
        '''
        if key not in config:
            raise ValueError(f"{func_name}: Missing required configuration key '{key}'.")
        
        val = config[key]
        if not isinstance(val, VALID_NUMERIC_TYPES):
            raise TypeError(f"{func_name}: Value for key '{key}' is of invalid type. Expected numeric, found {type(val)}.")
            
        return float(val)

    def _validate_range_value(config: dict, key: str, func_name: str) -> float:
        '''
        Checks that a numeric value is non-negative (>= 0).
        '''
        val = _get_key_float(config, key, func_name)
        if val < 0:
            raise ValueError(f"{func_name}: Value for key '{key}' must be non-negative (>= 0). Found {val}.")
        return val

    def _validate_aperture_or_resolution(config: dict, key: str, func_name: str) -> float:
        '''
        Checks that a numeric value is strictly positive (> 0).
        '''
        val = _get_key_float(config, key, func_name)
        if val <= 0:
            raise ValueError(f"{func_name}: Value for key '{key}' must be strictly positive (> 0). Found {val}.")
        return val

    '''
    Validates all constraints within the sonar and pose configuration dictionary for forward-looking to sidescan sonar.

    The validated dictionary, with all values cast to float, is returned.

    :param config: The dictionary containing all sensor and pose parameters.
    :type config: dict
    :returns: The validated configuration dictionary with values cast to float.
    :rtype: dict
    :raises TypeError: If any value is not numeric.
    :raises ValueError: If any value violates constraints (e.g., negative range, max < min, FOV bounds).
    '''
    func_name = "config_checks_v2.check_sonar_config_v2"
    
    # --- 1. Forward-Looking Sonar (FL) Checks ---
    
    # 1a. Ranges (>= 0)
    fl_min_range = _validate_range_value(config, "fl_min_range", func_name)
    fl_max_range = _validate_range_value(config, "fl_max_range", func_name)
    
    # 1b. Range Constraint (min < max)
    if fl_min_range >= fl_max_range:
        raise ValueError(f"{func_name}: 'fl_min_range' ({fl_min_range}) must be strictly less than 'fl_max_range' ({fl_max_range}).")
        
    # 1c. Apertures and Resolutions (> 0)
    fl_azimuth_aperture = _validate_aperture_or_resolution(config, "fl_azimuth_aperture", func_name)
    fl_elevation_aperture = _validate_aperture_or_resolution(config, "fl_elevation_aperture", func_name)
    fl_elevation_resolution = _validate_aperture_or_resolution(config, "fl_elevation_resolution", func_name) # Note: Only elevation resolution is listed

    # --- 2. Sidescan Sonar (SS) Checks ---
    
    # 2a. Ranges (>= 0)
    ss_min_range = _validate_range_value(config, "ss_min_range", func_name)
    ss_max_range = _validate_range_value(config, "ss_max_range", func_name)

    # 2b. Range Constraint (min < max)
    if ss_min_range >= ss_max_range:
        raise ValueError(f"{func_name}: 'ss_min_range' ({ss_min_range}) must be strictly less than 'ss_max_range' ({ss_max_range}).")

    # 2c. Apertures (> 0)
    ss_azimuth_aperture = _validate_aperture_or_resolution(config, "ss_azimuth_aperture", func_name)
    ss_elevation_aperture = _validate_aperture_or_resolution(config, "ss_elevation_aperture", func_name)

    # --- 3. Feature Configuration (feat_) Checks (Bound to FL Sonar) ---
    
    feat_range = _validate_range_value(config, "feat_range", func_name)
    feat_azimuth = _get_key_float(config, "feat_azimuth", func_name) # Can be negative/zero
    feat_elevation = _get_key_float(config, "feat_elevation", func_name) # Can be negative/zero

    # 3a. Feature Range must be within FL FOV [min, max]
    if not (fl_min_range <= feat_range <= fl_max_range):
        raise ValueError(f"{func_name}: 'feat_range' ({feat_range}) must be within FL range [{fl_min_range}, {fl_max_range}].")

    # 3b. Feature Azimuth must be within FL FOV aperture
    fl_az_bound = fl_azimuth_aperture / 2.0
    if not (-fl_az_bound <= feat_azimuth <= fl_az_bound):
        raise ValueError(f"{func_name}: 'feat_azimuth' ({feat_azimuth}) must be within FL azimuth FOV bounds [{-fl_az_bound}, {fl_az_bound}] degrees.")
        
    # 3c. Feature Elevation must be within FL FOV aperture
    fl_el_bound = fl_elevation_aperture / 2.0
    if not (-fl_el_bound <= feat_elevation <= fl_el_bound):
        raise ValueError(f"{func_name}: 'feat_elevation' ({feat_elevation}) must be within FL elevation FOV bounds [{-fl_el_bound}, {fl_el_bound}] degrees.")

    # --- 4. Relative Pose Configuration (rp_) Checks (Bound to SS Sonar) ---

    rp_range = _validate_range_value(config, "rp_range", func_name)
    rp_azimuth = _get_key_float(config, "rp_azimuth", func_name)
    rp_elevation = _get_key_float(config, "rp_elevation", func_name)
    
    # 4a. Relative Pose Range must be within SS FOV [min, max]
    if not (ss_min_range <= rp_range <= ss_max_range):
        raise ValueError(f"{func_name}: 'rp_range' ({rp_range}) must be within SS range [{ss_min_range}, {ss_max_range}].")

    # 4b. Relative Pose Azimuth must be within SS FOV aperture
    ss_az_bound = ss_azimuth_aperture / 2.0
    if not (-ss_az_bound <= rp_azimuth <= ss_az_bound):
        raise ValueError(f"{func_name}: 'rp_azimuth' ({rp_azimuth}) must be within SS azimuth FOV bounds [{-ss_az_bound}, {ss_az_bound}] degrees.")
        
    # 4c. Relative Pose Elevation must be within SS FOV aperture
    ss_el_bound = ss_elevation_aperture / 2.0
    if not (-ss_el_bound <= rp_elevation <= ss_el_bound):
        raise ValueError(f"{func_name}: 'rp_elevation' ({rp_elevation}) must be within SS elevation FOV bounds [{-ss_el_bound}, {ss_el_bound}] degrees.")
        
    # 4d. Check Roll, Pitch, Yaw (Must be numeric)
    rp_roll = _get_key_float(config, "rp_roll", func_name)
    rp_pitch = _get_key_float(config, "rp_pitch", func_name)
    rp_yaw = _get_key_float(config, "rp_yaw", func_name)

    # --- 5. Return Validated Config ---
    # Rebuild the dictionary using the validated values (all cast to float)
    
    validated_config = {
        # Forward-looking sonar config
        "fl_min_range": fl_min_range,
        "fl_max_range": fl_max_range,
        "fl_azimuth_aperture": fl_azimuth_aperture,
        "fl_elevation_aperture": fl_elevation_aperture,
        "fl_elevation_resolution": fl_elevation_resolution,
        # Sidescan sonar config
        "ss_min_range": ss_min_range,
        "ss_max_range": ss_max_range,
        "ss_azimuth_aperture": ss_azimuth_aperture,
        "ss_elevation_aperture": ss_elevation_aperture,
        # Feature config
        "feat_range": feat_range,
        "feat_azimuth": feat_azimuth,
        "feat_elevation": feat_elevation,
        # Relative pose config
        "rp_range": rp_range,
        "rp_azimuth": rp_azimuth,
        "rp_elevation": rp_elevation,
        "rp_roll": rp_roll,
        "rp_pitch": rp_pitch,
        "rp_yaw": rp_yaw,
    }

    return validated_config