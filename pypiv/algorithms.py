import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates

from openpiv import filters, smoothn
from openpiv.pyprocess import (
    extended_search_area_piv,
    get_rect_coordinates,
)

from pypiv.settings import CommonPIVSettings, SimplePIVSettings, WindefPIVSettings

# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------

DEFAULT_FIELD_INTERPOLATION_ORDER = 3

# ------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# ------------------------------------------------------------------------------

def generate_coordinates(shape, search_area_size, overlap, calibration_factor=1):
    x, y = get_rect_coordinates(shape, search_area_size, overlap)
    return x * calibration_factor, y * calibration_factor


def generate_background(video, start=0, stop=None, step=1):
    stop = len(video) if stop is None else stop

    background = np.zeros(video[0].shape, dtype=np.float64)
    n_frames = 0

    for i in range(start, stop, step):
        background += video[i].astype(np.float64)
        n_frames += 1

    return (background / n_frames) if n_frames != 0 else background


# ------------------------------------------------------------------------------
# ALGORITHM INTERNALS
# ------------------------------------------------------------------------------

def _correlate(frame_a, frame_b, settings: CommonPIVSettings, window_size, overlap, search_area_size=None):
    u, v, s2n = extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        width=settings.sig2noise_width,
        subpixel_method=settings.subpixel_method,
        sig2noise_method=settings.sig2noise_method,
        correlation_method=settings.correlation_method,
        normalized_correlation=settings.normalized_correlation,
        use_vectorized=settings.use_vectorized,
        backend=getattr(settings, "backend", "auto"),
    )
    return u, v, s2n


def _validate(u, v, s2n, settings: CommonPIVSettings, replace_flagged=False):
        flags = settings.validation_function(u, v, s2n)

        if replace_flagged:
            u, v = filters.replace_outliers(
                u, v, flags, 
                method=settings.filter_method, 
                max_iter=settings.max_filter_iteration, 
                kernel_size=settings.filter_kernel_size
            )

        return u, v, flags


def _reshape(shape, *args):
    reshaped_args = []
    for arg in args:
        reshaped_arg = arg.reshape(shape)
        reshaped_args.append(reshaped_arg)
    return reshaped_args


def _interpolate_field(field, x_old, y_old, x_new, y_new,
                       interpolation_order=DEFAULT_FIELD_INTERPOLATION_ORDER):
    
    spline = RectBivariateSpline(y_old, x_old, field, 
                                 kx=interpolation_order, ky=interpolation_order)

    return spline(y_new, x_new)


def _create_deformation_field(frame, x, y, u, v, interpolation_order=DEFAULT_FIELD_INTERPOLATION_ORDER):
    y_axis = y[:, 0]  # extract first coloumn from meshgrid
    x_axis = x[0, :]  # extract first row from meshgrid
    
    new_x_axis = np.arange(frame.shape[1])  # extract the image grid
    new_y_axis = np.arange(frame.shape[0])

    u_dense = _interpolate_field(
        u, x_axis, y_axis, new_x_axis, new_y_axis, 
        interpolation_order=interpolation_order
    )

    v_dense = _interpolate_field(
        v, x_axis, y_axis, new_x_axis, new_y_axis, 
        interpolation_order=interpolation_order
    )

    new_x, new_y = np.meshgrid(new_x_axis, new_y_axis)

    return new_x, new_y, u_dense, v_dense


def _windef_iteration(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    window_size: int,
    overlap: int,
    settings: WindefPIVSettings,
    replace_flagged: bool = True
):
    
    # ------------------------------------------------------------------------------
    # GET PREVIOUS FIELD IN NEW COORDINATES
    # ------------------------------------------------------------------------------
    y_old_axis = y[:, 0]
    x_old_axis = x[0, :]

    x_new, y_new = generate_coordinates(frame_a.shape, window_size, overlap)

    y_new_axis = y_new[:, 0]
    x_new_axis = x_new[0, :]

    u_pre = _interpolate_field(np.ma.filled(u, 0.), x_old_axis, y_old_axis, x_new_axis, y_new_axis)
    v_pre = _interpolate_field(np.ma.filled(v, 0.), x_old_axis, y_old_axis, x_new_axis, y_new_axis)

    # ------------------------------------------------------------------------------
    # DEFORM FRAMES
    # ------------------------------------------------------------------------------
    x_dense, y_dense, u_dense, v_dense = _create_deformation_field(frame_a, x_new, y_new, u_pre, v_pre)

    frame_a_def = map_coordinates(frame_a, ((y_dense - v_dense / 2, x_dense - u_dense / 2)),
                              order=settings.interpolation_order, mode='nearest')
    
    frame_b_def = map_coordinates(frame_b, ((y_dense + v_dense / 2, x_dense + u_dense / 2)),
                              order=settings.interpolation_order, mode='nearest')

    # ------------------------------------------------------------------------------
    # GET NEW DISPLACEMENTS
    # ------------------------------------------------------------------------------
    u_new, v_new, s2n = _correlate(frame_a_def, frame_b_def, settings, window_size, overlap)

    # ------------------------------------------------------------------------------
    # ADD NEW DISPLACEMENTS TO VELOCITY FIELD
    # ------------------------------------------------------------------------------
    u_new += u_pre
    v_new += v_pre

    # ------------------------------------------------------------------------------
    # FILTERS
    # ------------------------------------------------------------------------------

    u_new, v_new, flags = _validate(u_new, v_new, s2n, settings, replace_flagged=replace_flagged)

    return x_new, y_new, u_new, v_new, flags


# ------------------------------------------------------------------------------
# PIV ALGORITHMS
# ------------------------------------------------------------------------------

def simple_piv(frame_a: np.ndarray, frame_b: np.ndarray, settings: SimplePIVSettings, replace_flagged: bool = True, calibration_factor=1, dt=1):

    x, y = generate_coordinates(frame_a.shape, settings.search_area_size, settings.overlap)

    u, v, s2n = _correlate(frame_a, frame_b, settings,settings.window_size, settings.overlap, settings.search_area_size)
    u, v, s2n = _reshape(x.shape, u, v, s2n)
    u, v, flags = _validate(u, v, s2n, settings, replace_flagged=replace_flagged)


    return x * calibration_factor, y * calibration_factor, u * calibration_factor / dt, v * calibration_factor / dt, flags


def windef_piv(frame_a: np.ndarray, frame_b: np.ndarray, settings: WindefPIVSettings, replace_flagged: bool = True, calibration_factor=1, dt=1):

    x, y = generate_coordinates(frame_a.shape, settings.window_sizes[0], settings.overlaps[0])
    u, v, s2n = _correlate(frame_a, frame_b, settings, settings.window_sizes[0], settings.overlaps[0])
    u, v, flags = _validate(u, v, s2n, settings, replace_flagged=replace_flagged)

    for i in range(1, settings.num_iterations):
        if settings.use_smoothn:
            u, *_ = smoothn.smoothn(u, s=settings.smoothn_s)
            v, *_ = smoothn.smoothn(v, s=settings.smoothn_s)
        
        x, y, u, v, flags = _windef_iteration(frame_a, frame_b, x, y, u, v, settings.window_sizes[i], settings.overlaps[i], settings, replace_flagged=replace_flagged)

    return x * calibration_factor, y * calibration_factor, u * calibration_factor / dt, v * calibration_factor / dt, flags