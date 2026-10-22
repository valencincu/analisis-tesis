import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates, uniform_filter

from pypiv.visualization import compare_frames, piv_plot
from pypiv.settings import SimplePIVSettings, WindefPIVSettings

from openpiv import validation
from pypiv.algorithms import simple_piv, windef_piv

def normalized(image, scale=1, offset=0, astype=np.float32):
    img = image.astype(astype)
    range = (img.max() - img.min())
    return offset*np.ones_like(img) if range==0 else scale * (img - img.min()) / range + offset


def create_sythetic_image(size):
    rng = np.random.default_rng()
    image = rng.integers(low=0, high=32, endpoint=True, size=size)
    image = uniform_filter((image > 31) * 255.0, 3) + (image > 31) * 255.0
    image = image / (3 * 255)
    return image

synthetic_image = create_sythetic_image((512, 512))

x_axis = np.arange(synthetic_image.shape[1])
y_axis = np.arange(synthetic_image.shape[0])

x, y = np.meshgrid(x_axis, y_axis)

# Velocity field
u = 10 * y / 512
v = 10 * x / 512

test_image_2 = map_coordinates(synthetic_image, (y - v, x - u), order=3, mode='nearest')
test_image_1 = map_coordinates(test_image_2, (y + v, x + u), order=3, mode='nearest')

test_image_2 = normalized(test_image_2, scale=1).astype(np.float64)
test_image_1 = normalized(test_image_1, scale=1).astype(np.float64)

compare_frames(test_image_1, test_image_2, show=True)

piv_plot(x[::16,::16], y[::16,::16], u[::16,::16], v[::16,::16], show=True)

def validation_func(u, v, s2n):

    flags = validation.sig2noise_val(s2n, threshold=1.0)

    flags |= validation.local_norm_median_val(u, v, 0.1, 2)

    return flags

settings = WindefPIVSettings(
    window_sizes = (32, 16),
    overlaps = (16, 8),
    validation_function=validation_func
)

x, y, u_piv, v_piv, flags = windef_piv(test_image_1, test_image_2, settings, calibration_factor=0.5, dt=0.5)

piv_plot(x, y, u_piv, v_piv, show=True)


print(np.max(u), np.min(u))
print(np.max(u_piv), np.min(u_piv))

settings = SimplePIVSettings(
    window_size = 32,
    overlap = 16,
    validation_function=validation_func
)

x, y, u_piv, v_piv, flags = simple_piv(test_image_1, test_image_2, settings, calibration_factor=0.5, dt=0.5)

print(np.max(u), np.min(u))
print(np.max(u_piv), np.min(u_piv))

piv_plot(x, y, u_piv, v_piv, show=True)
