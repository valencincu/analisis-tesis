import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.fft import fft2, ifft2, ifftshift, fftshift
from skimage.draw import disk

import pyfcd.fourier_space as fourier_space


class Carrier:
    def __init__(self, reference_image, calibration_factor, peak, peak_radius):
        shape = reference_image.shape
        self.pixels = peak
        self.radius = peak_radius
        self.mask   = self.peak_mask(shape, peak, peak_radius)
        self.ccsgn  = self._ccsgn(reference_image)
        self.frequencies = fourier_space.pixels_to_wavenumbers(shape, peak, calibration_factor)

    def peak_mask(self, shape, pos, r):
        result = np.zeros(shape, dtype=bool)
        result[disk(pos, r, shape=shape)] = True
        return result

    def _ccsgn(self, reference_image):
        reference_image_fft = fftshift(fft2(reference_image))
        return np.conj(ifft2(ifftshift(reference_image_fft * self.mask)))

    def draw_vector(self, ax, *, color="red", scale=10.0):
        k = self.frequencies.copy()
        norm = np.linalg.norm(k)
        if norm == 0:
            warnings.warn("Carrier of length 0 not plotted.")
            return

        k /= norm

        start_point = self.pixels
        end_point = self.pixels + scale * k

        ax.arrow(
            start_point[1],
            start_point[0],
            end_point[1] - start_point[1],
            end_point[0] - start_point[0],
            head_width=10,
            head_length=10,
            fc=color,
            ec=color,
        )


    def draw_fft_marker(self, ax, *, color="red", fill=False):
        circle = plt.Circle(
            self.pixels[::-1],
            self.radius,
            color=color,
            fill=fill,
        )
        ax.add_patch(circle)

