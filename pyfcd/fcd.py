import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.restoration import unwrap_phase

from pyfcd.height_map import HeightMap
from pyfcd.carriers import Carrier
import pyfcd.fourier_space as fourier_space

class FCD:
    def __init__(self, reference_image, *, effective_height=1, calibration_factor=None, square_size=None):
        self._reference_image   = reference_image.astype(np.float32)

        self._square_size       = square_size
        self._peaks             = self._find_carrier_peaks()
        self.calibration_factor = calibration_factor
        self.calibration_factor = self._determine_calibration_factor()
        self._carriers          = self._build_carriers()
        self._effective_height  = effective_height

    @property
    def reference_image(self):
        return self._reference_image

    @property
    def carriers(self):
        return self._carriers

    @property
    def effective_height(self):
        return self._effective_height

    def analyze(self, displaced_image, unwrap=True, full_output=True, fixed_loc=None, fixed_val=0):
        displaced_image_fft = self._fft_image(displaced_image)
        phases              = self._find_phases(displaced_image_fft, unwrap)
        displacement_field  = self._find_displacement_field(phases)
        height_gradient     = -displacement_field / (self.effective_height)

        height_map = fourier_space.integrate_gradient(*height_gradient, self.calibration_factor)
        if fixed_loc is not None:
            height_map = height_map - height_map[fixed_loc[0], fixed_loc[1]] + fixed_val

        return HeightMap(height_map, phases, self.calibration_factor) if full_output else height_map

    def plot_carriers(self, show=True):
        fig, ax = plt.subplot()
        ax.imshow(self._reference_image, cmap="gray")

        for i, carrier in enumerate(self._carriers):
            carrier.draw_vector(ax)

        ax.axis("off")
        if show: plt.show()
        return fig

    def plot_fft(self, displaced_image, show=True):
        fig, axs = plt.subplots(1, 2)

        fft_reference_image = self._fft_image(self._reference_image)
        fft_displaced_image = self._fft_image(displaced_image)

        axs[0].imshow(np.abs(fft_reference_image), cmap='gray', norm=LogNorm())
        axs[0].set_title('Reference image')

        axs[1].imshow(np.abs(fft_displaced_image), cmap='gray', norm=LogNorm())
        axs[1].set_title('Deformed image')

        x_ticks, y_ticks, ticks = self._make_wavenumber_ticks()

        for ax in axs:
            for carrier in self._carriers:
                carrier.draw_fft_marker(ax, color="r", fill=False)

            ax.set_xticks(x_ticks, [f"{tick[0] / 1e3:.2f}" for tick in ticks])
            ax.set_yticks(y_ticks, [f"{tick[1] / 1e3:.2f}" for tick in ticks])
            ax.set_xlabel(r"$k_x$ [$\text{mm}^{-1}$]")
            ax.set_ylabel(r"$k_y$ [$\text{mm}^{-1}$]")

        if show: plt.show()
        return fig

    def _find_carrier_peaks(self):
        return fourier_space.find_carrier_peaks(self._reference_image)

    def _determine_calibration_factor(self):
        if self.calibration_factor is not None:
            return self.calibration_factor

        if self._square_size is None:
            return 1
        
        checkerboard_frequencies_px = fourier_space.pixels_to_wavenumbers(
            self._reference_image.shape,
            locations=self._peaks,
        )
                                
        checkerboard_wavelength_px = 2 * np.pi / np.mean(np.abs(checkerboard_frequencies_px))
        checkerboard_wavelength_m = 2 * self._square_size

        return checkerboard_wavelength_m / checkerboard_wavelength_px

    def _build_carriers(self):
        peaks = self._peaks
        peak_radius = np.linalg.norm(peaks[0] - peaks[1]) / 2

        return [Carrier(self._reference_image, self.calibration_factor, peak, peak_radius) for peak in peaks]

    def _find_phases(self, displaced_image_fft, unwrap): 
        phases = np.zeros((2, *displaced_image_fft.shape))
        for i, carrier in enumerate(self._carriers):
            angles    = -np.angle(ifft2(ifftshift(displaced_image_fft * carrier.mask)) * carrier.ccsgn)
            phases[i] = unwrap_phase(angles) if unwrap else angles
        return phases

    def _find_displacement_field(self, phases):
        det_a = self._carriers[0].frequencies[1] * self._carriers[1].frequencies[0] - \
                self._carriers[0].frequencies[0] * self._carriers[1].frequencies[1]
        
        u = (self._carriers[1].frequencies[0] * phases[0] - self._carriers[0].frequencies[0] * phases[1]) / det_a
        v = (self._carriers[0].frequencies[1] * phases[1] - self._carriers[1].frequencies[1] * phases[0]) / det_a

        return np.array([u, v])

    def _fft_image(self, image):
        return fftshift(fft2(image - np.mean(image)))

    def _make_wavenumber_ticks(self):
        x_ticks = np.linspace(0, self._reference_image.shape[1] - 1, 5, dtype=int)
        y_ticks = np.linspace(0, self._reference_image.shape[0] - 1, 5, dtype=int)
        ticks = fourier_space.pixels_to_wavenumbers(
            self._reference_image.shape,
            np.array([y_ticks, x_ticks]).T,
            self.calibration_factor,
        )
        return x_ticks, y_ticks, ticks


