import numpy as np
import matplotlib.pyplot as plt
import h5py

from matplotlib.colors import LogNorm
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.restoration import unwrap_phase

import pyfcd.fourier_space as fourier_space
from pyfcd.height_map import HeightMap
from pyfcd.carriers import Carrier

from pyaux.helpers import animatable, saveable

class FCD:
    def __init__(self, reference_image, *, effective_height=1, calibration_factor=None, square_size=None):
        self._reference_image   = reference_image.astype(np.float32)

        self._square_size       = square_size
        self._effective_height  = effective_height
        self.calibration_factor = calibration_factor

        self._peaks             = self._find_carrier_peaks()
        self.calibration_factor = self._determine_calibration_factor(self._peaks) if self.calibration_factor == None else self.calibration_factor
        self._carriers          = self._build_carriers()
        
        self.x, self.y   = self._generate_coordinates(self.calibration_factor)

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

        return HeightMap(self.x, self.y, height_map, phases, self.calibration_factor) if full_output else height_map

    @saveable("_save")
    @animatable("_animate")
    def analyze_series(self, displaced_series, fps=1, start=0, stop=None, step=1, unwrap=True, fixed_loc=None, fixed_val=0):
        stop = len(displaced_series) if stop == None else stop

        for i in range(start, stop, step):
            displaced_image = displaced_series[i]
            height_map = self.analyze(
                displaced_image, 
                unwrap=unwrap, 
                full_output=False, 
                fixed_loc=fixed_loc,
                fixed_val=fixed_val
            )
            yield i / fps, displaced_image, height_map

    def plot_carriers(self, show=True):
        fig, ax = plt.subplots()
        ax.imshow(self._reference_image, cmap="gray")

        for carrier in self._carriers:
            carrier.draw_vector(ax)

        ax.axis("off")
        if show: plt.show()
        return fig

    def plot_fft(self, displaced_image, show=True):
        fig, axs = plt.subplots(1, 2)

        x_ticks = np.linspace(0, self.reference_image.shape[1] - 1, 5, dtype=int)
        y_ticks = np.linspace(0, self.reference_image.shape[0] - 1, 5, dtype=int)
        kx = fourier_space.wavenumber_axis(self.reference_image.shape[1], self.calibration_factor)
        ky = fourier_space.wavenumber_axis(self.reference_image.shape[0], self.calibration_factor)

        ffts = {
            'Reference image': np.abs(self._fft_image(self._reference_image)), 
            'Deformed image' : np.abs(self._fft_image(displaced_image))
        }

        for i, (title, fft) in enumerate(ffts.items()):
            for carrier in self._carriers:
                carrier.draw_fft_marker(axs[i], color="r", fill=False)

            axs[i].imshow(fft, cmap='gray', norm=LogNorm(), rasterized=True, origin="lower")
            axs[i].set_title(title)
            axs[i].set_xticks(x_ticks, [f"{kx[tick] / 1e3:.2f}" for tick in x_ticks])
            axs[i].set_yticks(y_ticks, [f"{ky[tick] / 1e3:.2f}" for tick in y_ticks])

            axs[i].set_xlabel(r"$k_x$ [$\text{mm}^{-1}$]")
            axs[i].set_ylabel(r"$k_y$ [$\text{mm}^{-1}$]")

        if show: plt.show()
        return fig, axs

    def _generate_coordinates(self, calibration_factor):
        len_y, len_x = self.reference_image.shape
        x = np.arange(len_x) * calibration_factor
        y = np.arange(len_y) * calibration_factor
        return np.meshgrid(x, y)

    def _find_carrier_peaks(self):
        return fourier_space.find_carrier_peaks(self._reference_image)

    def _determine_calibration_factor(self, peaks):
        if self._square_size is None:
            return 1
        
        checkerboard_frequencies_px = fourier_space.pixels_to_wavenumbers(
            self._reference_image.shape,
            locations=peaks,
        )
        checkerboard_frequency_px  = np.mean(np.hypot(checkerboard_frequencies_px[:,0], checkerboard_frequencies_px[:,1])) / np.sqrt(2)
        checkerboard_wavelength_px = 2 * np.pi / checkerboard_frequency_px
        checkerboard_wavelength_m  = 2 * self._square_size

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


    def _save(self, gen, save_at):
        shape = self.reference_image.shape
        with h5py.File(str(save_at) + ".hdf5", "w") as f:
            height_maps = f.create_dataset(
                "height_maps [m]", (0, *shape), maxshape=(None, *shape), chunks=(1, *shape)
            )
            ts = f.create_dataset("t [s]", shape=(0,), maxshape=(None,), dtype=np.float64)
            f.create_dataset("x [m]", data=self.x)
            f.create_dataset("y [m]", data=self.y)
            f.create_dataset("calibration_factor [m]", data=self.calibration_factor, dtype=np.float64)

            i = 0
            for t, i_def, height_map in gen:
                height_maps.resize(i+1, axis=0)
                ts.resize(i+1, axis=0)

                height_maps[i] = height_map
                ts[i] = t

                i += 1

                yield t, i_def, height_map

    def _animate(self, gen, vmin=-5e-3, vmax=5e-3):
        plt.ion()
        fig, axs = plt.subplots(1, 2, figsize=(7.5, 3), dpi=150)
        fig.tight_layout(pad=2)

        title = plt.suptitle('')

        for t, i_def, height_map in gen:
            axs[0].clear()
            axs[1].clear()

            title.set_text(f"{t} s")
            axs[0].pcolormesh(self.x, self.y, i_def, cmap="Greys_r", rasterized=True)
            axs[1].pcolormesh(self.x, self.y, height_map, cmap="bone", vmin=vmin, vmax=vmax)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            yield t, i_def, height_map

        plt.ioff()

    
    




