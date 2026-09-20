import numpy as np
from scipy.fft import fft2, fftshift, fftfreq, ifft2
from skimage.measure import regionprops, label


def wavenumber_axis(size, calibration_factor=1, shifted=True):
    frequencies = fftfreq(size, calibration_factor / (2.0 * np.pi))
    return fftshift(frequencies) if shifted else frequencies


def wavenumber_meshgrid(shape, calibration_factor=1, shifted=True):
    k_rows = wavenumber_axis(shape[0], calibration_factor, shifted)
    k_cols = wavenumber_axis(shape[1], calibration_factor, shifted)
    return np.meshgrid(k_rows, k_cols, indexing='ij')


def remove_degeneracy(kx, ky, shape):
    if shape[1] % 2 == 0:
        kx[:, shape[1]//2+1] = 0  # Remove degeneracy at kx=Nx/2 leading to imaginary part.

    if shape[0] % 2 == 0:
        ky[shape[0]//2+1, :] = 0  # Remove degeneracy at ky=Ny/2 leading to imaginary part.


def pixels_to_wavenumbers(image_shape, locations, calibration_factor=1):
    k_space_rows = wavenumber_axis(image_shape[0], calibration_factor, shifted=True)
    k_space_cols = wavenumber_axis(image_shape[1], calibration_factor, shifted=True)

    if isinstance(locations[0], np.ndarray):
        wavenumbers =  np.array([[k_space_rows[loc[0]], k_space_cols[loc[1]]] for loc in locations])
    else:
        wavenumbers = np.array([k_space_rows[locations[0]], k_space_cols[locations[1]]])

    return wavenumbers


def integrate_gradient(gradient_x, gradient_y, calibration_factor=1): 
    ky, kx = wavenumber_meshgrid(gradient_x.shape, calibration_factor,shifted=False)  
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1

    remove_degeneracy(kx, ky, gradient_x.shape)

    gradient_x_hat, gradient_y_hat = fft2(gradient_x), fft2(gradient_y)
    integrated_hat = (-1.0j * kx * gradient_x_hat + -1.0j * ky * gradient_y_hat) / k2

    return np.real(ifft2(integrated_hat))


def find_peak_locations(image, threshold, no_peaks):
    blob_image = np.array(image > threshold)

    # make the borders false
    blob_image[0] *= False
    blob_image[-1] *= False
    blob_image[..., 0] *= False
    blob_image[..., -1] *= False

    blob_data = regionprops(label(blob_image.astype(np.uint8)))

    def blob_max_pixel_intensity(blob):
        pixels_with_coords = [(image[tuple(c)], c) for c in blob.coords]
        return max(pixels_with_coords, key=lambda x: x[0])

    blobs_with_max_intensity_and_coord = [blob_max_pixel_intensity(blob) for blob in blob_data]
    sorted_blobs = sorted(blobs_with_max_intensity_and_coord, key=lambda x: x[0])
    return [peak[1] for peak in sorted_blobs[:no_peaks]] 


def find_carrier_peaks(image):
    image_fft = fftshift(np.abs(fft2(image - np.mean(image))))

    def highpass_mask():
        ks_mesh_x, ks_mesh_y = wavenumber_meshgrid(image_fft.shape, shifted=True)
        kmin = 4 * np.pi / min(image.shape)
        return (ks_mesh_x**2 + ks_mesh_y**2) > kmin**2

    def angles(testing_peak):
        testing_peak_frequency = pixels_to_wavenumbers(image_fft.shape, testing_peak)
        return abs(np.arctan2(*testing_peak_frequency))

    def alignment(testing_peak):
        first_peak_frequency   = pixels_to_wavenumbers(image_fft.shape, rightmost_peak)
        testing_peak_frequency = pixels_to_wavenumbers(image_fft.shape, testing_peak)
        return abs(np.dot(first_peak_frequency, testing_peak_frequency))

    image_fft *= highpass_mask()
    threshold = 0.5 * np.max(image_fft)

    peak_locations     = find_peak_locations(image_fft, threshold, 4)
    rightmost_peak     = min(peak_locations, key=angles)
    perpendicular_peak = min(peak_locations, key=alignment)

    return rightmost_peak, perpendicular_peak
