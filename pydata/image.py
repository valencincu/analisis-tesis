from tkinter import filedialog
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey
from skimage.feature import canny
from skimage.filters import butterworth, gaussian
from skimage.transform import rotate, warp_polar

from pydata.roi import ROI

    
class Image(np.ndarray):
    def __new__(cls, image=None, image_path=None):
        if image is None:
            image_path = image_path if image_path is not None else Path(filedialog.askopenfilename(title='Select image path'))
            image = _read_image(image_path)

        result = np.asarray(image).view(cls)
        result._path = image_path if image_path is not None else getattr(image, '_path', None)
        return result


    def __array_finalize__(self, source):
        if source is None:
            return
        self._path = getattr(source, '_path', None)


    @property
    def path(self):
        return self._path


    def windowed(self, x_factor=0, y_factor=False):
        windowed = self * _tukey_window(self, x_factor=x_factor, y_factor=y_factor)(self)
        return Image(windowed, image_path=self.path)


    def cropped(self, roi=None, return_roi=False):
        roi = ROI(self, select=True) if roi is None else roi
        x, y, w, h = roi
        x, y = max(0, x), max(0, y)
        cropped = self[y:y+h, x:x+w]
        if not return_roi:
            return Image(cropped, image_path=self.path)
        else:
            return Image(cropped, image_path=self.path), roi


    def masked(self, mask=None, background=None, invert=False, sigma=None, return_mask=False):
        background = background if background is not None else np.zeros_like(self)

        mask = mask if mask is not None else np.ones_like(self)
        mask = 1-mask if invert else mask

        mask = mask if sigma is None else gaussian(mask, sigma=sigma).astype(np.float32)

        masked = Image(self * mask + background * (1-mask), image_path=self.path)
        return (mask, masked) if return_mask else masked


    def masked_circle(self, radius=None, center=None, background=None, invert=False, sigma=None, return_mask=False): 
        radius = np.min(self.shape) if radius is None else radius
        center = np.array(self.shape[::-1])//2 if center is None else center
        mask = _circle_mask(self, center, radius)
        return self.masked(mask, background, invert, sigma, return_mask)


    def masked_ring(self, radii, center=None, background=None, invert=False, sigma=None, return_mask=False): 
            center = np.array(self.shape[::-1])//2 if center is None else center
            mask = np.logical_or(_circle_mask(self, center, radii[0]), 1-_circle_mask(self, center, radii[1]))
            return self.masked(mask, background, invert, sigma, return_mask)


    def rotated(self, angle=None, center=None, resize=False):
        center  = center if center is not None else (self.shape[1]//2, self.shape[0]//2)
        angle   = angle if angle is not None else 0
        rotated = rotate(self, angle, center=center, resize=resize)
        return Image(rotated, image_path=self.path)


    def warp_polar(self, center=None, radius=None):
        return Image(warp_polar(self, center=center, radius=radius), image_path=self.path)


    def low_pass(self, cutoff_ratio=0.005, order=2): 
        low_pass = butterworth(self, cutoff_frequency_ratio=cutoff_ratio, order=order, high_pass=False, squared_butterworth=False)
        return Image(low_pass, image_path=self.path)


    def high_pass(self, cutoff_ratio=0.005, order=2): 
        high_pass = butterworth(self, cutoff_frequency_ratio=cutoff_ratio, order=order, high_pass=True, squared_butterworth=False)
        return Image(high_pass, image_path=self.path)


    def blurred(self, sigma=5): 
        blur = gaussian(self, sigma=sigma).astype(np.float32)
        return Image(blur, image_path=self.path)


    def with_sp_noise(self):
        pass

    
    def with_gauss_noise(self):
        pass

    
    def edges(self, low=0.7, high=0.98, sigma=5, mask=None):
        edges = canny(self, sigma=sigma, low_threshold=low, high_threshold=high, use_quantiles=True, mask=mask)
        return Image(edges, image_path=self.path)


    def correlate(self, template, image=None):
        correlation_map = cv2.matchTemplate(self, template, method=cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(correlation_map)
        return max_loc


    def normalized(self, scale=1, offset=0, astype=np.float32):
        img = self.astype(astype)
        range = (img.max() - img.min())
        return offset*np.ones_like(img) if range==0 else scale * (img - img.min()) / range + offset


    def inverted(self):
        return (self.max() - self) + self.min()


    def histogram(self, show=True, color='k', **histogram_kwargs):
        hist, bin_edges = np.histogram(self, **histogram_kwargs)

        _, ax = plt.subplots()

        ax.stairs(hist, bin_edges, color=color)

        if show:
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency density")
            plt.show()
        
        else:
            return ax
        

    def show(self, axis=None, show_plot=True):
        if axis is None:
            _, axis = plt.subplots()
        axis.imshow(self)
        if show_plot: plt.show()


    def save(self, filename="image.png", **kwargs): 
        plt.imshow(self)
        plt.savefig(filename, **kwargs)


def _read_image(path):                
    if path is not None:
        if path.suffix == ".npy": 
            image = np.load(path)
        else: 
            flag = cv2.IMREAD_UNCHANGED
            image = cv2.imread(path, flag)
    else:
        raise ValueError('Image.__init___: Image not found.')
    return image


def _circle_mask(image, center, radius):
    y_mesh, x_mesh = np.ogrid[:image.shape[0], :image.shape[1]]
    dist_from_center = np.sqrt((x_mesh - center[0]) ** 2 + (y_mesh - center[1]) ** 2)
    return dist_from_center < radius


def _tukey_window(image, x_factor=0.2, y_factor=False): 
    y_factor = y_factor if y_factor else x_factor
    def window_func(image):
        window_1d_x = np.abs(tukey(image.shape[1], x_factor))
        window_1d_y = np.abs(tukey(image.shape[0], y_factor))
        window_2d = np.sqrt(np.outer(window_1d_y, window_1d_x))
        return window_2d
    return window_func


if __name__ == '__main__': 
    image = Image()
    image.show()
    image.histogram(density=True, bins=255)