from tkinter import filedialog
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
from scipy.signal.windows import tukey
from skimage.feature import canny
from skimage.filters import butterworth, gaussian
from skimage.transform import hough_circle, hough_circle_peaks, rotate, warp_polar

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
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation_map)
        return max_loc

    def normalized(self, scale=1, offset=0, astype=np.float64):
        img = self.astype(astype)
        return scale * (img - img.min()) / (img.max() - img.min()) + offset

    def inverted(self):
        return (self.max() - self) + self.min()

    def show(self, axis=None, show_plot=True):
        if axis is None:
            fig, axis = plt.subplots()
        axis.imshow(self)
        if show_plot: plt.show()

    def save(self, filename="image.png", **kwargs): 
        plt.imshow(self)
        plt.savefig(filename, **kwargs)

class Structure: 
    def __init__(self, template_path=None, template=None, select_roi=False, find_structure=True, recenter=False):
        template = Image(template, image_path=template_path)
        roi = ROI(template, select=select_roi).square()
        temp_roi = roi

        center, in_radius, out_radius = roi.center, None, None
        if find_structure: 
            center, in_radius, out_radius = _find_circle_sliders(template, init_center=None, title='find_structure config')
            temp_roi = temp_roi.reshape((out_radius, out_radius)).recenter(center)
            if recenter: 
                roi.recenter(center)
                center = roi.center

        self._path     = template.path
        self._template = template.cropped(roi=temp_roi)
        self._inner_radius = in_radius
        self._outer_radius = out_radius
        self._roi          = roi
        self._center       = roi.absolute_to_local(center)
        self._temp_center  = temp_roi.absolute_to_local(center)
        self._position     = np.array(temp_roi.corner) + np.array(self._temp_center)

    @property
    def path(self):
        return self._path

    @property
    def template(self):
        return self._template

    @property
    def position(self): 
        return self._position

    @property
    def center(self): 
        return self._center

    @property
    def inner_radius(self):
        return self._inner_radius
    
    @inner_radius.setter
    def inner_radius(self, new_radius): 
        self._inner_radius = new_radius

    @property
    def outer_radius(self):
        return self._outer_radius
    
    @outer_radius.setter
    def outer_radius(self, new_radius): 
        self._outer_radius = new_radius

    @property
    def roi(self):
        return self._roi

    def track(self, image, sigma=10):
        template = self.template.blurred(sigma=sigma)
        image    = image.blurred(sigma=sigma)
        top_left = image.correlate(template)
        new_position  = np.array(top_left) + np.array(self._temp_center)
        self._position = [int(i) for i in new_position]
        return self.position


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

def _get_circle(edges, radii=None):
        radii = radii if radii is not None else np.arange(100, 500, 10)
        hough_res = hough_circle(edges, radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, radii, total_num_peaks=1)
        center = (int(cx[0]), int(cy[0]))
        return center, radii[0]

def _find_circle_sliders(image, init_center=None, title='find_circle sliders'):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.25, bottom=0.25)

    init_edge = 0.89
    im = ax.imshow(image.edges(low=max(0,init_edge-0.1), high=min(1,init_edge+0.1)), cmap='bone')

    if init_center is None:
        fig.suptitle('select center')
        init_center = fig.ginput(n=-1)[0]

    fig.suptitle(title)
    init_radius = min(image.shape)//8

    inner_circle = plt.Circle(init_center, init_radius, color='r', fill=False)
    outer_circle = plt.Circle(init_center, init_radius*2, color='b', fill=False)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    ax_edge_slider  = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax_inner_slider = fig.add_axes([0.15, 0.25, 0.0225, 0.63])
    ax_outer_slider = fig.add_axes([0.1, 0.25, 0.0225, 0.63])

    edge_slider   = Slider(ax=ax_edge_slider, label='edge_config', valmin=0, valmax=1, \
                           valinit=init_edge, valstep=1e-2)
    inner_slider = Slider(ax=ax_inner_slider, label='inner_radius', valmin=0, valmax=min(image.shape)//3, \
                           valinit=init_radius, valstep=min(image.shape)//400, orientation='vertical', color='r')
    outer_slider = Slider(ax=ax_outer_slider, label='outer_radius', valmin=0, valmax=min(image.shape)//2, \
                           valinit=init_radius*2, valstep=min(image.shape)//400, orientation='vertical', color='b')
    def update(val):
        im.set_data(image.edges(low=max(0, edge_slider.val-0.1), high=min(1, edge_slider.val+0.1)))
        inner_circle.radius = inner_slider.val
        outer_circle.radius = outer_slider.val
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    
    edge_slider.on_changed(update)
    inner_slider.on_changed(update)
    outer_slider.on_changed(update)

    ax_center = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(ax_center, 'Center')

    def recenter(event):
        edges = image.edges(low=max(0, edge_slider.val-0.1), high=min(1, edge_slider.val+0.1))
        radii = np.arange(inner_slider.val*0.9, inner_slider.val*1.1, 10)
        center, inner_radius = _get_circle(edges, radii)
        inner_circle.center = center
        outer_circle.center = center
        inner_circle.radius = inner_radius
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    button.on_clicked(recenter)

    fig.ginput(n=-1, show_clicks=False, timeout=120)
    plt.close(fig)

    edges = image.edges(low=max(0, edge_slider.val-0.1), high=min(1, edge_slider.val+0.1))
    radii = np.arange(inner_slider.val*0.9, inner_slider.val*1.1, 10)
    center, inner_radius = _get_circle(edges, radii)

    return center, inner_radius, outer_slider.val

if __name__ == '__main__': 
    image = Image(image_path='/home/valencincu/Facultad/L6y7/trapped-modes-ltg/laboratorio/datos/29_10/impulso_1_chico_202410_1024/impulso_1_chico_202410_1024000006.bmp')
    image.show()
    image = image.masked().cropped((1, 1, 10, 10))
    image.show()