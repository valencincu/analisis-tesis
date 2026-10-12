import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from pydata.image import Image
from pydata.roi import ROI
from skimage.transform import hough_circle, hough_circle_peaks

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
                roi = roi.recenter(center)
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