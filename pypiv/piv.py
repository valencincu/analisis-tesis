import matplotlib.pyplot as plt
import numpy as np
import functools
import h5py

from openpiv.filters import replace_outliers
from openpiv.pyprocess import extended_search_area_piv, get_coordinates
from openpiv.preprocess import contrast_stretch, high_pass, intensity_clip
from openpiv.validation import sig2noise_val, local_norm_median_val, local_median_val, global_std

from matplotlib import use as matplotlib_backend
matplotlib_backend("TkAgg")

def _animatable(animate_func):
    """animate_func: name of a method on the class that draws the results."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, *args, animate=False, **kwargs):
            gen = f(self, *args, **kwargs)
            if not animate:
                return gen
            return getattr(self, animate_func)(gen)
        return wrapper
    return decorator

def _saveable(save_func):
    """save_func: name of a method on the class that saves the results."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, *args, save_at=None, **kwargs):
            gen = f(self, *args, **kwargs)
            if save_at == None:
                return gen
            return getattr(self, save_func)(gen, save_at)
        return wrapper
    return decorator

class PIV:
    def __init__(self, video, fps, interval=1, window_size=16, search_size=20, overlap=8, to_distance=1, sig2noise_threshold=2, filter_func=None):
        self.video      = video
        self.fps        = fps
        self.interval   = interval
        self.image_size = self.video[0].shape

        self.window_size = window_size
        self.search_size = search_size
        self.overlap     = overlap

        self._to_distance = to_distance
        self._to_time     = 1 / fps * 1000

        self.threshold    = sig2noise_threshold
        self.filter_func  = filter_func if filter_func != None else lambda x: np.zeros_like(x)

        self.x, self.y   = self._generate_coordinates()

    @_saveable("_save")
    @_animatable("_animate")
    def run(self, start=0, stop=None, step=1):
        stop = len(self.video) - self.interval if stop == None else stop
        for frame_index in range(start, stop, step):
            u, v = self.velocity_field(frame_index, show=False)
            yield frame_index * self._to_time, u, v

    def velocity_field(self, frame_index, show=True):
        current_frame = self.video[frame_index] 
        next_frame    = self.video[frame_index + self.interval]

        u1, v1, sig2noise = extended_search_area_piv(
            current_frame, 
            next_frame, 
            window_size=self.window_size,
            overlap=self.overlap, 
            dt=self.interval / self.fps, 
            search_area_size=self.search_size, 
            sig2noise_method="peak2peak"
        )

        flags_s2n = sig2noise_val(sig2noise, threshold=self.threshold)

        u2, v2 = replace_outliers(
            u1, v1,
            flags_s2n,
            method='localmean',
            max_iter=10,
            kernel_size=2,
        )

        flags_extra = self.filter_func(u2, v2)

        u3, v3 = replace_outliers(
            u2, v2,
            flags_extra,
            method='distance',
            max_iter=10,
            kernel_size=2,
        )

        u4, v4 = u3 * self._to_distance , v3 * self._to_distance

        if show:
            self.plot(u4, v4)
            plt.show()

        return u4, v4

    def compare_frames(self, frame_index, ax=None, show=True):
        first_frame  = self.video[frame_index] 
        second_frame = self.video[frame_index + self.interval] 

        overlay = np.zeros((*first_frame.shape, 3), dtype=float)
        overlay[..., 0] = first_frame
        overlay[..., 2] = second_frame

        if ax is None:
            _, ax = plt.subplots(dpi=100)

        ax.imshow(overlay)
        ax.set_title(f"Frame {frame_index} (red) and {frame_index + self.interval} (blue)")
        ax.set_axis_off()
        if show:
            plt.show()
        else:
            return ax

    def plot(self, u, v, background=None, ax=None, scale=None, show=False):
        if ax == None:
            _, ax = plt.subplots(dpi=100)
        if background is not None:
            ax.pcolormesh(background, cmap="gray", rasterized=True)
            
        ax.quiver(self.x, self.y, u, v, scale=scale)
        ax.set_aspect('equal')
        
        if show: plt.show()
        return ax

    def _generate_coordinates(self):
        x, y = get_coordinates(image_size=self.image_size, 
                                         search_area_size=self.search_size, 
                                         overlap=self.overlap)
        return x * self._to_distance, y * self._to_distance

    def _save(self, gen, save_at):
        f    = h5py.File(str(save_at) + ".hdf5", "w")
        f.create_dataset("x", data=self.x)
        f.create_dataset("y", data=self.y)
        us = f.create_dataset("u [m/s]", shape=(0, *self.x.shape), maxshape=(len(self.video), *self.x.shape))
        vs = f.create_dataset("v [m/s]", shape=(0, *self.x.shape), maxshape=(len(self.video), *self.x.shape))  
        ts = f.create_dataset(  "t [ms]", shape=(0, 1), maxshape=(len(self.video), 1))
    
        i = 0
        for t, u, v in gen:
            us.resize(i+1, axis=0)
            vs.resize(i+1, axis=0)
            ts.resize(i+1, axis=0)

            us[i] = u
            vs[i] = v
            ts[i] = t

            i += 1

            yield t, u, v

        return None

    def _animate(self, gen):
        plt.ion()
        fig, ax = plt.subplots()

        for t, u, v in gen:
            ax.clear()

            ax.set_title(f"{t} ms")
            self.plot(u, v, ax=ax)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            yield t, u, v

        plt.ioff()

def generate_background(video, start=0, stop=None, step=1):
    stop = len(video) if stop == None else stop

    background = np.zeros(video[0].shape, dtype=np.float64)
    N = 0

    for i in range(start, stop, step):
        background += video[i].astype(np.float64)
        N += 1

    return (background / N) if N != 0 else background
    
if __name__ == "__main__":
    from pydata.video import Video
    from pydata.roi import ROI
    
    video = Video()
    roi = ROI(video[0], select=True).square()

    piv = PIV(video, fps=video.fps, interval=10, to_distance=1/3500)
    piv.compare_frames(15)
    plt.show()
    piv.velocity_field(15, show=True)
    plt.show()