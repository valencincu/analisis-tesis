import matplotlib.pyplot as plt
import numpy as np
import h5py

from openpiv.filters import replace_outliers
from openpiv.pyprocess import extended_search_area_piv, get_coordinates
from openpiv import windef
from openpiv import preprocess
from openpiv import validation

from pypiv.decorators import animatable, saveable

from matplotlib import use as matplotlib_backend
matplotlib_backend("TkAgg")

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

DEFAULT_INTERVAL = 1
DEFAULT_WINDOW_SIZE = 16
DEFAULT_SEARCH_SIZE = 20
DEFAULT_OVERLAP = 8
DEFAULT_TO_DISTANCE = 1
DEFAULT_SIG2NOISE_THRESHOLD = 2
OUTLIER_REPLACEMENT_MAX_ITER = 10
OUTLIER_REPLACEMENT_KERNEL_SIZE = 2

# --------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# --------------------------------------------------------------------------

def generate_background(video, start=0, stop=None, step=1):
    stop = len(video) if stop is None else stop

    background = np.zeros(video[0].shape, dtype=np.float64)
    n_frames = 0

    for i in range(start, stop, step):
        background += video[i].astype(np.float64)
        n_frames += 1

    return (background / n_frames) if n_frames != 0 else background

# --------------------------------------------------------------------------
# PIV CLASS
# --------------------------------------------------------------------------

class PIV:
    def __init__(self, video, fps, interval=DEFAULT_INTERVAL, window_size=DEFAULT_WINDOW_SIZE, search_size=DEFAULT_SEARCH_SIZE, overlap=DEFAULT_OVERLAP, to_distance=DEFAULT_TO_DISTANCE, sig2noise_threshold=DEFAULT_SIG2NOISE_THRESHOLD, filter_func=None):
        self.video        = video
        self.fps          = fps
        self.interval     = interval
        self.image_size   = self.video[0].shape

        self.window_size  = window_size
        self.search_size  = search_size
        self.overlap      = overlap

        self._to_distance = to_distance
        self._to_time     = 1 / fps

        self.threshold    = sig2noise_threshold
        self.filter_func  = (
            filter_func if filter_func is not None 
            else lambda x, y: [np.zeros_like(x), np.zeros_like(y)]
        )
        self.x, self.y    = self._generate_coordinates()


    @saveable("_save")
    @animatable("_animate")
    def run(self, start=0, stop=None, step=1):
        stop = len(self.video) - self.interval if stop is None else stop
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

        flags_s2n = validation.sig2noise_val(sig2noise, threshold=self.threshold)

        u2, v2 = replace_outliers(
            u1, v1,
            flags_s2n,
            method='localmean',
            max_iter=OUTLIER_REPLACEMENT_MAX_ITER,
            kernel_size=OUTLIER_REPLACEMENT_KERNEL_SIZE,
        )

        flags_extra = self.filter_func(u2, v2)

        u3, v3 = replace_outliers(
            u2, v2,
            flags_extra,
            method='distance',
            max_iter=OUTLIER_REPLACEMENT_MAX_ITER,
            kernel_size=OUTLIER_REPLACEMENT_KERNEL_SIZE,
        )

        u4, v4 = u3 * self._to_distance , v3 * self._to_distance

        if show:
            self.plot(u4, v4)
            plt.show()

        return u4, v4


    def plot(self, u, v, background=None, ax=None, scale=None, show=False):
        if ax is None:
            _, ax = plt.subplots(dpi=100)
        if background is not None:
            ax.pcolormesh(background, cmap="gray", rasterized=True)
            
        ax.quiver(self.x, self.y, u, v, scale=scale)
        ax.set_aspect('equal')
        
        if show: plt.show()
        return ax
    

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

    
    def histogram(u, v, **histogram2d_kwargs):
        hist, x_edges, y_edges = np.histogram2d(u, v, **histogram2d_kwargs)
        return hist, x_edges, y_edges


    def _generate_coordinates(self):
        x, y = get_coordinates(image_size=self.image_size, 
                                         search_area_size=self.search_size, 
                                         overlap=self.overlap)
        return x * self._to_distance, y * self._to_distance


    def _save(self, gen, save_at):
        length = len(self.video)
        shape  = self.x.shape
        with h5py.File(str(save_at) + ".hdf5", "w") as f:
            f.create_dataset("calibration_factor [m]", data=self._to_distance, dtype=np.float64)
            f.create_dataset("x [m]", data=self.x)
            f.create_dataset("y [m]", data=self.y)
            us = f.create_dataset("u [m per s]", shape=(0, *shape), maxshape=(length, *shape), chunks=(1, *shape))
            vs = f.create_dataset("v [m per s]", shape=(0, *shape), maxshape=(length, *shape), chunks=(1, *shape))  
            ts = f.create_dataset("t [s]", shape=(0,), maxshape=(length,), dtype=np.float64)

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


    def _animate(self, gen):
        plt.ion()
        fig, ax = plt.subplots()

        for t, u, v in gen:
            ax.clear()

            ax.set_title(f"{t} s")
            self.plot(u, v, ax=ax)

            fig.canvas.draw()
            fig.canvas.flush_events()

            yield t, u, v

        plt.ioff()
    
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