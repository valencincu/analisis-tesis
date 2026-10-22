import matplotlib.pyplot as plt
import numpy as np
import h5py

from pypiv.algorithms import simple_piv, windef_piv, generate_coordinates
from pypiv.execution import animatable, saveable
from pypiv.settings import WindefPIVSettings, SimplePIVSettings
from pypiv.visualization import piv_plot, compare_frames

from matplotlib import use as matplotlib_backend
matplotlib_backend("TkAgg")

# --------------------------------------------------------------------------
# PIV RUNNER CLASS
# --------------------------------------------------------------------------

class PIVRunner:
    def __init__(self, video, settings, fps=1, interval=1, calibration_factor=1):
        self.video = video
        self.image_shape = self.video[0].shape
        self.settings = settings
        self.fps = fps
        self.interval = interval
        self.dt = self.interval / self.fps
        self.calibration_factor = calibration_factor

        if isinstance(settings, SimplePIVSettings):
            self.process = simple_piv

        elif isinstance(settings, WindefPIVSettings):
            self.process = windef_piv

        else:
            raise TypeError("Settings must be SimplePIVSettings or WindefPIVSettings")

        self.x, self.y, *_ = self.velocity_field(0)


    @saveable("_save")
    @animatable("_animate")
    def run(self, start=0, stop=None, step=1):
        stop = len(self.video) - self.interval if stop is None else stop
        for frame_index in range(start, stop, step):
            _, _, u, v, flags = self.velocity_field(frame_index)
            yield frame_index / self.fps, u, v, flags

    
    def velocity_field(self, frame_index):
        current_frame = self.video[frame_index] 
        next_frame = self.video[frame_index + self.interval]

        x, y, u, v, flags = self.process(current_frame, next_frame, self.settings, calibration_factor=self.calibration_factor, dt=self.dt)

        return x, y, u, v, flags
    

    def plot(self, u, v, **kwargs):
        return piv_plot(self.x, self.y, u, v, **kwargs)


    def compare_frames(self, frame_index, ax=None, show=True):
        first_frame  = self.video[frame_index] 
        second_frame = self.video[frame_index + self.interval] 

        ax = compare_frames(first_frame, second_frame, show=False)
        ax.set_title(f"Frame {frame_index} (red) and {frame_index + self.interval} (blue)")

        if show:
            plt.show()
        else:
            return ax
        

    def _save(self, gen, save_at):
        length = len(self.video)
        shape  = self.x.shape
        with h5py.File(str(save_at) + ".hdf5", "w") as f:
            f.create_dataset("calibration_factor [m]", data=self.calibration_factor, dtype=np.float64)
            f.create_dataset("x [m]", data=self.x, dtype=np.float64)
            f.create_dataset("y [m]", data=self.y, dtype=np.float64)
            us = f.create_dataset("u [m per s]", shape=(0, *shape), maxshape=(length, *shape), chunks=(1, *shape), dtype=np.float64)
            vs = f.create_dataset("v [m per s]", shape=(0, *shape), maxshape=(length, *shape), chunks=(1, *shape), dtype=np.float64)  
            fs = f.create_dataset("flags", shape=(0, *shape), maxshape=(length, *shape), chunks=(1, *shape), dtype=bool)  
            ts = f.create_dataset("t [s]", shape=(0,), maxshape=(length,), dtype=np.float64)

            i = 0
            for t, u, v, flags in gen:
                us.resize(i+1, axis=0)
                vs.resize(i+1, axis=0)
                fs.resize(i+1, axis=0)
                ts.resize(i+1, axis=0)

                us[i] = u
                vs[i] = v
                fs[i] = flags
                ts[i] = t

                i += 1

                yield t, u, v, flags


    def _animate(self, gen):
        plt.ion()
        fig, ax = plt.subplots()

        for t, u, v, flags in gen:
            ax.clear()

            ax.set_title(f"{t} s")
            self.plot(u, v, ax=ax)

            fig.canvas.draw()
            fig.canvas.flush_events()

            yield t, u, v, flags

        plt.ioff()


if __name__ == "__main__":
    from openpiv import tools, validation

    frame_a = tools.imread( 'pypiv/test_images/exp1_001_a.bmp' )
    frame_b = tools.imread( 'pypiv/test_images/exp1_001_b.bmp' )

    video = [frame_a, frame_b]

    def validation_func(u, v, s2n):

        flags = validation.sig2noise_val(s2n, threshold=1.0)

        flags |= validation.local_norm_median_val(u, v, 0.1, 2)

        return flags

    settings = WindefPIVSettings(
        sig2noise_method="peak2peak",
        validation_function=validation_func
    )

    piv = PIVRunner(video, settings)

    u, v, flags = piv.velocity_field(0)

    piv.plot(u, v, show=True, flags=flags)