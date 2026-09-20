import glob
import os
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft

from pydata.image import Image

def read_metadata(path):
    metadata_file = glob.glob(os.path.join(path, "*.cih"))[0]
    metadata = {}
    with open(metadata_file, 'r') as file:
        for line in file:
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
    return metadata


allowed_formats = (".jpg", ".png", ".tiff", ".bmp", ".npy")


class Video:
    def __init__(self, directory_path=None):
        self.directory_path = directory_path if directory_path else filedialog.askdirectory(title="Selecciona una carpeta")
        self.frames = sorted(self.__get_frames())
        self._current_frame_index = 0
        self.current_frame = self.read_frame(frame_index=0)

        self.metadata = self.__read_metadata()
        self.fps = float(self.metadata["Record Rate(fps)"])
        self.length = len(self.frames)
        self.shape = self.current_frame.shape

    @property
    def current_frame_index(self):
        return self._current_frame_index

    @current_frame_index.setter
    def current_frame_index(self, new_index):
        self.read_frame(frame_index=new_index)

    def read_frame(self, frame_index=None):
        frame_index = (
            self.current_frame_index
            if frame_index is None
            else frame_index
        )

        if not 0 <= frame_index < len(self.frames):
            raise IndexError(f"Frame index out of range: {frame_index}")

        self._current_frame_index = frame_index
        self.current_frame = Image(image_path=self._frame_path(frame_index))
        return self.current_frame

    def play(self, start_frame=None, end_frame=None, step=1):
        start_frame = self.current_frame_index if start_frame is None else start_frame
        end_frame = self.length if end_frame is None else min(end_frame, self.length)

        for frame_index in range(start_frame, end_frame, step):
            self.read_frame(frame_index)
            yield self.current_frame
        
    def show_movie(self, *, start_frame=None, end_frame=None, step=1, pre_process=None, vmin=-1e-3, vmax=1e-3,):
        plt.ion()
        fig, ax = plt.subplots(1, 1)

        fig.canvas.draw()
        for frame in self.play(start_frame, end_frame, step):
            ax.clear()
            if pre_process is not None:
                to_show = pre_process(frame)
                ax.imshow(to_show, cmap='RdBu', vmin=vmin, vmax=vmax)
            else:
                frame.show(axis=ax, show_plot=False)

            fig.canvas.flush_events()

        plt.ioff()
        plt.show()

    def __read_metadata(self):
        metadata = read_metadata(self.directory_path)
        return metadata

    def __get_frames(self):
        return [file for file in os.listdir(self.directory_path) if file.endswith(allowed_formats)]

    def _frame_path(self, frame_index=None):
        frame_index = frame_index if frame_index is not None else self._current_frame_index
        return self.directory_path.joinpath(self.frames[frame_index])

def video_fft(frames, fps=1):
    frames_fft = fft.rfft(frames, axis=0, norm='forward')
    freqs = fft.rfftfreq(len(frames), 1/fps)
    return freqs, frames_fft

def video_ifft(freqs, frames_fft, freq_loc=None, sigmas=1):
    if freq_loc is not None:
        delta = (freqs[1] - freqs[0])*sigmas
        frames_fft = frames_fft.copy()
        frames_fft[np.abs(freqs - freq_loc) > delta,...] = 0

    video_ifft = fft.irfft(frames_fft, axis=0)
    return video_ifft