import numpy as np
import h5py
import os
import scienceplots
from matplotlib import pyplot as plt
from pathlib import Path
from pydata.video import video_fft
from scipy.signal import find_peaks, peak_widths
from tkinter import Tk
from tkinter.filedialog import askdirectory
from uncertainties import unumpy as unp
plt.style.use('science')
cm = 1/2.54

ffts  = []
freqs = []
colors = plt.cm.hsv(np.linspace(0,1,8))
plt.figure(figsize=(18*cm, 6*cm))

#%% OPEN FILE
Tk().withdraw() 
directory = Path(askdirectory(title="Select data directory"))
print(f"Analyzing: {directory}")
print(os.path.isfile(directory / 'height_maps.hdf5'))
f = h5py.File(directory / 'height_maps.hdf5', 'r')

#%% DATA
fps         = f['fps'][0] / f['step'][0]
cal         = f['calibration_factor'][0]
height_maps = f['height_maps']
shape = height_maps.shape
print(fps, shape)

#%% TIME
t = np.arange(0, len(height_maps))/fps
ti, tf = min(t), max(t)
idx_i, idx_f = np.array(*np.where(np.logical_and(ti<t, t<tf)))[np.array([1,-1])]

#%% NEAR FIELD
heights  = height_maps[:,280:610,280:610]
heights -= heights.mean()

#%% FREQUENCY ANALYSIS
freqs, fft = video_fft(heights[idx_i:idx_f], fps=fps)
peaks, _   = find_peaks(np.sum(np.abs(fft), axis=(1,2)), distance=1, prominence=0.5e-7, height=0.1*np.max(np.abs(fft)))


fig = plt.figure(figsize=(12*cm, 5*cm))
fig.tight_layout()

plt.plot(freqs, np.sum(np.abs(fft), axis=(1,2)), color='k', lw=1)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Suma de amplitudes")
plt.savefig(directory / "espectro.png", dpi=500)
plt.show()

im = plt.imshow(np.abs(fft[peaks[0]]), cmap='bone')
plt.colorbar(im, label="Amplitud")
plt.show()

im = plt.imshow(np.angle(fft[peaks[0]]), cmap='hsv')
plt.colorbar(im, label="Fase")
plt.show()

f.close()
