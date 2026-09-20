import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askdirectory
from matplotlib import use as matplotlib_backend
from pathlib import Path
matplotlib_backend("TkAgg")

# OPEN FILE
Tk().withdraw() 
directory = Path(askdirectory(title="Select data directory", initialdir="analysis"))
print(f"Analyzing: {directory}")
f = h5py.File(directory / "height_maps.hdf5", "r")

# AUXILIARY FUNCTION
def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min())

# DATA
fps         = int(f["fps"][0])
height_maps = f["height_maps"]
cal         = f["calibration_factor"][0]

shape = height_maps[0].shape

# PLAY VIDEO
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(directory / "output.avi", fourcc, 30.0, (shape[1], shape[0]), isColor=False)

plt.ion()

fig, ax = plt.subplots()
for i, frame in enumerate(height_maps):
    ax.clear()
    ax.imshow(frame, cmap="RdBu", vmin=-5e-3, vmax=5e-3)
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[0])
    plt.savefig(f"gif/{i:04}.png",dpi=150)
    fig.canvas.draw()
    fig.canvas.flush_events()  

    out.write((255*normalize_image(frame)).astype(np.uint8))

plt.ioff()

out.release()
f.close()
 

 
