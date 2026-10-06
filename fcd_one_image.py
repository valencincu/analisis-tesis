#%% SELECT DATA DIRECTORY
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from matplotlib import use as matplotlib_backend

from pyfcd.layer import Layer, refractive_indexes
from pyfcd.fcd import FCD

from pydata.video import Video
from pydata.roi import ROI
from pydata.image import *
matplotlib_backend("TkAgg")
plt.style.use("science")

# %% SELECT FILES
Tk().withdraw() 
displaced_image_path = Path(askopenfilename(title="Select deformed image"))
reference_image_path = Path(askopenfilename(title="Select reference image"))
print("Displaced image path:", displaced_image_path)
print("Reference image path:", reference_image_path)

# %% VIDEO & REFERENCE SETUP, ROI SELECTION
displaced_image = Image(image_path=displaced_image_path)
reference_image = Image(image_path=reference_image_path)
roi             = ROI(displaced_image, select=True).square()

# %% PHYSICAL PARAMETERS
alpha = 1 - refractive_indexes["Air"] / refractive_indexes["Water"]

H  = 80e-2  # m (distance between camera and pattern)

layers = [
    Layer(2.2e-2, "Air"), 
    Layer(0.5e-2, "Glass"), 
    Layer(8.1e-2, "Water")
]

hp =  np.sum([layer.effective_height(refractive_indexes["Water"]) for layer in layers])

parallax_correction = (1 - hp / H)
square_size = 2e-3 * parallax_correction # m
effective_height =  1 / (1 / (alpha*hp) - 1 / H)

# %% FCD SETUP
fcd = FCD(
    reference_image.cropped(roi=roi),
    square_size=square_size, 
    effective_height=effective_height
)

# %% FCD PROCESSING
height_field = fcd.analyze(displaced_image.cropped(roi=roi), full_output=False)

# %% PLOT
cm = 1/2.54

x = np.arange(0, height_field.shape[1]) * fcd.calibration_factor
y = np.arange(0, height_field.shape[1]) * fcd.calibration_factor
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(2, 1, figsize=(8.5*cm, 20*cm), dpi=200)
fig.tight_layout(pad=1)

axs[0].pcolormesh(X, Y, displaced_image.cropped(roi=roi), cmap="Greys_r", rasterized=True)
axs[0].set_aspect("equal")

im = axs[1].pcolormesh(X, Y, height_field*1e3, cmap="RdBu", rasterized=True)
axs[1].set_aspect("equal")
plt.colorbar(im, label="Altura [mm]", ax=axs[1], orientation='horizontal')

plt.savefig("analyzed" + os.sep + "fcd.pdf", dpi=200)
plt.show()


