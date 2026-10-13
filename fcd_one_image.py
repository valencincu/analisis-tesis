import os
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib import use as matplotlib_backend

from pydata.image import Image
from pydata.roi import ROI
from pyfcd.fcd import FCD
from pyfcd.layer import Layer, refractive_indexes

matplotlib_backend("TkAgg")
plt.style.use("science")

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

SQUARE_SIZE = 2e-3
PATTERN_CAMERA_DISTANCE = 80e-2
AIR_LAYER_THICKNESS = 2.2e-2
GLASS_LAYER_THICKNESS = 0.5e-2
WATER_LAYER_THICKNESS = 8.1e-2
ANALYSIS_STEP = 4

# --------------------------------------------------------------------------
# SELECT DISPLACED AND REFERENCE IMAGES
# --------------------------------------------------------------------------

Tk().withdraw() 
displaced_image_path = Path(askopenfilename(title="Select displaced image"))
reference_image_path = Path(askopenfilename(title="Select reference image"))
displaced_image = Image(image_path=displaced_image_path)
reference_image = Image(image_path=reference_image_path)
print("Displaced image path:", displaced_image_path)
print("Reference image path:", reference_image_path)

# select ROI
roi = ROI(displaced_image, select=True).square()

# --------------------------------------------------------------------------
# PHYSICAL CONFIG
# --------------------------------------------------------------------------

H  = PATTERN_CAMERA_DISTANCE

layers = [
    Layer(AIR_LAYER_THICKNESS, "Air"),
    Layer(GLASS_LAYER_THICKNESS, "Glass"),
    Layer(WATER_LAYER_THICKNESS, "Water"),
]

hp =  np.sum([
    layer.effective_height(refractive_indexes["Water"]) 
    for layer in layers
])

parallax_correction = (1 - hp / H) 
effective_square_size = SQUARE_SIZE * parallax_correction

alpha = 1 - refractive_indexes["Air"] / refractive_indexes["Water"]
effective_height =  (1 / (alpha*hp) - 1 / H)**(-1)

# --------------------------------------------------------------------------
# FCD PROCESSING
# --------------------------------------------------------------------------

fcd = FCD(
    reference_image.cropped(roi=roi),
    square_size=effective_square_size, 
    effective_height=effective_height
)

height_field = fcd.analyze(displaced_image.cropped(roi=roi), full_output=False)

# --------------------------------------------------------------------------
# PLOT
# --------------------------------------------------------------------------

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
plt.colorbar(im, label="Altura [mm]", ax=axs[1], orientation="horizontal")

plt.savefig("analyzed" + os.sep + "fcd.pdf", dpi=200)
plt.show()


