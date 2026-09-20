#%% SELECT DATA DIRECTORY
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
square_size = 2e-3 # m
alpha = 1 - refractive_indexes["Air"] / refractive_indexes["Water"]

layers = [
    Layer(2.2e-2, "Air"), 
    Layer(0.5e-2, "Glass"), 
    Layer(8.1e-2, "Water")
]

H  = 72e-2  # m (distance between camera and pattern)
hp =  np.sum([layer.effective_height(refractive_indexes["Water"]) for layer in layers])

effective_height = (1 / (alpha*hp) + 1 / H)**(-1)

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
plt.style.use("science")

fig, axs = plt.subplots(2, 1, figsize=(8.5*cm, 20*cm))
fig.tight_layout(pad=2)

axs[0].imshow(displaced_image.cropped(roi=roi), cmap="Greys_r")
axs[0].axis("off")

im = axs[1].imshow(height_field*1e3, cmap="RdBu")
axs[1].axis("off")
plt.colorbar(im, label="Altura [mm]", ax=axs[1], orientation='horizontal')

plt.savefig("fcd.pdf", dpi=200)
plt.show()


