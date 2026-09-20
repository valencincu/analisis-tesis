#%% SELECT DATA DIRECTORY
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

Tk().withdraw() 
dirname = askdirectory(title="Select data directory") 
print(f"Analyzing: {dirname}")

#%% IMPORT LIBRARIES
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from glob import glob
from matplotlib import use as matplotlib_backend
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

from pyfcd.layer import Layer, refractive_indexes
from pyfcd.fcd import FCD

from pydata.video import Video
from pydata.roi import ROI
from pydata.image import *

matplotlib_backend("TkAgg")

# %% IMPORT PATH SETUP
import_dir    = Path(os.path.abspath(dirname))
job_id        = input("Enter job identifier: ", )

reference_image_path = next(
    import_dir.joinpath(filename)
    for filename in import_dir.glob("*.bmp")
    if "referencia" in str(filename).lower()
)

displaced_images_path = import_dir / "Video"

print("Reference image path:", reference_image_path)
print("Displaced images directory:", displaced_images_path)

# %% EXPORT PATH SETUP
current_dir  = Path(os.path.dirname(os.path.realpath(__file__)))
export_dir   = current_dir / "analysis"

save_dir = export_dir.joinpath(import_dir.name + " - " + job_id)
save_dir.mkdir(parents=True, exist_ok=False)
copyfile(next(displaced_images_path.glob("*.cih")), save_dir / "metadata.cih")

notes_path = save_dir / "notes.txt"

# %% VIDEO & REFERENCE SETUP, ROI SELECTION
video     = Video(directory_path=displaced_images_path)
reference = Image(image_path=reference_image_path)
roi       = ROI(video.current_frame, select=True).square()
step      = 4

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

effective_height = alpha*hp

# %% FCD SETUP
fcd = FCD(
    reference.cropped(roi=roi),
    square_size=square_size, 
    effective_height=effective_height
)

# %% FCD PROCESSING
def apply_fcd(frame):
    height_field = fcd.analyze(frame.cropped(roi=roi), full_output=False)
    return height_field

# %% SAVE CONFIG
path_string = f"""PATHS
displaced_images_path : {displaced_images_path}
reference_image_path  : {reference_image_path}
"""
user_input = input("NOTES:\n")
notes = f"""NOTES:\n{user_input}\nROI:{str(roi)}
"""

preprocess_string = open("fcd-analysis.py", "r").read()
data = ("\n---\n".join([notes, path_string, preprocess_string]));
notes_path.write_text(data)

# %% ANALYZE
f = h5py.File(save_dir / "height_maps.hdf5", "w")
height_maps = f.create_dataset("height_maps", (video.length // step, *roi.shape), dtype=np.float16)
f.create_dataset("fps", data=(video.fps,), dtype=np.float16)
f.create_dataset("calibration_factor", data=(fcd.calibration_factor,))
f.create_dataset("effective_height", data=(fcd.effective_height,))
f.create_dataset("step", data=(step,))

# %% PLOT
cm = 1/2.54
plt.style.use("science")
max_height = 4e-3

def panel_update(fig, axs, i_def, height_map, vmin=None, vmax=None):
        axs[0].imshow(i_def, cmap="Greys_r")
        axs[1].imshow(height_map, cmap="bone", vmin=vmin, vmax=vmax)
        fig.canvas.draw()
        fig.canvas.flush_events()

plt.ion()
fig, axs = plt.subplots(1, 2, figsize=(15*cm, 6*cm))
fig.tight_layout(pad=2)

for frame in tqdm(video.play(step=step)):
    i = video.current_frame_index // step

    height_map     = apply_fcd(frame)
    height_maps[i] = height_map.astype(np.float16)

    if (i % 10) == 0: 
        panel_update(fig, axs, frame, height_map, vmin=-max_height, vmax=max_height)
        if i / 10 == 1.0:
            plt.savefig(str(save_dir / "panel.png"), dpi=500)

plt.close()
plt.ioff()

f.close()