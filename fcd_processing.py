from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from pathlib import Path
from matplotlib import use as matplotlib_backend
from tqdm import tqdm

from pyfcd.layer import Layer, refractive_indexes
from pyfcd.fcd import FCD, find_reference_image

from pydata.video import Video
from pydata.roi import ROI
from pydata.image import *

from pyaux.output_config import create_output_directory, write_analysis_notes

matplotlib_backend("TkAgg")
plt.style.use('science')

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

SQUARE_SIZE = 2e-3
PATTERN_CAMERA_DISTANCE = 80e-2
AIR_LAYER_THICKNESS = 2.2e-2
GLASS_LAYER_THICKNESS = 0.5e-2
WATER_LAYER_THICKNESS = 8.1e-2
ANALYSIS_STEP = 4
FFT_SAVE_DPI = 200

# --------------------------------------------------------------------------
# PHYSICAL CONFIG
# --------------------------------------------------------------------------

layers = [
    Layer(AIR_LAYER_THICKNESS, "Air"),
    Layer(GLASS_LAYER_THICKNESS, "Glass"),
    Layer(WATER_LAYER_THICKNESS, "Water"),
]

H  = PATTERN_CAMERA_DISTANCE
hp =  np.sum([
    layer.effective_height(refractive_indexes["Water"]) 
    for layer in layers
])

parallax_correction = (1 - hp / H) 
effective_square_size = SQUARE_SIZE * parallax_correction

alpha = 1 - refractive_indexes["Air"] / refractive_indexes["Water"]
effective_height =  (1 / (alpha*hp) - 1 / H)**(-1)

# --------------------------------------------------------------------------
# VIDEO AND REFERENCE SETUP
# --------------------------------------------------------------------------

video = Video()
reference = Image(image_path=find_reference_image(video.directory_path.parent))

roi = ROI(video[0], select=True).square()

reference = reference.cropped(roi=roi)
video.preprocess = lambda frame: frame.cropped(roi=roi)

fcd = FCD(reference, square_size=effective_square_size, effective_height=effective_height)

# --------------------------------------------------------------------------
# SAVE CONFIG
# --------------------------------------------------------------------------

script_path = Path(__file__).resolve()
date_tag = datetime.now().strftime("%Y_%m_%d")
output_root = script_path.parent / "processed" / "fcd" / date_tag

path_dictionary = {
    "displaced_images_path": video.directory_path.resolve(),
    "reference_image_path": reference.path.resolve(),
    "script_path": script_path,
}

parameter_dictionary = {
    'ROI': roi, 
    'effective_height': effective_height, 
    'calibration_factor': fcd.calibration_factor,
    'parallax_correction': parallax_correction,
    'original_fps': video.fps,
    'step': ANALYSIS_STEP
}

save_dir, notes_path = create_output_directory(video, video.directory_path.parent, output_root)
write_analysis_notes(path_dictionary, parameter_dictionary, notes_path)

# --------------------------------------------------------------------------
# FCD RUN
# --------------------------------------------------------------------------

fig,_ = fcd.plot_fft(video[0])
fig.savefig(save_dir/"fft.pdf", dpi=200)
plt.show()

results = fcd.analyze_series(
    video, 
    video.fps, 
    step=ANALYSIS_STEP, 
    animate=True, 
    save_at=save_dir/"height_maps"
)

try:
    for t, i_def, height_map in tqdm(results):
        pass
finally:
    results.close()
    plt.ioff()
