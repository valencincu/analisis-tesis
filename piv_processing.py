from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from openpiv import validation
from matplotlib import use as matplotlib_backend
from tqdm import tqdm

from pyaux.output_config import create_output_directory, write_analysis_notes
from pydata.roi import ROI
from pydata.video import Video

from pypiv.algorithms import generate_background
from pypiv.runner import PIVRunner
from pypiv.settings import SimplePIVSettings, WindefPIVSettings

matplotlib_backend("TkAgg")
plt.style.use('science')

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

DEFAULT_STEP = 4
BACKGROUND_AVERAGE_STEP = 10

CONTRAST_STRETCH_LOWER = 50
CONTRAST_STRETCH_UPPER = 100

PIV_INTERVAL = 12
PIV_WINDOW_SIZES = (64, 32, 16)
PIV_OVERLAPS = (32, 16, 8)
PIV_SIG2NOISE_THRESHOLD = 2.0

FILTER_SCALE = 0.1
FILTER_NEIGHBORHOOD = 2

CALIBRATION_FACTOR = 1/3400

# --------------------------------------------------------------------------
# SELECT VIDEO
# --------------------------------------------------------------------------

video = Video()

# --------------------------------------------------------------------------
# PREPROCESS
# --------------------------------------------------------------------------

first_frame = video[0]
roi = ROI(first_frame, select=True).square()
background = generate_background(video, step=BACKGROUND_AVERAGE_STEP)

def preprocess_func(image):
    processed_image = image.astype(np.float32) - background
    processed_image = processed_image.cropped(roi=roi)
    processed_image = processed_image.inverted().normalized()
    processed_image = processed_image.contrast_stretch(
        lower_limit=CONTRAST_STRETCH_LOWER, 
        upper_limit=CONTRAST_STRETCH_UPPER
    )
    return processed_image

video.preprocess = preprocess_func

# --------------------------------------------------------------------------
# VELOCITY FIELD FILTER FUNCTION
# --------------------------------------------------------------------------

def validation_func(u, v, s2n):

    flags = validation.sig2noise_val(s2n, threshold=PIV_SIG2NOISE_THRESHOLD)

    flags |= validation.local_norm_median_val(u, v, FILTER_SCALE, FILTER_NEIGHBORHOOD)

    return flags

# --------------------------------------------------------------------------
# PIV CONFIGURATION
# --------------------------------------------------------------------------

settings = WindefPIVSettings(
    window_sizes=(64, 32, 16),
    overlaps=(32, 16, 8),
    validation_function=validation_func,
    use_smoothn=True,
)

piv = PIVRunner(
    video=video,
    settings=settings,
    fps=video.fps,
    interval=PIV_INTERVAL,
    calibration_factor=CALIBRATION_FACTOR
)

# --------------------------------------------------------------------------
# SAVE CONFIG
# --------------------------------------------------------------------------

script_path  = Path(__file__).resolve()
date_tag = datetime.now().strftime("%Y_%m_%d")
output_root = script_path.parent / "processed" / "piv" / date_tag
save_dir, notes_path = create_output_directory(video, video.directory_path, output_root)

path_dictionary = {
    "video_path": str(video.directory_path.resolve()),
    "script_path": str(Path(__file__).resolve()),
}
write_analysis_notes(notes_path, path_dictionary, {'roi': roi, 'original_fps': video.fps, 'step': DEFAULT_STEP}, extra=settings.__repr__())

# --------------------------------------------------------------------------
# CHECK: PLOT ORIGINAL AND PREPROCESSED FIRST FRAME
# --------------------------------------------------------------------------

fig, axs = plt.subplots(1, 2)
axs[0].set_title("Original")
axs[0].imshow(first_frame)

axs[1].set_title("Preprocessed")
axs[1].imshow(video[0])

fig.savefig(save_dir/"preprocessing.pdf", dpi=200)
plt.show()

# --------------------------------------------------------------------------
# CHECK: COMPARE CONSECUTIVE FRAMES
# --------------------------------------------------------------------------

ax = piv.compare_frames(0, show=False)
ax.figure.savefig(save_dir/"frame_comparison.pdf", dpi=200)
plt.show()

# --------------------------------------------------------------------------
# RUN PIV
# --------------------------------------------------------------------------

results = piv.run(step=DEFAULT_STEP, animate=True, save_at=save_dir/"piv")
try:
    for time, u, v, flags in tqdm(results):
        pass
finally:
    results.close()
    plt.ioff()