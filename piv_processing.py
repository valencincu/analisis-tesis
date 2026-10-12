from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib import use as matplotlib_backend
from tqdm import tqdm

from pyaux.output_config import create_output_directory, write_analysis_notes
from pydata.roi import ROI
from pydata.video import Video
from pypiv.piv import PIV, generate_background, preprocess, validation

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
PIV_WINDOW_SIZE = 24
PIV_SEARCH_SIZE = 32
PIV_OVERLAP = 12
PIV_SIG2NOISE_THRESHOLD = 2

FILTER_SCALE = 0.1
FILTER_NEIGHBORHOOD = 2

# --------------------------------------------------------------------------
# SELECT VIDEO
# --------------------------------------------------------------------------

video = Video()

# --------------------------------------------------------------------------
# EXPORT PATH SETUP
# --------------------------------------------------------------------------

script_path  = Path(__file__).resolve()
date_tag = datetime.now().strftime("%Y_%m_%d")
output_root = script_path.parent / "processed" / "piv" / date_tag
save_dir, notes_path = create_output_directory(video, video.directory_path, output_root)

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
    processed_image = preprocess.contrast_stretch(
        processed_image, 
        lower_limit=CONTRAST_STRETCH_LOWER, 
        upper_limit=CONTRAST_STRETCH_UPPER
    )
    return processed_image

video.preprocess = preprocess_func

# --------------------------------------------------------------------------
# VELOCITY FIELD FILTER FUNCTION
# --------------------------------------------------------------------------

def filter_func(u, v):
    return validation.local_norm_median_val(
        u,
        v,
        FILTER_SCALE,
        FILTER_NEIGHBORHOOD,
    )

# --------------------------------------------------------------------------
# PIV CONFIGURATION
# --------------------------------------------------------------------------

piv = PIV(
    video=video,
    fps=video.fps,
    interval=PIV_INTERVAL,
    window_size=PIV_WINDOW_SIZE,
    search_size=PIV_SEARCH_SIZE,
    overlap=PIV_OVERLAP,
    sig2noise_threshold=PIV_SIG2NOISE_THRESHOLD,
    filter_func=filter_func,
)

path_dictionary = {
    "video_path": str(video.directory_path.resolve()),
    "script_path": str(Path(__file__).resolve()),
}
write_analysis_notes(path_dictionary, {'roi': roi, 'original_fps': video.fps, 'step': DEFAULT_STEP}, notes_path)

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
    for time, u, v in tqdm(results):
        pass
finally:
    results.close()
    plt.ioff()