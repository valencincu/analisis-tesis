from pathlib import Path
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pyaux.output_config import create_output_directory

from pydata.roi import ROI
from pydata.image import Image
from pydata.video import Video

from pypiv.runner import generate_background, preprocess

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

BACKGROUND_AVERAGE_STEP = 10

CONTRAST_STRETCH_LOWER = 50
CONTRAST_STRETCH_UPPER = 100

START_FRAME = 0
END_FRAME = 2000

# --------------------------------------------------------------------------
# SELECT VIDEO
# --------------------------------------------------------------------------

video = Video()
data_directory = video.directory_path

# --------------------------------------------------------------------------
# EXPORT PATH SETUP
# --------------------------------------------------------------------------

script_path  = Path(__file__).resolve()
date_tag = datetime.now().strftime("%Y_%m_%d")
output_root = script_path.parent / "processed" / "stacking" / date_tag
export_dir,_ = create_output_directory(video, video.directory_path, output_root)

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
# TRACE TRAJECTORIES
# --------------------------------------------------------------------------

noise_image = video[0].copy()
stacked_image = video[0].copy()

# Get noise by selecting least bright pixel in time series.
# Get trajectories by selecting brightest pixel in time series. 
for frame in video.play(start_frame=START_FRAME, end_frame=END_FRAME):
    noise_image[noise_image > frame] = frame[noise_image > frame]
    stacked_image[stacked_image < frame] = frame[stacked_image < frame]

# Substract noise.
stacked_image -= noise_image
stacked_image = stacked_image.normalized()

# --------------------------------------------------------------------------
# PLOT TRAJECTORIES
# --------------------------------------------------------------------------

plt.figure(figsize=(20,20))
plt.imshow(stacked_image, cmap="Greys")
plt.axis('off')
plt.show()

# --------------------------------------------------------------------------
# SAVE
# --------------------------------------------------------------------------

export_file_name = "stacking_" + data_directory.name + f"START{START_FRAME}_END{END_FRAME}_LOWER{CONTRAST_STRETCH_LOWER}_UPPER{CONTRAST_STRETCH_UPPER}.bmp"
cv2.imwrite(export_dir / export_file_name, cv2.bitwise_not((stacked_image*255).astype(np.uint8)))