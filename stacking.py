from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydata.image import Image

export_directory = Path("analyzed") / "stacking"

# Select directory
Tk().withdraw() 
data_directory = Path(askdirectory(title="Select data directory"))
print(f"Analyzing: {data_directory}")
image_paths = list(data_directory.glob("*.tif"))

# Configure image preprocessing
def get_processed_image(image_path, invert=True, gamma=2, roi=None, return_roi=False):
    image = Image(image_path=image_path)
    processed_image, roi = image.cropped(roi=roi, return_roi=True)
    processed_image = processed_image.inverted() if invert else processed_image
    processed_image = processed_image.normalized()
    processed_image = processed_image**gamma
    return (processed_image, roi) if return_roi else processed_image

# Process first frame. Select roi.
gamma  = 2
invert = True
processed_image, roi = get_processed_image(image_paths[0], return_roi=True, gamma=gamma, invert=invert)
noise_image   = processed_image.copy()
stacked_image = processed_image.copy()

# Get trajectories by selecting brightest pixel in time series. Get noise by selecting least bright pixel in time series.
N = 2000 # Numero de cuadros a procesar
for image_path in image_paths[1:N]:
    new_image = get_processed_image(image_path, roi=roi, gamma=gamma, invert=invert)
    noise_image[noise_image > new_image]     = new_image[noise_image > new_image]
    stacked_image[stacked_image < new_image] = new_image[stacked_image < new_image]

# Substract noise.
stacked_image -= noise_image
stacked_image = stacked_image.normalized()

# Increase contrast and saturate.
contrast = 1.5
stacked_image *= contrast
stacked_image[stacked_image > 1.0] = 1.0

plt.figure(figsize=(20,20))
plt.imshow(stacked_image, cmap="Greys")
plt.axis('off')
plt.show()

export_file_name = "stacking_" + str(data_directory.name) + f"_frames{N}_gamma{gamma}_contrast{contrast}.bmp"
cv2.imwrite(export_directory / export_file_name, cv2.bitwise_not((stacked_image*255).astype(np.uint8)))