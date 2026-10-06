#%% IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path
from shutil import copyfile

from pydata.video import Video
from pydata.roi import ROI
from pypiv.piv import PIV, contrast_stretch, high_pass, intensity_clip, generate_background, local_norm_median_val

from matplotlib import use as matplotlib_backend
matplotlib_backend("TkAgg")

#%% SELECT VIDEO
video = Video()

# %% EXPORT PATH SETUP
current_dir  = Path(os.path.dirname(os.path.realpath(__file__)))
export_dir   = current_dir / "analyzed" / "piv"

job_id        = input("Enter job identifier: ", )
save_dir = export_dir.joinpath(video.directory_path.name + " - " + job_id)
save_dir.mkdir(parents=True, exist_ok=False)
copyfile(next(video.directory_path.glob("*.cih")), save_dir / "metadata.cih")

notes_path = save_dir / "piv_analysis_record.txt"
preprocess_string = open("piv_analysis.py", "r").read()
notes_path.write_text(preprocess_string)

#%% PREPROCESS
first_frame = video[0]
roi = ROI(first_frame, select=True).square()
background = generate_background(video, step=10)

def preprocess(image):
    processed_image = image.astype(np.float64) - background
    processed_image = processed_image.cropped(roi=roi)
    processed_image = processed_image.inverted()
    processed_image = processed_image.normalized()
    processed_image = contrast_stretch(processed_image)
    processed_image = intensity_clip(processed_image, min_val=np.mean(processed_image) + np.std(processed_image))
    processed_image = high_pass(processed_image, clip=True)
    return processed_image

video.preprocess = preprocess

#%% FILTER FUNCTION
def filter(u, v):
    flags = local_norm_median_val(u, v, 0.1, 2)
    return flags

#%% RUN PIV
piv = PIV(
    video=video, 
    fps=video.fps,
    interval=12, 
    window_size=24,
    search_size=32,
    overlap=12,
    sig2noise_threshold=2,
    filter_func=filter
)

#%% COMPARE TWO FRAMES THE PIV ALGORITHM USES
piv.compare_frames(0, show=False)
plt.savefig(save_dir/"frame_comparison.pdf", dpi=200)
plt.show()

#%% RUN PIV
for time, u, v in piv.run(step=4, animate=True, save_at=save_dir/"piv"):
    pass