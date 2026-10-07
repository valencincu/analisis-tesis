# %% IMPORT LIBRARIES
from matplotlib import use as matplotlib_backend
matplotlib_backend("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from tqdm import tqdm

from pyaux.helpers import create_output_directory, write_analysis_notes
from pydata.video import Video
from pydata.roi import ROI
from pypiv.piv import PIV, generate_background, validation, preprocess

# %% SELECT VIDEO
video = Video()
step  = 4

# %% EXPORT PATH SETUP
script_path  = Path(__file__).resolve()
save_dir, notes_path = create_output_directory(video, video.directory_path, script_path.parent / "analyzed" / "piv")

# %% PREPROCESS
first_frame = video[0]
roi = ROI(first_frame, select=True).square()
background = generate_background(video, step=10)

def preprocess_func(image):
    processed_image = image.astype(np.float32) - background
    processed_image = processed_image.cropped(roi=roi)
    processed_image = processed_image.inverted()
    processed_image = processed_image.normalized()
    processed_image = preprocess.contrast_stretch(processed_image)
    processed_image = preprocess.intensity_clip(processed_image, min_val=np.mean(processed_image) + np.std(processed_image))
    processed_image = preprocess.high_pass(processed_image, clip=True)
    return processed_image

video.preprocess = preprocess_func

# %% FILTER FUNCTION
def filter_func(u, v):
    flags = validation.local_norm_median_val(u, v, 0.1, 2)
    return flags

# %% RUN PIV
piv = PIV(
    video=video, 
    fps=video.fps,
    interval=12, 
    window_size=24,
    search_size=32,
    overlap=12,
    sig2noise_threshold=2,
    filter_func=filter_func
)

path_dictionary = {
    "video_path": str(video.directory_path.resolve()),
    "script_path": str(Path(__file__).resolve()),
}
write_analysis_notes(path_dictionary, {'roi': roi, 'original_fps': video.fps, 'step': step}, notes_path)

# %% COMPARE TWO FRAMES THE PIV ALGORITHM USES
ax = piv.compare_frames(0, show=False)
ax.figure.savefig(save_dir/"frame_comparison.pdf", dpi=200)
plt.show()

#%% RUN PIV
results = piv.run(step=step, animate=True, save_at=save_dir/"piv")
try:
    for time, u, v in tqdm(results):
        pass
finally:
    results.close()
    plt.ioff()