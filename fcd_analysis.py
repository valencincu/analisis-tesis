#%% IMPORT LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from pathlib import Path
from matplotlib import use as matplotlib_backend
from tqdm import tqdm

from pyfcd.layer import Layer, refractive_indexes
from pyfcd.fcd import FCD

from pydata.video import Video
from pydata.roi import ROI
from pydata.image import *

from pyaux.helpers import create_output_directory, write_analysis_notes

matplotlib_backend("TkAgg")

def find_reference_image(import_dir):
    return next(
        import_dir.joinpath(filename)
        for filename in import_dir.glob("*.bmp")
        if "ref" in str(filename).lower()
    )

# %% FCD SETUP
H  = 80e-2  # m

layers = [
    Layer(2.2e-2, "Air"), 
    Layer(0.5e-2, "Glass"), 
    Layer(8.1e-2, "Water")
]

hp =  np.sum([layer.effective_height(refractive_indexes["Water"]) for layer in layers])

parallax_correction = (1 - hp / H)
square_size = 2e-3 * parallax_correction

alpha = 1 - refractive_indexes["Air"] / refractive_indexes["Water"]
effective_height =  (1 / (alpha*hp) - 1 / H)**(-1)

# %% VIDEO AND REFERENCE SETUP
video     = Video()
reference = Image(image_path=find_reference_image(video.directory_path.parent))

roi  = ROI(video[0], select=True).square()
step = 4

reference = reference.cropped(roi=roi)
video.preprocess = lambda frame: frame.cropped(roi=roi)

fcd = FCD(
    reference,
    square_size=square_size, 
    effective_height=effective_height
)

# %% SAVE CONFIG
script_path = Path(__file__).resolve()
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
    'step': step
}

save_dir, notes_path = create_output_directory(video, video.directory_path.parent, script_path.parent/"analyzed"/"fcd")
write_analysis_notes(path_dictionary, parameter_dictionary, notes_path)

# %% FCD RUN
fig,_ =fcd.plot_fft(video[0])
fig.savefig(save_dir/"fft.pdf", dpi=200)
plt.show()

results = fcd.analyze_series(video, video.fps, step=step, animate=True, save_at=save_dir/"height_maps")
try:
    for t, i_def, height_map in tqdm(results):
        pass
finally:
    results.close()
    plt.ioff()
