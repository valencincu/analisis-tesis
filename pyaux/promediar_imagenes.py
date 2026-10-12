import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

import cv2
import matplotlib.pyplot as plt
import numpy as np

# AUXILIARY FUNCTION
def normalize_image(img, scale=1, offset=0):
    img = img.astype(np.float32)
    range =  (img.max() - img.min())
    return offset*np.ones_like(img) if range==0 else scale * (img - img.min()) / range + offset

roi = None
allowed_formats = "tiff, tif, bmp, png"  


Tk().withdraw()
data_dir_name   = askdirectory(title="Select data directory") 
print(f"Averaging: {data_dir_name}")

save_dir_name   = askdirectory(title="Select save directory") 
print(f"Save dir: {save_dir_name}")

data_dir = os.path.abspath(data_dir_name)
save_dir =  os.path.abspath(save_dir_name)

image_paths = os.listdir(data_dir_name)

promediado = 0
N = 0
for image_path in image_paths:
    extension = image_path.split(".")[-1].lower() if "." in image_path else ""
    if extension in allowed_formats:
        i = cv2.imread(data_dir + os.sep + image_path, cv2.IMREAD_UNCHANGED)
        dtype = i.dtype
        i = np.array(i, dtype=np.float64)

        promediado += i
        N += 1

promediado /= N
promediado = promediado.astype(dtype)


plt.subplot(121)
plt.imshow(promediado)
plt.axis("off")
plt.show()

filename = data_dir_name.split(os.sep)[-1]
cv2.imwrite(save_dir + os.sep + f"{filename}_promedio." + extension, promediado)
