import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

roi = None
allowed_formats = "tiff, tif, bmp, png"  


Tk().withdraw()
data_dir_name   = askdirectory(title="Select data directory") 
print(f"Analyzing: {data_dir_name}")

save_dir_name   = askdirectory(title="Select save directory") 
print(f"Analyzing: {save_dir_name}")

data_dir = os.path.abspath(data_dir_name)
save_dir =  os.path.abspath(save_dir_name)

image_paths = os.listdir(data_dir_name)

promediado = 0
N = 0
for image_path in image_paths:
    extension = image_path.split(".")[-1].lower() if "." in image_path else ""
    if extension in allowed_formats:
        i = cv2.imread(data_dir + os.sep + image_path, cv2.IMREAD_UNCHANGED)
        i = np.array(i, dtype=np.float64)

        promediado += i
        N += 1

promediado /= N

print(np.max(promediado), np.min(promediado))

plt.subplot(121)
plt.imshow(promediado)
plt.axis("off")
plt.show()


filename = data_dir_name.split(os.sep)[-1]
cv2.imwrite(save_dir + os.sep + f"{filename}_promedio.bmp", promediado.astype(np.uint8))
