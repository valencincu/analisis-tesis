import os
import glob
from PIL import Image

frame_folder = "gif"
 
# create an empty list to store the frames
frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]

save_loc = input("Save location: ")
# save as new gif
frames[0].save(save_loc + os.sep + "gif.gif", save_all=True, append_images=frames[1:], optimize=False, duration=16, loop=0)