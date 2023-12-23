import pandas as pd
import numpy as np
import cProfile
import rasterio
from shapely.geometry import Point, MultiPolygon
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils.observatory import Observatory
import cv2
from glob import glob
from tqdm import tqdm


# def main():
#     im_path = r'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/eilat_new.tif'
#     image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
# 
#     # Create an instance of Observatory
#     camera1_path = 'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/Observatories/Camera1.txt'
#     camera2_path = 'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/Observatories/Camera2.txt'
#     #TODO: use glob for this, looking at all the files in Observatories
#     camera_paths = [camera1_path, camera2_path]
# 
#     for camera_path in camera_paths:
#         camera = Observatory.load_from_file(camera_path)
#         # Create a mask for the observatory
#         mask = camera.draw_mask(image.shape, im_path)
# 
#         # Blend the original image and the mask
#         image = cv2.addWeighted(image, 1, mask, 0.5, 0)
# 
# cProfile.run('main()')


# Create a blank image
result_size = 2000
im_path = r'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/eilat_new.tif'
image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

# Define the range of positions for the camera
num_steps = 120
positions = np.linspace(0, 1, num_steps)

# Get the camera paths
# camera_paths = glob('C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/Observatories_comparison/*.txt')
camera_paths = glob('C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/Observatories_still/*.txt')
cameras = [Observatory.load_from_file(camera_path) for camera_path in camera_paths]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
result_ratio = image.shape[0] / image.shape[1]
#vid_path = r'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/camera_movement_scan.mp4'
#vid_path = r'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/camera_movement_still.mp4'
#out = cv2.VideoWriter(vid_path, fourcc, 3.0, (result_size, int(result_size * result_ratio)))


for camera in cameras:
    # Create a mask for the observatory
    mask = camera.draw_mask(image.shape, im_path, 0.5)

    # Blend the original image and the mask
    image = cv2.addWeighted(image, 1, mask, 0.5, 0)

result_resized = cv2.resize(image, (result_size, int(result_size * result_ratio)))
cv2.imwrite('C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/camera_movement_still.jpg', result_resized)

out = cv2.VideoWriter('C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/camera_movement_still.mp4', fourcc, 3.0, (result_size, int(result_size * result_ratio)))
for pos in tqdm(positions):
    new_image = image.copy()
    for camera in cameras:
        # Create a mask for the observatory
        mask = camera.draw_mask(image.shape, im_path, pos)

        # Blend the original image and the mask
        new_image = cv2.addWeighted(new_image, 1, mask, 0.5, 0)

    result_resized = cv2.resize(new_image, (result_size, int(result_size * result_ratio)))
    # Write the frame
    out.write(result_resized)

out.release()
cv2.destroyAllWindows()
