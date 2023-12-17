import pandas as pd
import numpy as np
import rasterio
from shapely.geometry import Point, MultiPolygon
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils.observatory import Observatory
import cv2
import math


# Create a blank image
im_path = r'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/eilat_new.tif'
image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
# image = cv2.resize(image, (500, 500))

# Create an instance of Observatory
camera1_path = 'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/Observatories/Camera1.txt'
camera2_path = 'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/Observatories/Camera2.txt'
#TODO: use glob for this, looking at all the files in Observatories
camera_paths = [camera1_path, camera2_path]

for camera_path in camera_paths:
    camera = Observatory.load_from_file(camera_path)
    # Create a mask for the observatory
    mask = camera.draw_mask(image.shape, im_path)

    # Blend the original image and the mask
    image = cv2.addWeighted(image, 1, mask, 0.5, 0)

result_ratio = image.shape[0] / image.shape[1]
# Display the result
result_resized = cv2.resize(image, (500, int(500 * result_ratio)))
cv2.imshow('Slice of Pie', result_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
