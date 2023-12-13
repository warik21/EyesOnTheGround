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


def geo_to_pixel(lon, lat, geotransform):
    """Convert from geographic coordinates to pixel coordinates."""
    x_pixel = int((lon - geotransform[0]) / geotransform[1])
    y_pixel = int((lat - geotransform[3]) / geotransform[5])
    return x_pixel, y_pixel


# Create a blank image
image = cv2.imread('C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/eilatall_tp.tif',
                   cv2.IMREAD_UNCHANGED)
# image = cv2.resize(image, (500, 500))

# Create an instance of Observatory
camera1_path = 'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/eriktron/eriktron/src/Observatories/Camera1.txt'
camera1 = Observatory.load_from_file(camera1_path)
# camera1 = Observatory(**camera_data)
# Use attributes from camera1
# Assuming camera1.fov_horizontal is in degrees. If it's in radians, convert it to degrees
start_angle = 0
end_angle = start_angle + camera1.fov_horizontal

color = camera1.color  # Assuming color is in BGR format

# Other parameters
thickness = -1  # Fill the shape
center_coordinates = (250, 250)

# Create a mask and draw the slice of pie
mask_maximal = np.zeros_like(image)
mask_minimal = np.zeros_like(image)
cv2.ellipse(mask_maximal, center_coordinates, (camera1.distance_maximal, camera1.distance_maximal), 0, start_angle,
            end_angle, color, thickness)
cv2.ellipse(mask_minimal, center_coordinates, (camera1.distance_minimal, camera1.distance_minimal), 0, start_angle,
            end_angle, color, thickness)

mask = cv2.subtract(mask_maximal, mask_minimal)

# Blend the original image and the mask
result = cv2.addWeighted(image, 1, mask, 0.5, 0)

# Display the result
print(image.shape)
cv2.imshow('Slice of Pie', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
