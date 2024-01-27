import rasterio
from rasterio.transform import from_origin
import numpy as np
import pyproj
from utils.utils import *

zvia_coords = 29.544043, 34.975739  #top right (lat, lon)
zvia_pixels = 16269, 6045  #bottom right (x, y)

taba_coords = 29.490278, 34.901843  #bottom left (lat, lon) - gova, rohav
taba_pixels = 2506, 17567  #top left (x, y)

# Adding a third point for accuracy
third_point_coords = 29.552339, 34.956392  #top left (lat, lon)
third_point_pixels = 12659, 4263  #bottom left (x, y)


# Usage
im_path = 'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/images_and_reults/eilat_new.tif'
output_path = 'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/images_and_reults/eilat_updated.tif'
pixel_coords = np.array([taba_pixels, zvia_pixels, third_point_pixels])
geo_coords = np.array([taba_coords, zvia_coords, third_point_coords])

transform_matrix = find_transform_matrix(pixel_coords, geo_coords)

# Example/test for later for pixel to geo
new_pixel_x, new_pixel_y = 10528, 1497
test_pixel_to_geo(new_pixel_x, new_pixel_y, transform_matrix, 29.565234, 34.944937)

# Example/test for later for geo to pixel
new_geo_lon, new_geo_lat = 29.565234, 34.944937
inverse_transform_matrix = find_inverse_transform(transform_matrix)
test_geo_to_pixel(new_geo_lon, new_geo_lat, inverse_transform_matrix, 10528, 1497)

# Test the new coordinates
with rasterio.open(output_path, 'r') as dataset:
    # Transform the observatory's geographic coordinates to the image's coordinate system
    y, x = dataset.index(zvia_coords[1], zvia_coords[0])  # dataset.index(lon, lat) -> (x, y)
    print(f'Zvia coordinates in the new image are {x}, {y}')

    y, x = dataset.index(taba_coords[1], taba_coords[0])
    print(f'Taba coordinates in the new image are {x}, {y}')


print('Done')
israeli_grid = pyproj.Proj(init='epsg:2039')
