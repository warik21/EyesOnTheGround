import rasterio
from rasterio.transform import from_origin
import numpy as np

zvia_coords = 29.544043, 34.975739  #top right (lat, lon)
zvia_pixels = 16269, 6045  #bottom right (x, y)

taba_coords = 29.490278, 34.901843  #bottom left (lat, lon) - gova, rohav
taba_pixels = 2506, 17567  #top left (x, y)



def reset_coordinates(ref_point1_pixel, ref_point1_coord, ref_point2_pixel, ref_point2_coord, geotiff_path, output_path):
    with rasterio.open(geotiff_path) as dataset:
        # Calculate the pixel size
        pixel_width = (ref_point2_coord[1] - ref_point1_coord[1]) / (ref_point2_pixel[0] - ref_point1_pixel[0])
        pixel_height = (ref_point1_coord[0] - ref_point2_coord[0]) / (ref_point1_pixel[1] - ref_point2_pixel[1])

        # Since y-axis in image coordinates increases downwards (opposite to geographic latitude)
        # the pixel height should be negative.
        #  pixel_height = -abs(pixel_height)

        # Calculate the top-left corner of the top-left pixel in geographic coordinates
        # This is done by back-calculating from one of the reference points
        top_left_lon = ref_point1_coord[1] - ref_point1_pixel[0] * pixel_width  
        top_left_lat = ref_point1_coord[0] - ref_point1_pixel[1] * pixel_height
        print(f'Top left corner of the image is {top_left_lon}, {top_left_lat}')
        # Create the affine transformation
        transform = from_origin(top_left_lat, top_left_lon, -pixel_width, pixel_height)

        # Read data and update metadata
        data = dataset.read()
        new_meta = dataset.meta.copy() 
        new_meta.update({"transform": transform})

        # Write the data to a new file with updated metadata
        #TODO:Saving to the new file does not give good results, but instead looks awful.
        with rasterio.open(output_path, 'w', **new_meta) as dst:
            dst.write(data)

# Usage
im_path = 'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/images_and_reults/eilat_new.tif'
output_path = 'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/images_and_reults/eilat_updated.tif'
reset_coordinates(zvia_pixels, zvia_coords, taba_pixels, taba_coords, im_path, output_path)

# Test the new coordinates
with rasterio.open(output_path, 'r') as dataset:
    # Transform the observatory's geographic coordinates to the image's coordinate system
    x, y = dataset.index(zvia_coords[0], zvia_coords[1])
    print(f'Zvia coordinates in the new image are {x}, {y}')

    x, y = dataset.index(taba_coords[0], taba_coords[1])
    print(f'Taba coordinates in the new image are {x}, {y}')

