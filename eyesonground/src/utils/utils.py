import rasterio
import numpy as np
from affine import Affine
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
import math
import cv2

def pixel_to_geo(pixel_x, pixel_y, transform_matrix):
    geo_coords = np.dot(transform_matrix.T, [pixel_x, pixel_y, 1])
    return geo_coords[:2]  # This is actually lon, lat and only contains 2 elements.

def geo_to_pixel(lat, lon, inverse_transform_matrix):
    geo_coords_augmented = np.array([lat, lon, 1])  # Make it a 3-element vector
    pixel_coords = np.dot(inverse_transform_matrix, geo_coords_augmented)
    return np.round(pixel_coords[:2]).astype(int) 

def reset_coordinates(pixel_coords, geo_coords, geotiff_path, output_path):
    with rasterio.open(geotiff_path) as dataset:
        data = dataset.read()
        profile = dataset.profile
    
    gcps = []
    for (px, py), (lat, lon) in zip(pixel_coords, geo_coords):
        gcp = GroundControlPoint(row=py, col=px, x=lon, y=lat)
        gcps.append(gcp)

    # Compute Affine Transformation
    transform = from_gcps(gcps)

    # Update the Raster Metadata
    profile.update({
        'transform': transform,
        'crs': 'EPSG:2039'  # Update this with your CRS
    })

    # Save the Updated Raster
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)
        
def find_transform_matrix(pixel_coords, geo_coords):
    pixel_coords_augmented = np.concatenate([pixel_coords, np.ones((3, 1))], axis=1)
    # Solve for the transformation matrix
    # We're solving the matrix equation: pixel_coords_augmented * transform_matrix = geo_coords
    transform_matrix, _, _, _ = np.linalg.lstsq(pixel_coords_augmented, geo_coords, rcond=None)
    return transform_matrix

def find_inverse_transform(transform_matrix):
    transform_matrix_augmented = np.vstack([transform_matrix.T, [0, 0, 1]])
    inverse_transform_matrix = np.linalg.inv(transform_matrix_augmented)
    return inverse_transform_matrix

def test_pixel_to_geo(pixel_x, pixel_y, transform_matrix, pixel_lon, pixel_lat):
    calculated_geo_x, calculated_geo_y = pixel_to_geo(pixel_x, pixel_y, transform_matrix)
    print(f"Geographical Coordinates: Latitude = {calculated_geo_x}, Longitude = {calculated_geo_y}")
    # Check whether the geo coordinates are correct using np.allclose with tolerance of 0.001
    print(np.allclose([pixel_lon, pixel_lat], [calculated_geo_x, calculated_geo_y], atol=0.001))
    return True

def test_geo_to_pixel(new_geo_lon, new_geo_lat, inverse_transform_matrix, pixel_x, pixel_y):
    new_pixel_x, new_pixel_y = geo_to_pixel(new_geo_lon, new_geo_lat, inverse_transform_matrix)
    print(f"Pixel Coordinates: X = {new_pixel_x}, Y = {new_pixel_y}")
    # Check whether the pixel coordinates are correct using np.allclose, with a tolerance of 100 pixel
    print(np.allclose([pixel_x, pixel_y], [new_pixel_x, new_pixel_y], atol=10))
    return True

def get_transform_matrix(raster_file_path):
    """
    Get the affine transformation matrix from a raster file.

    :param raster_file_path: Path to the raster file.
    :return: Affine transformation matrix.
    """
    with rasterio.open(raster_file_path) as dataset:
        # Get the affine transformation matrix
        transform = dataset.transform
        resolution = dataset.res
        return transform, resolution
    
def affine_to_array(affine_obj):
    """
    Convert an affine transformation object to a 3x3 NumPy array.

    :param affine_obj: Affine object (from rasterio or similar).
    :return: 3x3 NumPy array representing the affine transformation.
    """
    return np.array([[affine_obj.d, affine_obj.a],
                     [affine_obj.e, affine_obj.b],
                     [affine_obj.f, affine_obj.c]]) 

def calculate_angle(pixel_location_observatory, pixel_location_top_left):
    """
    Calculate the angle from the observatory to the top-left point in pixel coordinates.

    :param pixel_location_observatory: Tuple (x, y) of the observatory's pixel location.
    :param pixel_location_top_left: Tuple (x, y) of the top-left point's pixel location.
    :return: Angle in degrees.
    """
    x_diff = pixel_location_top_left[0] - pixel_location_observatory[0]
    y_diff = pixel_location_top_left[1] - pixel_location_observatory[1]

    # Calculate angle in radians
    angle_radians = math.atan2(-y_diff, x_diff)  # Negative y_diff because pixel y-coordinates increase downwards

    # Convert angle to degrees
    angle_degrees = math.degrees(angle_radians)
    angle_degrees = 360 - angle_degrees  # Rotate 180 degrees because the y-axis is flipped

    # Normalize the angle to 0-360 degrees
    angle_degrees = (angle_degrees + 360) % 360

    return angle_degrees




# def setup_fov(observatory, scanning_area):
#     """
#     This function sets up the fov of the observatory.
#     :param observatory: The observatory object.
#     :param scanning_area: The scanning area object.
#     """
#     top_right_pixels = scanning_area.top_right_corner_pixels
#     pixel_distance = np.linalg.norm(np.array(top_right_pixels) - np.array(observatory.pixels))
#     observatory.distance_maximal = pixel_distance
#     observatory.calc_min_distance()  # Assuming this updates some internal state
#     observatory.start_angle = calculate_angle(observatory.pixels, top_right_pixels)
#     observatory.end_angle = observatory.start_angle + observatory.fov_horizontal
#     fov = observatory.get_fov()
#     fov.get_middle()
# 
# def draw_and_save(image, observatory, scanning_area, output_path):
#     """
#     This function draws the observatory and scanning area on the image and saves it.
#     :param image: The image to draw on.
#     :param observatory: The observatory object.
#     :param scanning_area: The scanning area object.
#     :param output_path: The path to save the image to.
#     """
#     image = scanning_area.draw(image, color=(0, 0, 255), thickness=10)
#     fov = observatory.fov
#     image = fov.draw(image, thickness=10)
#     
#     result_size = 2000
#     result_ratio = image.shape[0] / image.shape[1]
#     result_resized = cv2.resize(image, (result_size, int(result_size * result_ratio)))
#     cv2.imshow("Direction", result_resized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite(output_path, result_resized)
# 
# def draw_moving_fov(base_image, observatory, scanning_area, num_frames, output_video_file):
#     # Define the codec and create VideoWriter object
#     height, width, _ = base_image.shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust the codec as needed
#     video = cv2.VideoWriter(output_video_file, fourcc, 20.0, (width, height))
# 
#     for frame in range(num_frames):
#         # Update FOV position - implement this method to update the FOV position
#         observatory.update_position(...)
# 
#         # Copy base image for drawing
#         frame_image = base_image.copy()
# 
#         # Draw the FOV
#         fov = observatory.get_fov()  
#         frame_image = fov.draw(frame_image, thickness=10) 
# 
#         # Write the frame to the video
#         video.write(frame_image)
# 
#     video.release()