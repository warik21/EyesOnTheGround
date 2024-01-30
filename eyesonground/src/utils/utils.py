import rasterio
import numpy as np
from affine import Affine
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
import math

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
    
def draw_direction(image_size, start_point, observatory, length=100, color=(255, 0, 0), thickness=2):
    """
    Draw an arrow representing the direction on a blank image.

    :param image_size: Size of the image (width, height).
    :param start_point: Starting point (x, y) for the arrow.
    :param angle: Angle in degrees.
    :param length: Length of the arrow.
    :param color: Color of the arrow (B, G, R).
    :param thickness: Thickness of the arrow.
    :return: Image with the arrow.
    """
    # Create a blank image
    mask_maximal: np.ndarray = np.zeros((image_size[0], image_size[1], 3), dtype='uint8')
    mask_minimal: np.ndarray = np.zeros((image_size[0], image_size[1], 3), dtype='uint8')

    # Calculate end point of the arrow
    end_x = int(start_point[0] + length * math.cos(math.radians(observatory.start_angle)))
    end_y = int(start_point[1] + length * math.sin(math.radians(observatory.start_angle)))  # Subtract because y increases downwards

    # Draw the ellipses:
    # cv2.ellipse(mask_maximal, start_point, (observatory.distance_maximal, observatory.distance_maximal), 0,
    #             startAngle=observatory.start_angle, endAngle=observatory.end_angle, color=color, thickness=-1)
    cv2.ellipse(mask_minimal, start_point, (int(observatory.distance_minimal), int(observatory.distance_minimal)), 0,
                startAngle=observatory.start_angle, endAngle=observatory.end_angle, color=color, thickness=-1)

    cv2.ellipse(mask_maximal, start_point, (int(observatory.distance_maximal), int(observatory.distance_maximal)), 0,
                startAngle=observatory.start_angle, endAngle=observatory.end_angle, color=color, thickness=-1)

    mask = cv2.subtract(mask_maximal, mask_minimal)
    # Draw the lines

    cv2.arrowedLine(mask, start_point, (end_x, end_y), (0,0,255), thickness)

    return mask

def calculate_pixel_distance(real_world_distance, resolution):
    """
    Calculate the distance in pixels based on the real world distance and the resolution.

    :param real_world_distance: The distance in meters.
    :param resolution: The resolution of the image.
    :return: The distance in pixels.
    """
    return real_world_distance / resolution[0]

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


    
    