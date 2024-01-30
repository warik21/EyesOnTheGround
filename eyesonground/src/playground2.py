import rasterio
from rasterio.transform import from_origin
import numpy as np
from utils.utils import *
from geopy.distance import great_circle
import numpy as np
from utils.observatory import Observatory
from geopy.point import Point
import math
import cv2


class ScanningArea:
    def __init__(self, corner1, corner2):
        """
        Initializes a ScanningArea object with a rectangular shape based on two diagonal corners.

        :param corner1: Tuple of (latitude, longitude) for the first corner.
        :param corner2: Tuple of (latitude, longitude) for the opposite corner.
        """
        self.corner1 = Point(corner1)
        self.corner2 = Point(corner2)
        self._calculate_rectangle()

    def _calculate_rectangle(self):
        """
        Calculate the rectangle's corners based on two diagonal corners.
        """
        # Determine the top left and bottom right corners based on latitude and longitude
        top_left = Point(max(self.corner1.latitude, self.corner2.latitude), 
                         min(self.corner1.longitude, self.corner2.longitude))
        bottom_right = Point(min(self.corner1.latitude, self.corner2.latitude), 
                             max(self.corner1.longitude, self.corner2.longitude))

        self.top_left_corner = (top_left.latitude, top_left.longitude)
        self.bottom_right_corner = (bottom_right.latitude, bottom_right.longitude)
        self.top_right_corner = (top_left.latitude, bottom_right.longitude)
        self.bottom_left_corner = (bottom_right.latitude, top_left.longitude)

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

    cv2.arrowedLine(mask, start_point, (end_x, end_y), color, thickness)

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

    # Normalize the angle to 0-360 degrees
    angle_degrees = (angle_degrees + 360) % 360

    return angle_degrees

    
image_path = r'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/images_and_reults/eilat_updated.tif'

# Define the location of the observatory
third_point_coords = 29.552339, 34.956392  #top left (lat, lon)
third_point_pixels = 12659, 4263  #bottom left (x, y)
observatory = Observatory(latitude=third_point_coords[0], longitude=third_point_coords[1])

# Define the scanning area
top_right = 29.548411, 34.974032
bottom_left = 29.543247, 34.952230
scanning_area = ScanningArea(top_right, bottom_left)

starting_point = scanning_area.top_right_corner
real_world_distance = great_circle(observatory.coordinates(), scanning_area.top_right_corner).meters
# To find the distance in pixels, we need to use our image's transform matrix
# We'll use the inverse transform matrix to convert from pixels to geo
transform_matrix, resolution = get_transform_matrix(image_path)
transform_matrix = affine_to_array(transform_matrix)

inv_transform_matrix = inv_transform_matrix = find_inverse_transform(transform_matrix)
observatory_pixels = geo_to_pixel(observatory.latitude, observatory.longitude, inv_transform_matrix)
scanning_area_top_right_pixels = geo_to_pixel(starting_point[0], starting_point[1], inv_transform_matrix)

pixel_distance = np.sqrt((scanning_area_top_right_pixels[1]-observatory_pixels[1])**2 + (scanning_area_top_right_pixels[0]-observatory_pixels[0])**2)  # Euclidean distance
# pixel_distance = calculate_pixel_distance(real_world_distance, resolution)
observatory.distance_maximal = pixel_distance

# Calculate the inner arc:
small_radius = observatory.calc_min_distance()

# Calculate the angle:
observatory.start_angle = calculate_angle(observatory_pixels, scanning_area_top_right_pixels)
observatory.end_angle = observatory.start_angle + observatory.fov_horizontal

# Example usage
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
image_size = (image.shape[0], image.shape[1])  # Width, heig
start_point = observatory_pixels  # Center of the image

# Draw the arrow and display the image
arrow_image = draw_direction(image_size, start_point, observatory)
result = cv2.addWeighted(image, 1, arrow_image, 0.5, 0)
result_ratio = image.shape[0] / image.shape[1]
result_size = 500
result_resized = cv2.resize(result, (result_size, int(result_size * result_ratio)))
# result_resized2 = cv2.resize(image, (result_size, int(result_size * result_ratio)))
cv2.imshow("Direction", result_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

