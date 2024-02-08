from geopy.point import Point
import math
from utils.utils import geo_to_pixel
import cv2
import json
import numpy as np

class ScanningArea:
    def __init__(self, latitude1, latitude2, longitude1, longitude2, 
                 top_left_corner=None, bottom_right_corner=None, 
                 top_right_corner=None, bottom_left_corner=None):
        """
        Initializes a ScanningArea object with a rectangular shape based on two diagonal corners.

        :param corner1: Tuple of (latitude, longitude) for the first corner.
        :param corner2: Tuple of (latitude, longitude) for the opposite corner.
        """
        self.latitude_bounds = (min(latitude1, latitude2), max(latitude1, latitude2)) # min is west, max is east
        self.longitude_bounds = (min(longitude1, longitude2), max(longitude1, latitude2)) # min is south, max is north
        self.get_corners()

    def get_corners(self):
        """
        Calculate the rectangle's corners based on two diagonal corners.
        """
        self.top_left_corner_geo = (self.latitude_bounds[1], self.longitude_bounds[0])
        self.bottom_right_corner_geo = (self.latitude_bounds[0], self.longitude_bounds[1])
        self.top_right_corner_geo = (self.latitude_bounds[1], self.longitude_bounds[1])
        self.bottom_left_corner_geo = (self.latitude_bounds[0], self.longitude_bounds[0])

    def get_pixel_corners(self, transform_matrix):
        """
        Calculate the rectangle's corners in pixel coordinates.
        :param transform_matrix: The inverse transform matrix of the image, transforming coordinates into pixels.
        """
        self.top_left_corner_pixels = geo_to_pixel(self.top_left_corner_geo[0], self.top_left_corner_geo[1], transform_matrix)
        self.bottom_right_corner_pixels = geo_to_pixel(self.bottom_right_corner_geo[0], self.bottom_right_corner_geo[1], transform_matrix)
        self.top_right_corner_pixels = geo_to_pixel(self.top_right_corner_geo[0], self.top_right_corner_geo[1], transform_matrix)
        self.bottom_left_corner_pixels = geo_to_pixel(self.bottom_left_corner_geo[0], self.bottom_left_corner_geo[1], transform_matrix)
        # Calculate the pixel bounds:
        self.pixel_x_bounds = (min(self.top_left_corner_pixels[0], self.bottom_right_corner_pixels[0]), max(self.top_left_corner_pixels[0], self.bottom_right_corner_pixels[0]))
        self.pixel_y_bounds = (min(self.top_left_corner_pixels[1], self.bottom_right_corner_pixels[1]), max(self.top_left_corner_pixels[1], self.bottom_right_corner_pixels[1]))

    @classmethod
    def load_from_file(cls, json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)

        return cls(**data)

    def draw(self, image, color=(255, 0, 0), thickness=10):
        """
        Draw a rectangle on an image.

        :param image: The image to draw on.
        :return: The image with the rectangle.
        """
        # Draw the rectangle
        cv2.rectangle(image, self.top_left_corner_pixels, self.bottom_right_corner_pixels, color, thickness)

        return image
    
def load_and_prepare_scanning_area(file_path, inv_transform_matrix):
    """
    This function loads a scanning area from a json file and calculates its pixel corners.
    :param file_path: The path to the json file.
    :param inv_transform_matrix: The inverse transform matrix of the image, transforming coordinates into pixels.
    :return: The scanning area object.
    """
    scanning_area = ScanningArea.load_from_file(file_path)
    scanning_area.get_pixel_corners(inv_transform_matrix)
    return scanning_area
    
class Annulus:
    """
    This class is meant to draw and work on the annulus which represents the FOV of the observatory.
    """
    def __init__(self, center, distance_maximal, distance_minimal, start_angle, end_angle):
        """
        Initializes an Annulus object representing the area between two ellipses.

        :param center: Tuple (x, y) representing the center of the ellipses in pixels.
        :param distance_maximal: The major radius of the outer ellipse.
        :param distance_minimal: The major radius of the inner ellipse.
        :param start_angle: The starting angle of the elliptical section in degrees.
        :param end_angle: The ending angle of the elliptical section in degrees.
        """
        self.center = center
        self.distance_maximal = distance_maximal
        self.distance_minimal = distance_minimal
        self.start_angle = start_angle
        self.end_angle = end_angle

    def contains_point(self, point):
        """
        Check if a point is inside the annulus with a margin of error.

        :param point: Tuple (x, y) representing the point to check.
        :return: Boolean indicating if the point is inside the annulus.
        """
        x, y = np.array(point) - np.array(self.center)
        distance_squared = x**2 + y**2

        # Check if within outer ellipse with a 1% margin
        in_outer = distance_squared <= (self.distance_maximal**2) * 1.01

        # Check if outside inner ellipse with a 1% margin
        out_inner = distance_squared > (self.distance_minimal**2) * 0.99

        return in_outer and out_inner

    def contains_points(self, points):
        """
        Check if a list of points is inside the annulus with a margin of error.

        :param points: List of tuples (x, y) representing the points to check.
        :return: Boolean indicating if the points are inside the annulus.
        """
        return all([self.contains_point(point) for point in points])
    
    def contains_points_fast(self, points):
        """
        Check if a list of points is inside the annulus with a margin of error.

        :param points: List of tuples (x, y) representing the points to check.
        :return: Boolean indicating if the points are inside the annulus.
        """
        x, y = np.array(points) - np.array(self.center)
        distance_squared = x**2 + y**2

        # Check if within outer ellipse with a 1% margin
        in_outer = distance_squared <= (self.distance_maximal**2) * 1.01

        # Check if outside inner ellipse with a 1% margin
        out_inner = distance_squared > (self.distance_minimal**2) * 0.99

        return np.logical_and(in_outer, out_inner)

    def get_middle(self):
        """
        Find the middle point of the annulus.
        """
        dist = (self.distance_maximal + self.distance_minimal) / 2
        angle = (self.start_angle + self.end_angle) / 2
        # add the distance in the direction of the angle
        x = self.center[0] + dist * math.cos(math.radians(angle))
        y = self.center[1] + dist * math.sin(math.radians(angle))
        self.middle = (int(np.round(x)), int(np.round(y))) 
        return self.middle

    def draw(self, image, color=(255, 0, 0), thickness=10):
        # Create a blank mask with the same dimensions as the input image
        mask = np.zeros(image.shape, dtype=np.uint8)

        # Draw the ellipses on the mask instead of the image
        cv2.ellipse(mask, self.center, 
                    (int(np.round(self.distance_maximal)), int(np.round(self.distance_maximal))), 0,
                    startAngle=self.start_angle, endAngle=self.end_angle, color=color, thickness=thickness)  # Use white color (255) for the mask

        cv2.ellipse(mask, self.center, 
                    (int(np.round(self.distance_minimal)), int(np.round(self.distance_minimal))), 0,
                    startAngle=self.start_angle, endAngle=self.end_angle, color=color, thickness=thickness)

        if thickness > 0:
            # Calculate start and end points on both ellipses
            # Start point is the point corresponding to the start angle, end point is the point corresponding to the end angle
            start_point_outer = (int(self.center[0] + self.distance_maximal * np.cos(np.deg2rad(self.start_angle))),
                                 int(self.center[1] + self.distance_maximal * np.sin(np.deg2rad(self.start_angle))))
            end_point_outer = (int(self.center[0] + self.distance_maximal * np.cos(np.deg2rad(self.end_angle))),
                               int(self.center[1] + self.distance_maximal * np.sin(np.deg2rad(self.end_angle))))
            
            start_point_inner = (int(self.center[0] + self.distance_minimal * np.cos(np.deg2rad(self.start_angle))),
                                 int(self.center[1] + self.distance_minimal * np.sin(np.deg2rad(self.start_angle))))
            end_point_inner = (int(self.center[0] + self.distance_minimal * np.cos(np.deg2rad(self.end_angle))),
                               int(self.center[1] + self.distance_minimal * np.sin(np.deg2rad(self.end_angle))))

            # Draw lines connecting outer and inner ellipse at start and end points
            cv2.line(image, start_point_outer, start_point_inner, color, thickness)
            cv2.line(image, end_point_outer, end_point_inner, color, thickness)

        cv2.circle(image, self.middle, 10, (0, 0, 255), -1)

        new_image = cv2.addWeighted(image, 1, mask, 0.5, 0)
        return new_image


    def get_bottom_point_longitude(self):
        """
        Get the longitude of the bottom point of the annulus.
        """
        right_y = int(self.center[1] + self.distance_maximal * np.sin(np.deg2rad(self.start_angle)))
        left_y = int(self.center[1] + self.distance_maximal * np.sin(np.deg2rad(self.end_angle)))
        return max(right_y, left_y)
    
