from geopy.point import Point
import math
from utils.utils import geo_to_pixel
import cv2
import numpy as np

class ScanningArea:
    def __init__(self, corner1, corner2):
        """
        Initializes a ScanningArea object with a rectangular shape based on two diagonal corners.

        :param corner1: Tuple of (latitude, longitude) for the first corner.
        :param corner2: Tuple of (latitude, longitude) for the opposite corner.
        """
        self.latitude_bounds = (min(corner1[0], corner2[0]), max(corner1[0], corner2[0])) # min is west, max is east
        self.longitude_bounds = (min(corner1[1], corner2[1]), max(corner1[1], corner2[1])) # min is south, max is north
        self.top_left_corner = None

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
        :param transform_matrix: The inverse tramsform matrix of the image, transforming coordinates into pixels.
        """
        if self.top_left_corner is None:
            self.get_corners()
        self.top_left_corner_pixels = geo_to_pixel(self.top_left_corner_geo[0], self.top_left_corner_geo[1], transform_matrix)
        self.bottom_right_corner_pixels = geo_to_pixel(self.bottom_right_corner_geo[0], self.bottom_right_corner_geo[1], transform_matrix)
        self.top_right_corner_pixels = geo_to_pixel(self.top_right_corner_geo[0], self.top_right_corner_geo[1], transform_matrix)
        self.bottom_left_corner_pixels = geo_to_pixel(self.bottom_left_corner_geo[0], self.bottom_left_corner_geo[1], transform_matrix)

    def draw(self, image, color=(255, 0, 0), thickness=10):
        """
        Draw a rectangle on an image.

        :param image: The image to draw on.
        :return: The image with the rectangle.
        """
        # Draw the rectangle
        cv2.rectangle(image, self.top_left_corner_pixels, self.bottom_right_corner_pixels, color, thickness)

        return image


class Annulus:
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

    def find_middle(self):
        """
        Find the middle point of the annulus.
        """
        dist = (self.distance_maximal + self.distance_minimal) / 2
        angle = (self.start_angle + self.end_angle) / 2
        # add the distance in the direction of the angle
        x = self.center[0] + dist * math.cos(math.radians(angle))
        y = self.center[1] + dist * math.sin(math.radians(angle))
        self.middle = (x, y) 
        return self.middle

    def draw(self, image, color=(255, 0, 0), thickness=10):
        """
        Draw the annulus on an image.

        :param image: The image to draw on.
        :return: The image with the annulus.
        """
        # Draw the ellipses:
        cv2.ellipse(image, self.center, (self.distance_maximal, self.distance_maximal), 0,
                    startAngle=self.start_angle, endAngle=self.end_angle, color=color, thickness=thickness)

        cv2.ellipse(image, self.center, (self.distance_minimal, self.distance_minimal), 0,
                    startAngle=self.start_angle, endAngle=self.end_angle, color=color, thickness=thickness)

        return image