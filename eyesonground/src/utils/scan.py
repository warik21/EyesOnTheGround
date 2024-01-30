from geopy.point import Point
import math
from utils.utils import geo_to_pixel
import cv2

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

    # TODO : This shouldn't be a class method
    def draw_rectangle(self, image, color=(255, 0, 0), thickness=50):
        """
        Draw a rectangle on an image.

        :param image: The image to draw on.
        :return: The image with the rectangle.
        """
        # Draw the rectangle
        cv2.rectangle(image, self.top_left_corner_pixels, self.bottom_right_corner_pixels, color, thickness)

        return image
