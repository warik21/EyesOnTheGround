import pandas as pd
import numpy as np
import rasterio
import cv2
import rasterio
from rasterio.Transform import Affine
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


class Observatory:
    """This class is used to store the observatory information.
    It will include the geo-location of every point, its elevation, azimuth, height and FOV.
    It will also include the minimal and maximal range that it can see, and the color.
    """

    def __init__(self, latitude, longitude, distance_minimal, distance_maximal, start_angle, end_angle, fov_horizontal, 
                 fov_vertical, height, color, thickness):
        """
        Initializes the attributes of an observatory object with the provided values.

        Args:
            latitude (float): The latitude of the observatory.
            longitude (float): The longitude of the observatory.
            distance_minimal (int): The minimal distance that the observatory can see.
            distance_maximal (int): The maximal distance that the observatory can see.
            start_angle (int): The starting angle of the observatory's field of view.
            end_angle (int): The ending angle of the observatory's field of view.
            fov_horizontal (int): The horizontal field of view of the observatory.
            fov_vertical (int): The vertical field of view of the observatory.
            height (int): The height of the observatory.
            color (tuple): The color of the observatory.
            thickness (int): The thickness of the observatory.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.distance_minimal = distance_minimal
        self.distance_maximal = distance_maximal
        self.start_angle = start_angle  # The angle in which the camera starts the scan. Should be smaller than end_angle
        self.end_angle = end_angle  # The angle in which the camera ends the scan. Should be larger than start_angle
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.height = height
        self.color = color
        self.thickness = thickness


    def check_validity(self):
        """
        This function checks whether the observatory's parameters are valid.
        :return: True if the parameters are valid, False otherwise.
        """
        if self.start_angle > self.end_angle:
            return False
        if self.distance_minimal > self.distance_maximal:
            return False
        if self.fov_horizontal > 360 or self.fov_horizontal < 0:
            return False
        if self.fov_vertical > 180 or self.fov_vertical < 0:
            return False
        if self.height < 0:
            return False
        if not np.isclose(self.calc_max_distance(), self.distance_maximal, atol=50):
            return False
        if not np.isclose(self.calc_min_distance(), self.distance_minimal, atol=50):
            return False
        if not np.isclose(self.calc_fov_horizontal(), self.fov_horizontal, atol=5):
            return False
        return True

    def calc_min_distance(self):
        """
        This function calculates the minimal distance that the observatory can see based on
        its height, FOV and maximal distance.
        :return:
        """
        height = self.height
        fov_horizontal = self.fov_horizontal
        max_distance = self.distance_maximal

        deg_max = np.arctan(max_distance / height)

        deg_min = deg_max - fov_horizontal

        min_distance = height * np.tan(deg_min)

        return min_distance

    def calc_max_distance(self):
        """
        This function calculates the maximal distance that the observatory can see based on
        its height, FOV and minimal distance.
        :return:
        """
        height = self.height
        fov_horizontal = self.fov_horizontal
        min_distance = self.distance_minimal

        deg_min = np.arctan(min_distance / height)

        deg_max = deg_min + fov_horizontal

        max_distance = height * np.tan(deg_max)

        return max_distance

    def calc_fov_horizontal(self):
        """
        This function calculates the horizontal FOV that the observatory can see based on
        its height, minimal distance and maximal distance.
        :return:
        """
        height = self.height
        min_distance = self.distance_minimal
        max_distance = self.distance_maximal

        deg_min = np.arctan(min_distance / height)
        deg_max = np.arctan(max_distance / height)

        fov_horizontal = deg_max - deg_min

        return fov_horizontal

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, 'r') as file:
            data = {}
            for line in file:
                key, value = line.strip().split('=')
                data[key.strip()] = value.strip()

            # Parsing the values
            latitude = float(data['latitude'])
            longitude = float(data['longitude'])
            distance_minimal = int(data['distance_minimal'])
            distance_maximal = int(data['distance_maximal'])
            start_angle = int(data['start_angle'])
            end_angle = int(data['end_angle'])
            fov_horizontal = int(data['fov_horizontal'])
            fov_vertical = int(data['fov_vertical'])
            height = int(data['height_value'])
            color = tuple(map(int, data['color_value'][1:-1].split(',')))  # Parses "(255,0,0)" to (255, 0, 0)
            thickness = int(data['thickness'])

            return cls(latitude, longitude, distance_minimal, distance_maximal, start_angle, end_angle, fov_horizontal, 
                       fov_vertical, height, color, thickness)
    
    def get_pixel_location(self, geotiff_path):
        """
    This function gets the pixel location of the observatory in the image.
    :param geotiff_path: The path of the geotiff file.
    :return: The pixel location of the observatory in the image.
    """
        # #TODO: Check whether this actually gets only the metadata and not the whole image
        # with rasterio.open(geotiff_path, 'r') as src:
        #     # Transform the observatory's geographic coordinates to the image's coordinate system
        #     x, y = src.index(self.latitude, self.longitude)
# 
        # if x < 0 or y < 0:
        #     raise ValueError("Pixel location is out of image bounds")
# 
        x, y = int(self.latitude), int(self.longitude)
        return x, y

    def get_pixel_location2(self, image):
        """
        This function gets the pixel location of the observatory in the image.
        :param geotiff_path: The path of the geotiff file.
        :return: The pixel location of the observatory in the image.
        """
        return image.index(self.latitude, self.longitude)

    def draw_mask(self, image_shape, geotiff_path, position):
        """
        This function draws the mask of the observatory on the image.
        params:
        geotiff_path(str): The path of the geotiff file.
        image_shape(tuple): The shape of the image.
        position(float): The position of the camera, rangning from 0 to 1.
        :return: The mask of the observatory.
        """
        mask_maximal: np.ndarray = np.zeros(image_shape, dtype='uint8')
        mask_minimal: np.ndarray = np.zeros(image_shape, dtype='uint8')

        center = self.get_pixel_location(geotiff_path)
        dist_from_start = (self.end_angle - self.start_angle) * position
        top_arc = self.start_angle + dist_from_start
        bottom_arc = top_arc + self.fov_horizontal
        thickness = self.thickness


        cv2.ellipse(mask_maximal, center, (self.distance_maximal, self.distance_maximal), 0,
                    startAngle=top_arc, endAngle=bottom_arc, color=self.color, thickness=thickness)
        if thickness > 0:
            start_point = (int(center[0] + self.distance_maximal * np.cos(np.deg2rad(top_arc))),
                       int(center[1] + self.distance_maximal * np.sin(np.deg2rad(top_arc))))
            end_point = (int(center[0] + self.distance_maximal * np.cos(np.deg2rad(bottom_arc))),
                     int(center[1] + self.distance_maximal * np.sin(np.deg2rad(bottom_arc))))

            # Draw the lines
            cv2.line(mask_maximal, center, start_point, self.color, self.thickness)
            cv2.line(mask_maximal, center, end_point, self.color, self.thickness)
    
        cv2.ellipse(mask_minimal, center, (self.distance_minimal, self.distance_minimal), 0,
                    startAngle=top_arc, endAngle=bottom_arc, color=self.color, thickness=thickness)
        
        mask = cv2.subtract(mask_maximal, mask_minimal)

        return mask


    def reset_coordinates(self, image_shape, known_pixels, known_coords, geotiff_path):
        """
        This function resets the observatory's coordinates to the image's coordinate system.
        :param 
        image_shape: The shape of the image.
        known_pixels: The known pixel coordinates of the observatory.
        known_coords: The known geographic coordinates of the observatory.
        geotiff_path: The path of the geotiff file.
        :return: None
        """
        pixel_coords = 