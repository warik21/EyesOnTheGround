import pandas as pd
import numpy as np
import rasterio
import cv2
from shapely.geometry import Point, MultiPolygon, Polygon
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


class Observatory:
    """This class is used to store the observatory information.
    It will include the geo-location of every point, its elevation, azimuth, height and FOV.
    It will also include the minimal and maximal range that it can see, and the color.
    """

    def __init__(self, latitude, longitude, distance_minimal, distance_maximal, start_angle, fov_horizontal, 
                 fov_vertical, height, color):
        self.latitude = latitude
        self.longitude = longitude
        self.distance_minimal = distance_minimal
        self.distance_maximal = distance_maximal
        self.start_angle = start_angle
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.height = height
        self.color = color

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
            fov_horizontal = int(data['fov_horizontal'])
            fov_vertical = int(data['fov_vertical'])
            height = int(data['height_value'])
            color = tuple(map(int, data['color_value'][1:-1].split(',')))  # Parses "(255,0,0)" to (255, 0, 0)

            return cls(latitude, longitude, distance_minimal, distance_maximal, start_angle, fov_horizontal, 
                       fov_vertical, height, color)
    
    def get_pixel_location(self, geotiff_path):
        """
    This function gets the pixel location of the observatory in the image.
    :param geotiff_path: The path of the geotiff file.
    :return: The pixel location of the observatory in the image.
    """
        #TODO: Check whether this actually gets only the metadata and not the whole image
        with rasterio.open(geotiff_path, 'r') as src:
            # Transform the observatory's geographic coordinates to the image's coordinate system
            x, y = src.index(self.latitude, self.longitude)


        return x, y

    def draw_mask(self, image_shape, geotiff_path):
        """
        This function draws the mask of the observatory on the image.
        :param geotiff_path: The path of the geotiff file.
        :return: The mask of the observatory.
        """
        mask_maximal: np.ndarray = np.zeros(image_shape)
        mask_minimal: np.ndarray = np.zeros(image_shape)

        center = self.get_pixel_location(geotiff_path)
        start_angle = self.start_angle
        end_angle = start_angle + self.fov_horizontal
        thickness = -1


        cv2.ellipse(mask_maximal, center, (self.distance_maximal, self.distance_maximal), 0,
                    startAngle=start_angle, endAngle=end_angle, color=self.color, thickness=thickness)
        cv2.ellipse(mask_minimal, center, (self.distance_minimal, self.distance_minimal), 0,
                    startAngle=start_angle, endAngle=end_angle, color=self.color, thickness=thickness)
        
        mask = cv2.subtract(mask_maximal, mask_minimal)
        mask = mask.astype('uint8')


        return mask