import pandas as pd
import numpy as np
import rasterio
from shapely.geometry import Point, MultiPolygon, Polygon
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


class Observatory:
    """This class is used to store the observatory information.
    It will include the geo-location of every point, its elevation, azimuth, height and FOV.
    It will also include the minimal and maximal range that it can see, and the color.
    """

    def __init__(self, latitude, longitude, distance_minimal, distance_maximal, fov_horizontal, fov_vertical, height,
                 color):
        self.latitude = latitude
        self.longitude = longitude
        self.distance_minimal = distance_minimal
        self.distance_maximal = distance_maximal
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
            fov_horizontal = int(data['fov_horizontal'])
            fov_vertical = int(data['fov_vertical'])
            height = int(data['height_value'])
            color = tuple(map(int, data['color_value'][1:-1].split(',')))  # Parses "(255,0,0)" to (255, 0, 0)

            return cls(latitude, longitude, distance_minimal, distance_maximal, fov_horizontal, fov_vertical, height, color)
