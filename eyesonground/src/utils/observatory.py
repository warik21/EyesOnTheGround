import numpy as np
import cv2
from utils.scan import Annulus
import rasterio
from geopy.distance import great_circle
from rasterio.transform import from_origin
from utils.utils import *
import json

class Observatory:
    """This class is used to store the observatory information.
    It will include the geo-location of every point, its elevation, azimuth, height and FOV.
    It will also include the minimal and maximal range that it can see, and the color.
    """

    def __init__(self, latitude, longitude, 
                fov_horizontal=32.0, fov_vertical=18.0, height=15.0, angluar_velocity=0.1,
                tif_image_path=r'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/images_and_reults/eilat_updated.tif',
                x=None, y=None, 
                distance_minimal=None, distance_maximal=None, 
                start_angle=None, looking_angle=None, end_angle=None,
                color=None, thickness=None):
        """
        Initializes the attributes of an observatory object. Optional parameters are set to None by default.

        Args:
            latitude (float): The latitude of the observatory.
            longitude (float): The longitude of the observatory.
            distance_minimal (int, optional): The minimal distance that the observatory can see.
            distance_maximal (int, optional): The maximal distance that the observatory can see.
            start_angle (int, optional): The starting angle of the observatory's field of view.
            end_angle (int, optional): The ending angle of the observatory's field of view.
            fov_horizontal (int, optional): The horizontal field of view of the observatory.
            fov_vertical (int, optional): The vertical field of view of the observatory.
            height (int, optional): The height of the observatory.
            color (tuple, optional): The color of the observatory.
            thickness (int, optional): The thickness of the observatory.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.distance_minimal = distance_minimal
        self.distance_maximal = distance_maximal
        self.start_angle = start_angle
        self.looking_angle = looking_angle
        self.end_angle = end_angle
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.height = height
        self.angluar_velocity = angluar_velocity
        self.tif_image_path = tif_image_path
        self.x = x
        self.y = y
        self.color = color if color is not None else (255, 255, 255)  # Default color set as white
        self.thickness = thickness if thickness is not None else 1   # Default thickness
        self.transform_matrix = None
        self.inverse_transform_matrix = None
        self.fov = None
        self.pixels = None
        self.positions = None

    def coordinates(self):
        """
        This function returns the observatory's coordinates.
        :return: The observatory's coordinates.
        """
        return self.latitude, self.longitude
    
    def get_transform_matrix(self):
        """
        This function returns the transform matrix of the geotiff file.
        :param geotiff_path: The path of the geotiff file.
        :return: The transform matrix of the geotiff file.
        """
        with rasterio.open(self.tif_image_path) as src:
            self.transform_matrix = src.transform
        self.transform_matrix = affine_to_array(self.transform_matrix)
        return self.transform_matrix
    
    def find_inverse_transform(self):
        """
        This function finds the inverse transform matrix of the geotiff file.
        :param transform_matrix: The transform matrix of the geotiff file.
        :return: The inverse transform matrix of the geotiff file.
        """
        if self.transform_matrix is None:
            self.get_transform_matrix()
        self.inverse_transform_matrix = find_inverse_transform(self.transform_matrix)
        return self.inverse_transform_matrix

    def get_fov(self):
        """
        This function return the observatory's FOV.
        :return: The observatory's FOV.
        """
        self.fov = Annulus(self.get_pixel_location(), self.distance_maximal, self.distance_minimal, self.start_angle, self.end_angle)
        return self.fov

    def get_pixel_location(self):
        """
        We use the inverse transform matrix to find the pixel location of the observatory.
        :param geotiff_path: The path of the geotiff file.
        :return: The pixel location of the observatory in the image.
        """
        if self.inverse_transform_matrix is None:
            self.find_inverse_transform()
        #TODO: make sure that the transform matrix and its inverse are defined.
        self.pixels = geo_to_pixel(self.latitude, self.longitude, self.inverse_transform_matrix)  # A function from utils/utils.py
        return self.pixels

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

    def calc_min_distance(self, print_result=True):
        """
        This function calculates the minimal distance that the observatory can see based on
        its height, FOV, and maximal distance. It updates the distance_minimal attribute.
        """
        deg_max = np.arctan(self.distance_maximal / self.height)
        deg_min = deg_max - np.radians(self.fov_vertical)

        # Calculate the new minimal distance
        new_min_distance = self.height * np.tan(deg_min)
        
        # Print the existing and new minimal distance
        if print_result:
            print(f"Existing Min Distance:{self.distance_minimal}, New Min Distance:{new_min_distance}")
        
        # Update the distance_minimal attribute
        self.distance_minimal = new_min_distance

        return new_min_distance

    def calc_max_distance(self):
        """
        This function calculates the maximal distance that the observatory can see based on
        its height, FOV, and minimal distance. It updates the distance_maximal attribute.
        """
        deg_min = np.arctan(self.distance_minimal / self.height)
        deg_max = deg_min + np.radians(self.fov_vertical)

        # Calculate the new maximal distance
        new_max_distance = self.height * np.tan(deg_max)
        # Print the existing maximal distance
        print(f"Existing Max Distance:{self.distance_maximal}, New Max Distance:{new_max_distance}")
        # Update the distance_maximal attribute
        self.distance_maximal = new_max_distance
        
        return new_max_distance

    def calc_fov_vertical(self):
        """
        This function calculates the vertical FOV that the observatory can see based on
    its height, minimal distance, and maximal distance. It updates the fov_vertical attribute.
        """
        deg_min = np.arctan(self.distance_minimal / self.height)
        deg_max = np.arctan(self.distance_maximal / self.height)

        # Calculate the new FOV horizontal
        new_fov_vertical = np.degrees(deg_max - deg_min)

        # Print the existing and new FOV horizontal
        print(f"Existing FOV Vertical:{self.fov_vertical}, New FOV Vertical:{new_fov_vertical}")
        
        # Update the fov_horizontal attribute
        self.fov_horizontal = new_fov_vertical

        return new_fov_vertical
    
    def calc_fov_horizontal(self):
        """
        This function calculates the horizontal FOV that the observatory can see based on
        The minimal and maximal angles."""
        new_fov_horizontal = self.end_angle - self.start_angle
        print(f"Existing FOV Horizontal:{self.fov_horizontal}, New FOV Horizontal:{new_fov_horizontal}")
        self.fov_horizontal = new_fov_horizontal
        return new_fov_horizontal

    @classmethod
    def load_from_file(cls, json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
            return cls(**data)  # Unpack the data as arguments
    
    def find_starting_point(self, extreme_point):
        """
        This function takes the extreme point and finds the maximal_distance by understanding
        the distance from the extreme point to the observatory.
        :param extreme_point: The extreme point of the observatory.
        :param geotiff_path: The path of the geotiff file.
        :return: The maximal distance.
        """
        extreme_point = np.array(extreme_point)
        observatory_point = np.array(self.longitude, self.latitude)
        distance = np.linalg.norm(observatory_point - extreme_point)
        return distance
    
    def draw_mask(self, image_shape, geotiff_path, position=0):
        """
        This function draws the mask of the observatory on the image. It is used foo
        when only the angle is moving, without change of distance or zoom, this method
        will be deprecated in the future, and all drawing will be in the fov/annulus class.
        params:
        geotiff_path(str): The path of the geotiff file.
        image_shape(tuple): The shape of the image.
        position(float): The position of the camera, ranging from 0 to 1.
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
        # If thickness isn't 0, draw the lines which indicate the FOV
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

    def get_next_positions(self, destination_angle, destination_distance):
        """
        This function returns the next position of the observatory.
        :return: The next position of the observatory.
        """
        num_frames = int((destination_angle - self.start_angle) / self.angluar_velocity)
        angles = np.linspace(self.start_angle, destination_angle, num_frames)
        distances = np.linspace(self.distance_maximal, destination_distance, num_frames)
        return list(zip(angles, distances))

    def scan(self):
        """
        This function scans the area.
        """


        #smallest_angle = self.get_smallest_angle()
        #largest_angle = self.get_largest_angle()
        #initial_distance = self.distance_maximal

    def get_smallest_angle(self, scan_area):
        """
        This function returns the smallest angle relevant in the scanning area for the observatory.
        This angle depends directly on the top right corner of the scanning area
        :return: The smallest angle.
        """
        angle = calculate_angle(self.pixels, scan_area.top_right_corner_pixels)
        angle += self.fov_horizontal/2  # We add half of the FOV to get the angle of the middle of the FOV
        return angle

    def get_largest_angle(self, scan_area):
        """
        This function returns the largest angle.
        :return: The largest angle.
        """
        angle = calculate_angle(self.pixels, scan_area.top_left_corner_pixels)
        angle -= self.fov_horizontal/2  # We subtract half of the FOV to get the angle of the middle of the FOV
        return angle

def load_and_prepare_observatory(file_path, tif_image_path=None, transform_matrix=None):
    """
    This function loads an observatory from a json file and calculates its pixel location.
    :param file_path: The path to the json file.
    :param tif_image_path: The path to the tif image.
    :param transform_matrix: The transform matrix of the image, transforming pixels into coordinates.
    :return: The observatory object.
    """
    observatory = Observatory.load_from_file(file_path)
    observatory.tif_image_path = tif_image_path
    observatory.get_pixel_location()
    return observatory