import rasterio
import numpy as np
from utils.utils import *
from geopy.distance import great_circle
import numpy as np
from utils.observatory import Observatory
import math
import cv2
from utils.scan import ScanningArea
from utils.utils import *

    
image_path = r'C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/images_and_reults/eilat_updated.tif'
transform_matrix, resolution = get_transform_matrix(image_path)
transform_matrix = affine_to_array(transform_matrix)  # Convert pixels to geo
inv_transform_matrix = inv_transform_matrix = find_inverse_transform(transform_matrix)  # Convert geo to pixels

# Define the location of the observatory
third_point_coords = 29.552339, 34.956392  #top left (lat, lon)
third_point_pixels = 12659, 4263  #bottom left (x, y)
observatory = Observatory(latitude=third_point_coords[0], longitude=third_point_coords[1], height=50)

# Define the scanning area
top_right = 29.548411, 34.974032
bottom_left = 29.540247, 34.952230
scanning_area = ScanningArea(top_right, bottom_left)
scanning_area.get_pixel_corners(inv_transform_matrix)

starting_point = scanning_area.top_right_corner_geo
real_world_distance = great_circle(observatory.coordinates(), scanning_area.top_right_corner_geo).meters
# To find the distance in pixels, we need to use our image's transform matrix
# We'll use the inverse transform matrix to convert from pixels to geo


observatory_pixels = geo_to_pixel(observatory.latitude, observatory.longitude, inv_transform_matrix)
scanning_area_top_right_pixels = scanning_area.top_right_corner_pixels

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

# Draw the green arrow and display the image
arrow_image = draw_direction(image_size, start_point, observatory,length=pixel_distance, thickness=100)
result = cv2.addWeighted(image, 1, arrow_image, 0.5, 0)
# Draw the rectangle of the scanning area in red:
result = scanning_area.draw_rectangle(result, color=(0, 0, 255))

result_ratio = image.shape[0] / image.shape[1]
result_size = 2000
result_resized = cv2.resize(result, (result_size, int(result_size * result_ratio)))
# result_resized2 = cv2.resize(image, (result_size, int(result_size * result_ratio)))
cv2.imshow("Direction", result_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save the image
cv2.imwrite('C:/Users/eriki/OneDrive/Documents/all_folder/other_projects/images_and_reults/arrow.jpg', result_resized)

