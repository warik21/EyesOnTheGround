import rasterio
import numpy as np
from affine import Affine
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint

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


    
    
    