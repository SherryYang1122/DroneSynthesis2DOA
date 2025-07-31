import numpy as np

# Calculate the symmetrical point of a given point about a plane
def sym_point_about_plane(point, plane_coeff):
    # Extract the coefficients of the plane equation, ax + by + cz + d = 0
    a, b, c, d = plane_coeff
    # Calculate the distance from the point to the plane
    distance = (a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(a**2 + b**2 + c**2)
    # Calculate the sym vector
    sym_vector = (np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)) * (2 * distance)
    # Calculate the sym point
    sym_point = point - sym_vector
    return sym_point

def reflection_distance(point1, point1_sym, point2, plane_coeff):
    # Calculate the dot products with plane_coefficients
    dot_product1 = np.dot(point1, plane_coeff[:3]) + plane_coeff[3]
    dot_product2 = np.dot(point2, plane_coeff[:3]) + plane_coeff[3]
    # Check if the points are on opposite sides of the plane
    if dot_product1 * dot_product2 <= 0:
        return 0
    else:
        # Calculate the distance between the symmetrical point and point2
        distance = np.linalg.norm(point1_sym - point2)
        return distance

# Convert  Cartesian coordinates to azimuth and elevation angles   
def cart2sph(vector):
    # Calculate elevation angle, vector = [x,y,z]
    elevation_angle = np.degrees(np.arctan2(np.sqrt(vector[0]**2 + vector[1]**2), vector[2]))

    # Calculate azimuth angle
    azimuth_angle = np.degrees(np.arctan2(vector[1], vector[0]))
    azimuth_angle = (azimuth_angle + 360) if azimuth_angle<0 else azimuth_angle
    return [elevation_angle, azimuth_angle]

# Convert azimuth and elevation angles to Cartesian coordinates
def sph2cart(elevation_angle, azimuth_angle):
    elevation = np.radians(elevation_angle)
    azimuth = np.radians(azimuth_angle)
    x = np.sin(elevation) * np.cos(azimuth)
    y = np.sin(elevation) * np.sin(azimuth)
    z = np.cos(elevation)
    return [x, y, z]

# Create a grid of possible source directions
# Generate evenly distributed points on the spherical surface
# n is the amount of directions    
def half_fibonacci_sphere(n):  
    n *= 2

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:,0] = radius * np.cos(theta)
    points[:,1] = radius * np.sin(theta)
    points[:,2] = z

    points = points[points[:, 2] >= 0]
    return np.transpose(points)

def fibonacci_sphere(n):  
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:,0] = radius * np.cos(theta)
    points[:,1] = radius * np.sin(theta)
    points[:,2] = z
    return np.transpose(points)