import numpy as np

# mic array -- Tetraheadon
# The starting point with all rotX,Y,Z = 0 is a pyramide with one floor edge alongside the x axis of the coordinate system. 
# The length of one edge would be size_m in meter. Usually your array would be very small, eg, size_m = 0.07;
# put this array in certain height (Add at the end to the z-axis).

def get_mic_array_tetra(size_m, microphone_center, rotX, rotY, rotZ):
    # Define the vertices of a unit tetrahedron
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]
    ])
    # Scale the vertices based on the size_m
    vertices *= size_m
    # Rotate the vertices around X, Y, and Z axes
    rotation_matrix_X = np.array([
        [1, 0, 0],
        [0, np.cos(rotX), -np.sin(rotX)],
        [0, np.sin(rotX), np.cos(rotX)]
    ])
    
    rotation_matrix_Y = np.array([
        [np.cos(rotY), 0, np.sin(rotY)],
        [0, 1, 0],
        [-np.sin(rotY), 0, np.cos(rotY)]
    ])
    
    rotation_matrix_Z = np.array([
        [np.cos(rotZ), -np.sin(rotZ), 0],
        [np.sin(rotZ), np.cos(rotZ), 0],
        [0, 0, 1]
    ])
    
    rotated_vertices = vertices.dot(rotation_matrix_X.T).dot(rotation_matrix_Y.T).dot(rotation_matrix_Z.T)
    
    # Translate the vertices to the center
    if microphone_center!=None:
        mic_pos_shift = np.array(microphone_center) - np.mean(rotated_vertices, axis=0)
        translated_vertices = rotated_vertices + mic_pos_shift
    else:
        translated_vertices = rotated_vertices 
    return translated_vertices

# mic array -- octahedron
def get_mic_array_octahedron(microphone_center, rotX, rotY, rotZ):
    vertices = np.array([
        [0, 0, 0],
        [0, 0.09, 0.15],
        [0.15, 0.01875, 0.1145],
        [0.13125, 0.075, 0.009],
        [0.01875, 0.15, 0.15],
        [0.05625, 0.15, 0.009],
        [0.05625, 0.0375, 0.15],
        [0.15, 0.15, 0.075]
    ])

    # Rotate the vertices around X, Y, and Z axes
    rotation_matrix_X = np.array([
        [1, 0, 0],
        [0, np.cos(rotX), -np.sin(rotX)],
        [0, np.sin(rotX), np.cos(rotX)]
    ])
    
    rotation_matrix_Y = np.array([
        [np.cos(rotY), 0, np.sin(rotY)],
        [0, 1, 0],
        [-np.sin(rotY), 0, np.cos(rotY)]
    ])
    
    rotation_matrix_Z = np.array([
        [np.cos(rotZ), -np.sin(rotZ), 0],
        [np.sin(rotZ), np.cos(rotZ), 0],
        [0, 0, 1]
    ])

    rotated_vertices = vertices.dot(rotation_matrix_X.T).dot(rotation_matrix_Y.T).dot(rotation_matrix_Z.T)
    
    # Translate the vertices to the center
    if microphone_center!=None:
        mic_pos_shift = np.array(microphone_center) - np.mean(rotated_vertices, axis=0)
        translated_vertices = rotated_vertices + mic_pos_shift
    else:
        translated_vertices = rotated_vertices 
    return translated_vertices

# mic array -- individual

def get_mic_array_ind(mic_pos, microphone_center, rotX, rotY, rotZ):
    vertices = np.array(mic_pos)

    # Rotate the vertices around X, Y, and Z axes
    rotation_matrix_X = np.array([
        [1, 0, 0],
        [0, np.cos(rotX), -np.sin(rotX)],
        [0, np.sin(rotX), np.cos(rotX)]
    ])
    
    rotation_matrix_Y = np.array([
        [np.cos(rotY), 0, np.sin(rotY)],
        [0, 1, 0],
        [-np.sin(rotY), 0, np.cos(rotY)]
    ])
    
    rotation_matrix_Z = np.array([
        [np.cos(rotZ), -np.sin(rotZ), 0],
        [np.sin(rotZ), np.cos(rotZ), 0],
        [0, 0, 1]
    ])

    rotated_vertices = vertices.dot(rotation_matrix_X.T).dot(rotation_matrix_Y.T).dot(rotation_matrix_Z.T)
    
    # Translate the vertices to the center
    if microphone_center!=None:
        mic_pos_shift = np.array(microphone_center) - np.mean(rotated_vertices, axis=0)
        translated_vertices = rotated_vertices + mic_pos_shift
    else:
        translated_vertices = rotated_vertices 
    return translated_vertices