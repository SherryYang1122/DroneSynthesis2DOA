# mic_generator_main.py
# This code generates multiple microphone arrays and their coordinates based 
# on configuration parameters.
# (c) 2025, X. Yang, Fraunhofer IDMT, Germany, MIT License
# version 1.0, August 2025

import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from get_microphone_array import get_mic_array_tetra, get_mic_array_octahedron, get_mic_array_ind


# Purpose: Generate multiple microphone arrays and their coordinates
def main():
    # Get the current working directory
    abspath = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory path
    parent_dir = os.path.dirname(abspath)
    result_dir = os.path.join(parent_dir, "exps", args.exp)
    os.makedirs(result_dir, exist_ok=True)

    # read YAML file
    yaml_file = args.yaml_file
    with open(os.path.join(result_dir, yaml_file), "r") as file:
        config_data = yaml.safe_load(file)

    arrays_data = []
    # Iterate through each array's parameters
    for i, array in enumerate(config_data["arrays"]):
        microphone_type = array["microphone_type"]
        microphone_center = array["microphone_center"]
        rotXYZ = array["rotation_angles_xyz"]
        mic_white_noise = 10 ** (array["mic_white_noise(dB)"] / 10.0)

        # Process each array's parameters
        # rotating the given 3D object's vertex coordinates around the corresponding axis
        # RotX, RotY, and RotZ perform rotation operations around the X, Y, and Z axes (in degrees), respectively
        # If the value is positive, the rotation direction follows the right-hand rule; if negative, it's the opposite.
        if rotXYZ is None:
            rotX = 0
            rotY = 0
            rotZ = 0
        else:
            rotX = np.radians(rotXYZ[0])
            rotY = np.radians(rotXYZ[1])
            rotZ = np.radians(rotXYZ[2])

        if microphone_type == 'tetra':
            array_size = array["array_size"]
            mic_pos_matrix = get_mic_array_tetra(array_size, microphone_center, rotX, rotY, rotZ)
        elif microphone_type == 'octahedron':
            mic_pos_matrix = get_mic_array_octahedron(microphone_center, rotX, rotY, rotZ)
        elif microphone_type == 'individual':
            mic_pos = array["microphone_positions"]
            mic_pos_matrix = get_mic_array_ind(mic_pos, microphone_center, rotX, rotY, rotZ)
        array_data = {
            "name": 'MicArray' + str(i),
            "type": microphone_type,
            "mic_center": microphone_center,
            "mic_pos": mic_pos_matrix.tolist(),
            "mic_white_noise": mic_white_noise
        }
        arrays_data.append(array_data)
    # save the data
    mic_dir = result_dir
    os.makedirs(mic_dir, exist_ok=True)

    json_name = 'mic_config.json'
    with open(os.path.join(mic_dir, json_name), "w") as json_file:
        json.dump(arrays_data, json_file, indent=4)

def plot_arrays(arrays_data, yaml_file, mic_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for array in arrays_data:
        center = array["mic_center"]
        ax.scatter(center[0], center[1], center[2], label=array["name"] + "_" + array["type"])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Microphone Arrays')
    ax.legend()
    plt.savefig(os.path.join(mic_dir, yaml_file.split('.')[0] + '.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process microphone array parameters from YAML file.')
    parser.add_argument('--yaml_file', type=str, default='mic_config.yaml',
                        help='YAML file containing microphone array parameters')
    parser.add_argument('--exp', type=str, default='exp_test', help='Enter experiment ID')
    args = parser.parse_args()
    main()
