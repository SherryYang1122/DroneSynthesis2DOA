# doa_runner_main.py
# This code provides the main runner for DOA estimation algorithms and evaluation.
# (c) 2025, X. Yang, Fraunhofer IDMT, Germany, MIT License
# version 1.0, August 2025

import numpy as np
import argparse
import warnings
import os
import soundfile as sf
import pandas as pd
import json
from tqdm import tqdm
import librosa
import shutil
import csv
import yaml
import torch

from doa_alg import get_doa_srp_phat
from geometry import cart2sph

# Hyperparameter settings
fib_num = 500  # Number of points in fibonacci sphere for SRP (half sphere)


# localization part
def main():
    mic_jason = "mic_config.json"
    wav_path = args.dataset
    
    algorithm = args.algorithm
    # creat a folder to save the results
    if args.beta:
        algorithm = algorithm + "_beta"
    if args.mask:
        algorithm = algorithm + "_mask"

    output_folder = os.path.join(result_dir, "DOA_" + algorithm)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Open the JSON file and load its contents
    with open(os.path.join(result_dir, mic_jason), 'r') as file:
        mic_data = json.load(file)
    mic_array = mic_data[0]
    mic_pos_matrix = np.array(mic_array["mic_pos"])

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # List to store file names
    # Iterate through all files in the folder
    wav_path = os.path.join(result_dir, wav_path)
    for folder_name in tqdm(os.listdir(wav_path)):
        # Get the full file path
        file_name = folder_name + '.wav'
        file_path = os.path.join(wav_path, folder_name)
        file_path = os.path.join(file_path, file_name)
        if os.path.exists(file_path):
            # Add the file name to the list
            # read the WAV file
            signal, _ = librosa.load(file_path, sr=sampling_rate, mono=False)
            doa_est_cart = get_doa_est(args.algorithm, signal, frame_length, frame_hop,
                                       mic_pos_matrix, sampling_rate, beta=args.beta,
                                       mask=args.mask, device=device)

            # save doa estimated postion
            csv_file = folder_name + ".csv"
            with open(os.path.join(output_folder, csv_file), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["time", "X", "Y", "Z", "ele", "azi"])
                for i, point in enumerate(doa_est_cart):
                    angles = cart2sph(point)
                    writer.writerow([i * frame_hop + frame_length / 2] + list(point) + angles)

# Real-time localization function
def get_doa_est(algorithm, signal, frame_length, frame_hop, mic_pos_matrix, sampling_rate, beta, mask, device):
    if algorithm == 'srp_phat':
        doa_est_cart = get_doa_srp_phat(signal, frame_length, frame_hop, mic_pos_matrix,
                                        sampling_rate, fib_num, speed_of_sound, beta, mask, device)
    return doa_est_cart


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODA algorithms')
    parser.add_argument('--dataset', type=str, default='MicArrayDataTest', help='Path to the wav data for localization')
    parser.add_argument('--exp', type=str, default='exp_test', help='Enter experiment ID')
    # TODA algorithms
    parser.add_argument('--algorithm', choices=['srp_phat'], default='srp_phat', help='Localization algorithm choice')
    parser.add_argument('--beta', action='store_true', help='Only consider adding beta to srp phat')
    parser.add_argument('--mask', action='store_true', help='Only consider adding mask in frequqency domain')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    # Get the current working directory
    abspath = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory path
    parent_dir = os.path.dirname(abspath)
    result_dir = os.path.join(parent_dir, "exps", args.exp)
    os.makedirs(result_dir, exist_ok=True)

    hyperpara_path = os.path.join(result_dir, 'exp_config.yaml')
    with open(hyperpara_path, 'r') as yamlfile:
        hyperparams = yaml.safe_load(yamlfile)
    # Access hyperparameters
    speed_of_sound = hyperparams['speed_of_sound']  # Speed of sound in air (m/s)
    sampling_rate = hyperparams['sampling_rate']
    # real time localizaton: frame parameters
    frame_length = hyperparams['frame_length']
    frame_hop = hyperparams['frame_hop']
    
    main()
