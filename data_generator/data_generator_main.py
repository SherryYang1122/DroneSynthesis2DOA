# data_generator_main.py
# This code generates synthetic drone audio data with spatial information 
# for training DOA estimation models.
# (c) 2025, X. Yang, Fraunhofer IDMT, Germany, MIT License
# version 1.0, August 2025

import numpy as np
import argparse
#import matplotlib.pyplot as plt
import warnings
import os
import soundfile as sf
import pandas as pd
import json
from tqdm import tqdm
import librosa
import random
import shutil
import csv
import yaml
import datetime
import torch
import scipy

from drone_moving_simulation import get_drone_position, receive_reflected_signal, get_radiation_signal, get_doa_real
from geometry import sym_point_about_plane, cart2sph


# Train and Test data generator：
# Mixing of  Signal generation of drone + environmental noise + ground/wall settings + microphone array
def main():
    if args.example:
        # get an example for reference
        # get_example(), ......(this part remains)
        return None
    else:
        # create a folder to save the results
        output_folder = os.path.join(result_dir, args.output)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # get basic parameters 
        drone_data_folder = args.drone_data
        env_folder = args.env_path
        mic_jason = "mic_config.json"

        # get all subdirectories of drone wavs
        drone_directories = [f.path for f in os.scandir(os.path.join(result_dir, drone_data_folder)) if f.is_dir()]

        # get all microphone arrays
        # Open the JSON file and load its contents
        with open(os.path.join(result_dir, mic_jason), 'r') as file:
            mic_data = json.load(file)
        # choose only one for generation
        mic_array = mic_data[0] 
        mic_pos_matrix = np.array(mic_array["mic_pos"])
        mic_center = mic_array["mic_center"]
        mic_white_noise = mic_array["mic_white_noise"]
        
        NUM_SAMPLE = len(drone_directories)

        # load background noise
        if env_folder != '':
            env_path = os.path.join(abspath, env_folder)
            env_wav_files = [f for f in os.listdir(env_path) if f.endswith('.wav')]
        else:
            env_wav_files = []

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"Using device: {device}")        

        # generate dataset 
        with tqdm(total=NUM_SAMPLE, desc="Processing", unit="iteration") as pbar:
            iterations = 0
            while iterations < NUM_SAMPLE:
                # choose one drone wav file randomly
                #random_drone_dir = random.choice(drone_directories)
                random_drone_dir = drone_directories[iterations]

                # read drone metadata
                with open(os.path.join(random_drone_dir, "metadata.json"), 'r') as file:
                    drone_data = json.load(file)
                random_drone = drone_data[0]
                state_num = len(random_drone["flight_path"])
                audio_sample_path = os.path.join(random_drone_dir, random_drone['drone_source'])
                drone_signal, _ = librosa.load(audio_sample_path, sr=sampling_rate)

                # get fight path
                flight_path = random_drone['flight_path']
                flight_path = np.array(flight_path)
                drone_position = get_drone_position(flight_path, sampling_rate)
                flag = False
                if args.wall:
                    # Check if the drone is within the valid area
                    for i in range(state_num - 1):
                        dot_product1 = np.dot(drone_position[i], wall_coeff[:3]) + wall_coeff[3]
                        dot_product2 = np.dot(drone_position[i + 1], wall_coeff[:3]) + wall_coeff[3]
                        if dot_product1 * dot_product2 < 0:
                            flag = True
                            break
                else:
                    wall_ref_coeff = 0
                if flag:
                    continue 
                ground_ref_coeff = random.choice(ground_ref_coeffs) 
                duration = flight_path[-1][-1]
                signal = get_signal(drone_signal, drone_position, duration, mic_pos_matrix, mic_white_noise, wall_coeff, wall_ref_coeff, ground_ref_coeff)
   
                if env_folder != '':
                    selected_file = random.choice(env_wav_files)
                    env_wav_path = os.path.join(env_path, selected_file)
                    env_noise, _ = librosa.load(env_wav_path, sr=sampling_rate, mono=False)
                    env_noise = np.array(env_noise)
                    max_start = env_noise.shape[1] - signal.shape[1]
                    SNR = random.choice(SNR_values)
                    lowcut = 200
                    highcut = 4000
                    order = 6
                    if max_start > 0:
                        start_point = int(np.random.uniform(0, max_start))
                        end_point = start_point + signal.shape[1]
                        snr_temp = calculate_snr(drone_signal, env_noise[0, start_point:start_point+len(drone_signal)])
                        adjusted_noise = adjust_noise_for_snr(env_noise[:, start_point:end_point], snr_temp, SNR)
                    else:
                        n_repeats = int(np.ceil(signal.shape[1] / env_noise.shape[1]))
                        repeated_env_noise = np.tile(env_noise, (1,n_repeats))
                        max_start = repeated_env_noise.shape[1] - signal.shape[1]
                        start_point = int(np.random.uniform(0, max_start))
                        end_point = start_point + signal.shape[1]
                        snr_temp = calculate_snr(drone_signal, repeated_env_noise[0, start_point:start_point+len(drone_signal)])
                        adjusted_noise = adjust_noise_for_snr(repeated_env_noise[:, start_point:end_point], snr_temp, SNR)
                
                    real_snr = calculate_snr(signal, adjusted_noise)
                    real_bandpass_snr = calculate_bandpass_snr(signal, adjusted_noise, sampling_rate, lowcut, highcut, order)
                    
                    for i in range(len(mic_pos_matrix)):
                        signal[i] = signal[i] + adjusted_noise[i]
                else:
                    real_snr = None
                    SNR = None
                    real_bandpass_snr = None
                
                # save data
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d%H%M")
                folder_name = f"sample{timestamp}_{iterations}"
                output_wav_folder = os.path.join(output_folder, folder_name)
                os.makedirs(output_wav_folder, exist_ok=True)
                # Save audio signals to a wav file
                filename = f"{folder_name}.wav"
                sf.write(os.path.join(output_wav_folder, filename), signal.T, sampling_rate)
                # save doa real postion 
                doa_cart = get_doa_real(drone_position, mic_center, resolution, sampling_rate)
                csv_file = "DOATruePos.csv"
                with open(os.path.join(output_wav_folder, csv_file), 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["time","X", "Y", "Z", "ele", "azi"])  
                    for i, point in enumerate(doa_cart):
                        angles = cart2sph(point)
                        writer.writerow([i*resolution]+list(point)+angles)
                # write metadata 
                random_drone["filename"] = filename
                random_drone["mic_center"] = mic_center
                random_drone["SNR"] = SNR
                random_drone["real_SNR"] = real_snr
                random_drone["real_bandpass_SNR"] = real_bandpass_snr
                
                # save drone metadata
                jason_name = 'metadata.json'
                with open(os.path.join(output_wav_folder, jason_name), "w") as json_file:
                    json.dump([random_drone], json_file, indent=4)
 
                iterations += 1
                pbar.update(1)
              

# get signal from microphone array
def get_signal(drone_signal, drone_position, duration, mic_pos_matrix, mic_white_noise, wall_coeff, wall_ref_coeff, ground_ref_coeff):        
    # Define the environment with the coefficients of the planes, ax+by+cz+d = 0
    ground_coeff = [0, 0, 1, 0]
    mic2wall_sym = [sym_point_about_plane(mic_pos, wall_coeff) for mic_pos in mic_pos_matrix]
    mic2ground_sym = [sym_point_about_plane(mic_pos, ground_coeff) for mic_pos in mic_pos_matrix]
    t = np.linspace(0, duration, int(sampling_rate * duration))
    if len(t) > len(drone_signal) or len(t) > drone_position.shape[0]:
        t = t[:min(len(drone_signal), drone_position.shape[0])] 
    # simulate the signals that the microphones receive from the drone
    mic_noise = [np.random.normal(0, mic_white_noise, len(t)) for i in range(mic_pos_matrix.shape[0])]  # Add white noise to the microphone signal
    signal = [[] for i in range(len(mic_pos_matrix))]
    drone_signal_rad = []
    # vertical radiation directivity model
    update_period = 0.01 #second, every one period, the elevation angle of the microphone to the drone is updated
    for mic_pos in mic_pos_matrix:
        drone_signal_rad.append(get_radiation_signal(drone_signal, drone_position, mic_pos, sampling_rate, update_period))
        #drone_signal_rad.append(drone_signal)

    for j, mic_pos in enumerate(mic_pos_matrix):
        for i, real_time in enumerate(t):
            drone_pos = drone_position[i]
            received_signal = receive_reflected_signal(real_time, drone_pos, mic_pos, wall_coeff, ground_coeff, mic2wall_sym[j], mic2ground_sym[j], speed_of_sound, drone_signal_rad[j], sampling_rate, ground_ref_coeff, wall_ref_coeff) + mic_noise[j][i]
            signal[j].append(received_signal)
    signal = np.array(signal)
    return signal

def butter_bandpass_filter(data, fs, lowcut, highcut, order=6):
    """ Apply a 6th-order Butterworth band-pass filter to the signal """
    nyquist = 0.5 * fs  
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')  
    return scipy.signal.filtfilt(b, a, data)  

def calculate_bandpass_snr(source, noise, fs, lowcut=200, highcut=4000, order=6):
    # Ensure the source and noise arrays have the same length
    if len(source) != len(noise):
        raise ValueError("Source and noise must have the same length.")
    """ Compute the SNR in a specific frequency band using a Butterworth filter """
    if source.ndim == 1: 
        source_filt = butter_bandpass_filter(source, fs, lowcut, highcut, order)
    else:
        source_filt = np.array([butter_bandpass_filter(ch, fs, lowcut, highcut, order) for ch in source])
    if noise.ndim == 1:
        noise_filt = butter_bandpass_filter(noise, fs, lowcut, highcut, order)
    else:
        noise_filt = np.array([butter_bandpass_filter(ch, fs, lowcut, highcut, order) for ch in noise])

    power_signal = np.mean(source_filt ** 2)
    power_noise = np.mean(noise_filt ** 2)

    snr = 10 * np.log10(power_signal / power_noise)
    return snr

def adjust_noise_for_snr(noise, snr_temp, target_snr):
    scaling_factor = 10 ** ((snr_temp - target_snr) / 20)
    adjusted_noise = noise * scaling_factor
    return adjusted_noise


def calculate_snr(source, noise):
    """
    Calculate the Signal-to-Noise Ratio (SNR)
    
    Parameters:
    - source: Pure signal (numpy array)
    - noise: Noise signal (numpy array)
    
    Returns:
    - SNR: Signal-to-Noise Ratio in decibels (dB)
    """
    # Ensure the source and noise arrays have the same length
    if len(source) != len(noise):
        raise ValueError("Source and noise must have the same length.")
    
    # Calculate the signal power as the mean squared value of the source
    signal_power = np.mean(source**2)
    
    # Calculate the noise power as the mean squared value of the noise
    noise_power = np.mean(noise**2)
    
    # Compute the SNR using the formula (in dB)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Test data generator')
    # parameters
    parser.add_argument('--exp', type=str, default='exp_test', help='Enter experiment ID')  
    parser.add_argument('--drone_data', type=str, default='DroneAudioData', help='Path to the drone data file')    
    parser.add_argument('--env_path', type=str, default='', help='Path to the background noise folder')    
    parser.add_argument('--output', type=str, default='MicArrayData', help='Simluated mic data are solved in this folder') 
    # Add the --wall argument, set to True if specified by the user, otherwise False (default)
    parser.add_argument('--wall', action='store_true', help='Whether there is wall reflection')
    
    # generate one example of drone flight and microphone array, results saved in "example" folder
    parser.add_argument('--example', action='store_true', help='Enable give an example of drone flight')
    
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    # Get the current working directory
    abspath = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory path
    parent_dir = os.path.dirname(abspath)
    result_dir = os.path.join(parent_dir, "exps", args.exp)
    os.makedirs(result_dir, exist_ok=True)

    if args.example is False:
        # Load hyperparameters from yaml
        hyperpara_path = os.path.join(result_dir, 'exp_config.yaml')
        if os.path.exists(hyperpara_path):
            with open(hyperpara_path, 'r') as yamlfile:
                hyperparams = yaml.safe_load(yamlfile)
        else:
            default_hyperpara_path = os.path.join(abspath, 'exp_config.yaml')
            shutil.copyfile(default_hyperpara_path, hyperpara_path)
            with open(hyperpara_path, 'r') as yamlfile:
                hyperparams = yaml.safe_load(yamlfile)            
        # Access hyperparameters
        speed_of_sound = hyperparams['speed_of_sound'] # Speed of sound in air (m/s)
        sampling_rate = hyperparams['sampling_rate']
        wall_coeff = hyperparams['wall_coeff']
        wall_ref_coeff = hyperparams['wall_ref_coeff']
        ground_ref_coeffs = hyperparams['ground_ref_coeffs']
        resolution = hyperparams['resolution']
        SNR_values = hyperparams['SNR']

    main()
