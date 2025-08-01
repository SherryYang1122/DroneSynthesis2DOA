# doa_alg.py
# This code provides DOA estimation algorithms including GCC-PHAT and SRP-PHAT methods.
# (c) 2025, X. Yang, Fraunhofer IDMT, Germany, MIT License
# version 1.0, August 2025

import numpy as np
import os
from geometry import fibonacci_sphere, half_fibonacci_sphere
from geometry import sph2cart, cart2sph
import torch

# Functions for localization (e.g., GCC-PHAT and SRP-PHAT)
# TDOA (Time Difference of Arrival) is estimated using different methods

# GCC-PHAT
# A realtime DOA (Direction Of Arrival) using 2 microphones, far-field
# Compute time delay using GCC-PHAT
def gcc_phat(signal_1, signal_2, fs, max_tau=None, interp=16):
    # Compute the cross-correlation function
    n = signal_1.shape[0]
    cross_correlation = np.fft.fft(signal_2) * np.conj(np.fft.fft(signal_1))
    # Compute the GCC-PHAT spectrum
    gcc_phat_spectrum = cross_correlation / np.abs(cross_correlation)
    # Find the index of the maximum value
    cc = np.fft.ifft(gcc_phat_spectrum, n=(interp * n))
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    max_index = np.argmax(np.abs(cc)) - max_shift
    # Calculate time delay in seconds
    tau = max_index / float(interp * fs)
    return tau, max_index / interp

def calculate_angle(time_delay, microphone_distance, speed_of_sound):
    # equation: tau = d*cos(theta)/c
    angle_rad = np.arccos(time_delay * speed_of_sound / microphone_distance)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Calculate a direction vector from microphone positions and time differences.
def calculate_direction_vector_from_time_differences(mic_pos_matrix, angle_deg1,
                                                     angle_deg2, angle_deg3,
                                                     size_m):
    v1 = mic_pos_matrix[0] - mic_pos_matrix[1]
    v2 = mic_pos_matrix[0] - mic_pos_matrix[2]
    v3 = mic_pos_matrix[0] - mic_pos_matrix[3]
    A = np.array([v1, v2, v3])
    b = size_m * np.array([np.cos(np.radians(angle_deg1)),
                           np.cos(np.radians(angle_deg2)),
                           np.cos(np.radians(angle_deg3))])
    x = np.linalg.solve(A, b)
    normalized_x = x / np.linalg.norm(x)
    return normalized_x


def get_direction_gcc_phat(signal, window_period, overlap_rate, mic_pos_matrix,
                           sampling_rate, speed_of_sound, size_m, interp=512):
    window_size = int(sampling_rate * window_period)
    overlap = int(window_size * overlap_rate)
    num_windows = int((signal.shape[1] - overlap) / (window_size - overlap))
    angle1_degrees = []
    angle2_degrees = []
    angle3_degrees = []
    tau1_values = []
    tau2_values = []
    tau3_values = []
    md1 = np.linalg.norm(mic_pos_matrix[1] - mic_pos_matrix[0])
    md2 = np.linalg.norm(mic_pos_matrix[2] - mic_pos_matrix[0])
    md3 = np.linalg.norm(mic_pos_matrix[3] - mic_pos_matrix[0])
    for i in range(num_windows):
        start_idx = i * (window_size - overlap)
        end_idx = start_idx + window_size
        window_signal0 = signal[0][start_idx:end_idx]
        window_signal1 = signal[1][start_idx:end_idx]
        window_signal2 = signal[2][start_idx:end_idx]
        window_signal3 = signal[3][start_idx:end_idx]
        # Calculate time delays
        tau1, _ = gcc_phat(window_signal0, window_signal1, sampling_rate,
                           md1 / speed_of_sound, interp)
        tau2, _ = gcc_phat(window_signal0, window_signal2, sampling_rate,
                           md2 / speed_of_sound, interp)
        tau3, _ = gcc_phat(window_signal0, window_signal3, sampling_rate,
                           md3 / speed_of_sound, interp)
        tau1_values.append(tau1)
        tau2_values.append(tau2)
        tau3_values.append(tau3)
        angle1_degrees.append(calculate_angle(tau1, md1, speed_of_sound))
        angle2_degrees.append(calculate_angle(tau2, md2, speed_of_sound))
        angle3_degrees.append(calculate_angle(tau3, md3, speed_of_sound))
    drone_direction_est = [
        calculate_direction_vector_from_time_differences(
            mic_pos_matrix, angle1_degrees[i], angle2_degrees[i],
            angle3_degrees[i], size_m)
        for i in range(len(angle1_degrees))
    ]
    return drone_direction_est


def srp_phat(microphone_positions, microphone_signals, sphere_points,
             sampling_rate, speed_of_sound, ind, beta_flag, mask_flag, device):
    num_microphones = len(microphone_positions)
    num_samples = len(microphone_signals[0])
    # Initialize an accumulator for each grid point
    accumulator = np.zeros(sphere_points.shape[1])
    # accumulator = torch.zeros(sphere_points.shape[1], device=device)

    # Calculate PHAT weights for each microphone pair
    spectrum = np.fft.fft(microphone_signals)
    min_smooth = 1
    # Band-pass
    low_cut = 250
    high_cut = 7000
    freqs = np.fft.fftfreq(len(spectrum[0]), d=1 / sampling_rate)  # Generate frequency bins
    # Create a mask for desired frequency range
    mask = (np.abs(freqs) >= low_cut) & (np.abs(freqs) <= high_cut)
    for i in range(num_microphones):
        for j in range(i + 1, num_microphones):
            cross_correlation = spectrum[j] * np.conj(spectrum[i])
            # cross_correlation = spectrum[j] * torch.conj(spectrum[i])
            # Prevent division by zero and add a smoothing term.
            beta = 0.7  # between 0.65 and 0.7
            smoothing = 1e-6  # -6
            if beta_flag:
                gcc_phat_spectrum = cross_correlation / (np.abs(cross_correlation)**beta + smoothing)
                # gcc_phat_spectrum = cross_correlation / (torch.abs(cross_correlation)**beta + smoothing)
            else:
                gcc_phat_spectrum = cross_correlation / (np.abs(cross_correlation) + smoothing)
                # gcc_phat_spectrum = cross_correlation / (torch.abs(cross_correlation) + smoothing)
            # if min_smooth > np.min(np.abs(cross_correlation)):
            #     min_smooth = np.min(np.abs(cross_correlation))
            if mask_flag:
                # Apply the mask: retain only desired frequencies
                gcc_phat_spectrum = gcc_phat_spectrum * mask
            cc = np.fft.ifft(gcc_phat_spectrum)
            # cc = torch.fft.ifft(gcc_phat_spectrum)
            microphone_pair = microphone_positions[i] - microphone_positions[j]
            # the grid-search step
            for k in range(sphere_points.shape[1]):
                delay = np.dot(sphere_points[:, k], microphone_pair) / speed_of_sound
                steered_response = cc[int(delay * sampling_rate)
                                      if int(delay * sampling_rate) >= 0
                                      else int(delay * sampling_rate + num_samples)]
                accumulator[k] += np.abs(steered_response)
                # accumulator[k] += torch.abs(steered_response)
    # Find the direction with the maximum power
    estimated_direction_index = np.argmax(accumulator)
    # estimated_direction_index = torch.argmax(accumulator)
    # plot_srp_points(accumulator, sphere_points, ind)
    return sphere_points[:, estimated_direction_index]

import matplotlib.pyplot as plt
def plot_srp_points(accumulator, sphere_points, ind):
    elevation_angles = []
    azimuth_angles = []
    for k in range(sphere_points.shape[1]):
        elevation_angles.append(cart2sph(sphere_points[:, k])[0])
        azimuth_angles.append(cart2sph(sphere_points[:, k])[1])
    # max ind point
    max_idx = np.argmax(accumulator)
    max_value = accumulator[max_idx]
    max_azimuth = azimuth_angles[max_idx]
    max_elevation = elevation_angles[max_idx]
    plt.scatter(azimuth_angles, elevation_angles, c=accumulator, cmap='hot', s=50)
    plt.scatter(max_azimuth, max_elevation, color='blue', marker='x', s=50,
                label=f'Max Value: {max_value}\nAzimuth: {max_azimuth}\nElevation: {max_elevation}')
    plt.legend()
    plt.colorbar(label='Intensity')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Elevation Angle (degrees)')
    plt.title('Intensity Heatmap')
    plt.grid(True)
 
    plt.savefig(os.path.join("example", f"srp_points_{ind}.png"))
    plt.close()
    plt.scatter(range(len(accumulator)), accumulator, s=5)
    top3_indices = sorted(range(len(accumulator)), key=lambda i: accumulator[i], reverse=True)[:3]
    title = ""
    for i in top3_indices:
        plt.scatter(i, accumulator[i], color='red', s=6)
        plt.text(i, accumulator[i], f'{i}+{accumulator[i]:.2f}', ha='center', va='bottom')
        title = title + f'{i}, ele {elevation_angles[i]:.2f}, azi {azimuth_angles[i]:.2f}; '
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.savefig(os.path.join("example", f"srp_value_{ind}.png"))
    plt.close()

    
def get_doa_srp_phat(signal, frame_length, frame_hop, mic_pos_matrix,
                     sampling_rate, fib_num, speed_of_sound, beta, mask, device):
    # Real-time localization
    window_size = int(sampling_rate * frame_length)
    hop_size = int(sampling_rate * frame_hop)
    frame_num = (signal.shape[1] - window_size) // hop_size + 1
    # sphere_points = fibonacci_sphere(fib_num)
    sphere_points = half_fibonacci_sphere(fib_num)
    doa_est = []

    for i in range(frame_num):
        start_idx = i * hop_size
        end_idx = start_idx + window_size
        window_signals = signal[:, start_idx:end_idx]
        # srp_phat -- cart
        doa_est.append(srp_phat(mic_pos_matrix, window_signals, sphere_points,
                                sampling_rate, speed_of_sound, i, beta, mask, device))
    return doa_est


