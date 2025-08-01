# drone_generator_main.py
# This code represents a drone signal generator that can generate arbitrary numbers of drone flight paths and emitted signals.
# (c) 2025, X. Yang, Fraunhofer IDMT, Germany, MIT License
# version 1.0, August 2025,

import numpy as np
import argparse
import warnings
import os
import random
from tqdm import tqdm
import soundfile as sf
import yaml
import shutil
import json
import datetime

from visualization import plot_drone_scenario, plot_signal_spectrogram, plot_drone_frequency
from drone_signal_generation import getFlightPath, flightstate2RPM, getFullDroneSignal, get_micro_modulation
from geometry import sph2cart


# Main program to simulate signal the drone emits when it flies
def main():
    if args.example:
        # get an example for reference
        get_example()
    else:
        # Create a directory to store dataset of drone audio files if it doesn't exist
        output_folder = os.path.join(result_dir, args.drone_data)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # get basic paramters for drone signal generation
        NUM_SAMPLE = args.num

        # Generate drone data
        with tqdm(total=NUM_SAMPLE, desc="Processing", unit="iteration") as pbar:
            iterations = 0
            while iterations < NUM_SAMPLE:
                # Generate random parameters
                drone_type = random.choice(drone_types) 
                drone_white_noise = random.uniform(drone_white_nois_min, drone_white_noise_max)
                signaltype = random.choice(signaltypes)
                pulse_len = random.choice([i for i in range(pulse_len_min, pulse_len_max + 1)])
                change_time = random.uniform(change_time_min, change_time_max)
                s_shape_coef = random.uniform(s_shape_coef_min, s_shape_coef_max)
                drone_loudness = np.sqrt(loudness ** 2 / pulse_len)
                # Generate random values from the normal distribution for drone initial position
                x_start, y_start, z_start = np.random.normal(x_init, x_init_std), np.random.normal(y_init, y_init_std), np.random.normal(z_init, z_init_std)
                
                starting_point = [x_start, y_start, z_start, 0.0]
                flight_state = []
                forward_dir = []
                time = 0
                # generate random states
                for i in range(state_num):
                    state = random.choices(drone_states, weights=drone_states_weights)
                    if state[0] == 'hover':
                        state_time = round(random.uniform(state_dur_hover_min, state_dur_hover_max), decimal_places)
                    elif state[0] == 'forward':
                        state_time = round(random.uniform(state_dur_forward_min, state_dur_forward_max), decimal_places)
                    elif state[0] == 'climb' or state[0] == 'sink':
                        state_time = round(random.uniform(state_dur_updown_min, state_dur_updown_max), decimal_places)
                    time += state_time
                    state.append(time)
                    flight_state.append(state)
                    # add forward direction
                    if state[0] == 'forward':
                        azimuth = random.uniform(azimuth_min, azimuth_max)
                        elevation = 90
                        forward_dir.append(sph2cart(elevation, azimuth))
                # weather condition    
                wind_speed = round(random.uniform(wind_speed_min, wind_speed_max), decimal_places)
                down_wind = random.choice([True, False])
 
                # generate drone audio
                drone_signal, flight_path, state_speed, rpm_freqs = get_drone_signal(args.simple_noise, starting_point, flight_state, forward_dir, drone_type, wind_speed, down_wind, drone_loudness, drone_white_noise, signaltype, pulse_len, change_time, s_shape_coef, climb_speed_max, sink_speed_max, speed_min)
                
                # check if the drone always flies in this area 
                flag = False
                smoothing = 1e-6
                for state in flight_path:
                    x_pos = state[0]
                    y_pos = state[1]
                    z_pos = state[2]
                    if x_pos < x_min or x_pos > x_max or y_pos < (y_min - smoothing) or y_pos > (y_max + smoothing) or z_pos < (z_min - smoothing) or z_pos > (z_max + smoothing):
                        flag = True
                        break
                if flag:
                    continue 

                # Compute RMS (Root Mean Square) energy
                rms = np.sqrt(np.mean(drone_signal ** 2))
                # Set a threshold (adjust as needed)
                threshold_min = drone_loudness * 0.01  # Modify based on actual conditions
                threshold_max = drone_loudness * 100
                # Check if the audio volume is too low or high
                if rms < threshold_min or rms > threshold_max:
                    print(f"Warning: The audio volume is extremely low or high! Fail to generate signal using {drone_type}. Skip")
                    continue 

                # save data
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d%H%M")
                folder_name = f"drone{iterations}_{timestamp}"
                output_wav_folder = os.path.join(output_folder, folder_name)
                os.makedirs(output_wav_folder, exist_ok=True)
                # save audio signals to wav file
                filename = f"drone{iterations}_{timestamp}.wav"
                sf.write(os.path.join(output_wav_folder, filename), drone_signal, sampling_rate)
                # write metadata to CSV file
                drone_inf = {
                    "drone_source": filename,
                    "drone_type": drone_type,
                    "wind_speed": wind_speed,
                    "flight_path": flight_path,
                    "drone_speed": state_speed
                }
                # save drone metadata
                jason_name = 'metadata.json'
                with open(os.path.join(output_wav_folder, jason_name), "w") as json_file:
                    json.dump([drone_inf], json_file, indent=4)
                iterations += 1
                pbar.update(1)
                


# get one example of drone flight
def get_example():
    # generate drone signal 
    drone_loudness = np.sqrt(loudness**2/pulse_len)
    drone_signal, flight_path, state_speed, rpm_freqs = get_drone_signal(starting_point, flight_state, forward_dir, drone_type, wind_speed, down_wind, drone_loudness, drone_white_noise, signaltype, pulse_len, change_time, s_shape_coef, climb_speed_max, sink_speed_max, speed_min)

    # visualize the drone flight results in the 3d space
    duration = flight_state[-1][-1] # seconds
    if drone_type == "S-9":
        num_rotor = 6
    else:
        num_rotor = 4
    plot_drone_scenario(np.array(flight_path), output_folder)
    plot_drone_frequency(rpm_freqs, num_rotor, duration, sampling_rate, output_folder)
    # frequency domain and save as wav form
    plot_signal_spectrogram(drone_signal, sampling_rate, output_folder)
   
    # save data
    folder_name = f"sample_example"
    output_wav_folder = os.path.join(output_folder, folder_name)
    os.makedirs(output_wav_folder, exist_ok=True)
    # save audio signals to wav file
    filename = f"drone_sample.wav"
    sf.write(os.path.join(output_wav_folder, filename), drone_signal, sampling_rate)
    # write metadata to CSV file
    drone_inf = {
        "drone_source": filename,
        "drone_type": drone_type,
        "wind_speed": wind_speed,
        "flight_path": flight_path,
        "drone_speed": state_speed
    }
    # save drone metadata
    jason_name = 'metadata.json'
    with open(os.path.join(output_wav_folder, jason_name), "w") as json_file:
        json.dump([drone_inf], json_file, indent=4)

# generate drone signal 
def get_drone_signal(simple_noise_flag, starting_point, flight_state, forward_dir, drone_type, wind_speed, down_wind, drone_loudness, drone_white_noise, signaltype, pulse_len, change_time, s_shape_coef, climb_speed_max, sink_speed_max, speed_min):
    if drone_type == "S-9":
        num_rotor = 6
    else:
        num_rotor = 4
    duration = flight_state[-1][-1] # seconds
    # get the fight path and RPM of the drone
    flight_path, rpm_std, state_speed = getFlightPath(starting_point, flight_state, drone_type, forward_dir, wind_speed, down_wind, climb_speed_max, sink_speed_max, speed_min)
    rpm_matrix = flightstate2RPM(flight_state, state_speed, climb_speed_max, sink_speed_max, drone_type)
    # get the drone signal
    freq_mod = get_micro_modulation(duration, sampling_rate, modulation_range, seg_dura_min, seg_dura_max) 
    drone_signal, rpm_freqs = getFullDroneSignal(simple_noise_flag, flight_state, rpm_matrix, rpm_std, freq_mod, sampling_rate, drone_loudness, drone_type, signaltype, drone_white_noise, num_rotor, change_time, s_shape_coef, pulse_len)
    return drone_signal, flight_path, state_speed, rpm_freqs



if __name__ == '__main__':
    # Generate a dataset consisting of a variable number (e.g., 1000-100000) of different drone situations
    # (flight paths, drone types, wind types...) and the resulting audio files.
    parser = argparse.ArgumentParser(description='Generate a dataset of drone audio files.')
    parser.add_argument('--num', type=int, default=5, help='Number of samples to generate different drone flight situations') 
    parser.add_argument('--exp', type=str, default='exp_test', help='Enter experiment ID')   
    parser.add_argument('--simple_noise', action='store_true', help='If set, the UAV signal will be simplified to white noise.')
    parser.add_argument('--drone_data', type=str, default='DroneAudioData', help='Path to the drone data file')  
    # generate one example of drone flight for reference, all results saved in "example" folder
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
            default_hyperpara_path = abspath +'/exp_config.yaml'
            shutil.copyfile(default_hyperpara_path, hyperpara_path)
            with open(hyperpara_path, 'r') as yamlfile:
                hyperparams = yaml.safe_load(yamlfile)            
        # Access hyperparameters
        speed_of_sound = hyperparams['speed_of_sound']
        sampling_rate = hyperparams['sampling_rate']
        change_time_min = hyperparams['change_time_min']
        change_time_max = hyperparams['change_time_max']
        s_shape_coef_min = hyperparams['s_shape_coef_min']
        s_shape_coef_max = hyperparams['s_shape_coef_max']
        drone_types = hyperparams['drone_types']
        drone_states = hyperparams['drone_states']
        drone_states_weights = hyperparams['drone_states_weights']
        climb_speed_max = hyperparams['climb_speed_max']
        sink_speed_max = hyperparams['sink_speed_max']
        speed_min = hyperparams['speed_min']
        modulation_range = hyperparams['modulation_range']
        seg_dura_min = hyperparams['seg_dura_min']
        seg_dura_max = hyperparams['seg_dura_max']
        decimal_places = hyperparams['decimal_places']
        signaltypes = hyperparams['signaltypes']
        pulse_len_min = int(hyperparams['pulse_len_min'])
        pulse_len_max = int(hyperparams['pulse_len_max'])
        drone_white_nois_min = hyperparams['drone_white_noise_min']
        drone_white_noise_max = hyperparams['drone_white_noise_max']
        loudness = hyperparams['loudness']
        state_num = hyperparams['state_num']

        # the flying area of the drone (m)
        x_min = hyperparams["x_min"]
        x_max = hyperparams["x_max"]
        y_min = hyperparams["y_min"]
        y_max = hyperparams["y_max"]
        z_min = hyperparams["z_min"]
        z_max = hyperparams["z_max"]
        # the range for the initial position of the drone
        x_init = hyperparams["x_init"]
        y_init = hyperparams["y_init"]
        z_init = hyperparams["z_init"]
        x_init_std = hyperparams["x_init_std"]
        y_init_std = hyperparams["y_init_std"]
        z_init_std = hyperparams["z_init_std"]
        # forword direction range
        azimuth_min = hyperparams["azimuth_min"]
        azimuth_max = hyperparams["azimuth_max"]
        # each duration in one state (second)
        state_dur_updown_min = hyperparams["state_dur_updown_min"]
        state_dur_updown_max = hyperparams["state_dur_updown_max"]
        state_dur_hover_min = hyperparams["state_dur_hover_min"]
        state_dur_hover_max = hyperparams["state_dur_hover_max"]
        state_dur_forward_min = hyperparams["state_dur_forward_min"]     
        state_dur_forward_max = hyperparams["state_dur_forward_max"]
        # wind speed range (m/s)
        wind_speed_min = hyperparams["wind_speed_min"]
        wind_speed_max = hyperparams["wind_speed_max"]

    else:
        output_folder = os.path.join(result_dir, "drone_example")
        os.makedirs(output_folder, exist_ok=True)
        # customization of the drone type, positions and weather in example_config.yaml
        # Load parameters from example config
        with open(os.path.join(result_dir, 'drone_example_config.yaml'), 'r') as yamlfile:
            example_params = yaml.safe_load(yamlfile)
        # Access example parameters
        drone_type = example_params['drone_type']
        starting_point = example_params['starting_point']
        starting_point.append(0.0)
        flight_state = example_params['flight_state']
        forward_dir = example_params['forward_dir']
        wind_speed = example_params['wind_speed']
        down_wind = example_params['down_wind']

        sampling_rate = example_params['sampling_rate']
        loudness = example_params['loudness']
        drone_white_noise = example_params['drone_white_noise']
        signaltype = example_params['signaltype']
        pulse_len = example_params['pulse_len']
        change_time = example_params['change_time']
        s_shape_coef = example_params['s_shape_coef']
        s_shape_coef = example_params['s_shape_coef']
        modulation_range = example_params['modulation_range']
        seg_dura_min = example_params['seg_dura_min']
        seg_dura_max = example_params['seg_dura_max']

        climb_speed_max = example_params['climb_speed_max']
        sink_speed_max = example_params['sink_speed_max']
        speed_min = example_params['speed_min']


    main()
