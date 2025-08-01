# drone_signal_generation.py
# This code generates emitted signals from drone body according to drone type, flying states, weather conditions and other parameters.
# (c) 2025, X. Yang, Fraunhofer IDMT, Germany, MIT License
# version 1.0, August 2025

import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.signal import stft, istft
import random


# drone has 4 states:  1 hover, 2 climb, 3 sink, 4 forward
# RPM(Revolutions Per Minute) is the most important parameter for the signal we will create
# PRM roughly estimated from Table 4 in Heutschi's paper, ignoring acceleration
# 5 types of drones mentioned in this paper, M2, I2, P4, F-4, S-9; more details in Table 1 of this paper, including weight and size;
# due to the lack of information about climb and sink RPM of F-4, assume these are the same as hover

# RPM information of drones as well as speed (m/s)
DRONES_INFO = {
    'M2': {
        'hover': 6150,
        'climb': 7800,
        'sink': 6000,
        'forward': 6600,
        'forward_speed': 15,
    },
    'I2': {
        'hover': 4350,
        'climb': 4950,
        'sink': 3600,
        'forward': 4800,
        'forward_speed': 15,
    },
    'P4': {
        'hover': 6300,
        'climb': 7800,
        'sink': 5400,
        'forward': 7200,
        'forward_speed': 15,
    },
    'S-9': {
        'hover': 5100,
        'climb': 6300,
        'sink': 4100,
        'forward': 6900,
        'forward_speed': 16,
    },
    'F-4': {
        'hover': 5250,
        'climb': 5800, 
        'sink': 4800,
        'forward': 5700,
        'forward_speed': 8,
    },
}  

# random_dev is related with the wind_condition and drone state
# Model parameters a and b [s/m] to estimate the normalised standard deviation of the rotational speed variation in dependency of the wind speed
# std = a + b * abs(wind_speed)
# hover_10 means the flight altitude of drones is 10 m
STD_WIND_INFO = {
    'hover_10': [0.005, 0.0052],
    'hover_20': [0.005, 0.0079],
    'hover_50': [0.006, 0.01],
    'climb': [0.001, 0.0084],
    'sink': [0.063, 0.0017],
    'forward_down': [0.012, 0.0033],
    'forward_up': [0.005, 0.0130]
}

# Rref (reference PRM) values for different drone types
Rref_VALUES = {'M2': 6540, 'I2': 4560, 'F-4': 6420, 'S-9': 6900} #'F-4' parameters have issue
# Frequencies corresponding to equalizer settings
EQ_FREQ = np.array([0, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000])
# Equalizer slope parameter S [dB/r/min] in equation for the different drone models
EQ_SLOPE = {
    'M2': np.array([1.5E-3, 1.5E-3, 2.9E-3, -1.3E-4, 4.4E-3, 3.8E-3, 4.6E-3, 4.4E-3, 4.3E-3, 4.0E-3, 3.6E-3, 3.7E-3, 2.7E-3, 3.8E-3, 3.8E-3, 3.6E-3, 3.5E-3, 3.5E-3, 3.3E-3, 3.3E-3, 3.3E-3, 2.3E-3, 7.5E-4, 1.8E-4]),
    'I2': np.array([6.6E-3, 6.6E-3, 5.4E-3, 9.8E-3, 1.1E-2, 8.7E-3, 9.3E-3, 1.1E-2, 8.1E-3, 8.5E-3, 8.8E-3, 7.7E-3, 7.8E-3, 8.0E-3, 8.0E-3, 8.0E-3, 8.1E-3, 8.4E-3, 8.4E-3, 9.0E-3, 8.3E-3, 7.9E-3, 6.2E-3, 4.9E-3]),
    'F-4': np.array([2.3E-4, 2.3E-4, 3.6E-2, 0, 0, 3.3E-2, 1.7E-2, 1.2E-2, 4.1E-2, 1.8E-2, 4.5E-2, 2.3E-2, 2.9E-2, 2.6E-2, 2.3E-2, 1.8E-2, 2.2E-2, 1.8E-2, 2.0E-2, 2.0E-2, 1.4E-2, 1.7E-2, 1.9E-2, 1.9E-2]), #2.3E-4 is weird
    'S-9': np.array([5.8E-3, 5.8E-3, 1.4E-3, 4.1E-3, 7.4E-3, 6.3E-3, 1.0E-2, 7.0E-3, 8.2E-3, 6.0E-3, 1.4E-3, 4.1E-3, 3.0E-3, 5.2E-4, 2.4E-3, 3.8E-3, 3.6E-3, 4.3E-3, 4.0E-3, 3.8E-3, 4.3E-3, 2.8E-3, 7.0E-4, 2.9E-3 ])
}



# the interface between flight_state and the basic RPM of the drone
# flight_state = [[State, Time]...], eg[["hover", 1], ["forward", 3]...]
# to derive [RPM Time] from the drone's flight state
# The result could be the matrix with RPM and time 
# eg, rpm_matrix = [[400,1.0],[200,2.0],[150,3.0],[200,4.0]]
def flightstate2RPM(flight_state, state_speed, climb_speed_max, sink_speed_max, drone_type="M2"):
    rpm_matrix = []
    for i, state in enumerate(flight_state):
        if state[0] == "forward":
            rpm_max = DRONES_INFO[drone_type]["forward"]
            rpm_min = DRONES_INFO[drone_type]["hover"]
            rpm = (rpm_max-rpm_min)*state_speed[i]/DRONES_INFO[drone_type]["forward_speed"] + rpm_min
        elif state[0] == "climb":
            rpm_max = DRONES_INFO[drone_type]["climb"]
            rpm_min = DRONES_INFO[drone_type]["hover"]
            rpm = (rpm_max-rpm_min)*state_speed[i]/climb_speed_max + rpm_min
        elif state[0] == "sink":
            rpm_max = DRONES_INFO[drone_type]["sink"]
            rpm_min = DRONES_INFO[drone_type]["hover"]
            rpm = (rpm_max-rpm_min)*state_speed[i]/sink_speed_max + rpm_min
        else:
            rpm = DRONES_INFO[drone_type][state[0]]
        rpm_matrix.append([rpm, state[1]])
    return np.array(rpm_matrix)


# function to get FlightPath and rpm_std (the variation of the rotational speed)
# eg. flightpath = [[10,10,2,0.0],[10,10,5,2.0],[10,10,2,4.0]]
def getFlightPath(starting_point, flight_state, drone_type="M2", forward_dir=[], wind_speed=2.5, down_wind=True, climb_speed_max=4, sink_speed_max=3, speed_min=2):
    flightpath = [starting_point]
    current_point = starting_point[:]
    forward_dir_ind = 0
    rpm_std = []
    state_speed = []
    for state_time in flight_state:
        if state_time[0] == "sink":
            sink_speed = random.uniform(speed_min,sink_speed_max)
            current_point[2] = current_point[2] - sink_speed*(state_time[1]-current_point[-1])
            std_para = STD_WIND_INFO['sink']
            state_speed.append(sink_speed)
        elif state_time[0] == "climb":
            climb_speed = random.uniform(speed_min,climb_speed_max)
            current_point[2] = current_point[2] + climb_speed*(state_time[1]-current_point[-1])
            std_para = STD_WIND_INFO['climb']
            state_speed.append(climb_speed)
        elif state_time[0] == "forward":
            forward_speed_max = DRONES_INFO[drone_type]["forward_speed"]
            # choose the forward speed randomly in the range
            speed = random.uniform(speed_min, forward_speed_max)
            distance = speed*(state_time[1]-current_point[-1])
            dir = forward_dir[forward_dir_ind]
            forward_dir_ind += 1
            current_point[0] = current_point[0] + dir[0]/np.linalg.norm(dir)*distance
            current_point[1] = current_point[1] + dir[1]/np.linalg.norm(dir)*distance
            current_point[2] = current_point[2] + dir[2]/np.linalg.norm(dir)*distance
            state_speed.append(speed)
            if down_wind == True:
                std_para = STD_WIND_INFO['forward_down']
            else:
                std_para = STD_WIND_INFO['forward_up']
        elif state_time[0] == "hover":
            # When the drone altitude is less than 15 meters, 
            # fluctuations are modeled using the 'hover_10' figure
            state_speed.append(0)
            if current_point[2] < 15:        
                std_para = STD_WIND_INFO['hover_10']
            else:
                if current_point[2] < 35:
                    std_para = STD_WIND_INFO['hover_20']
                else:
                    std_para = STD_WIND_INFO['hover_50']
        current_point[-1] = state_time[1]
        std = std_para[0] + std_para[1]*wind_speed 
        rpm_std.append([std, state_time[1]])
        flightpath.append(current_point[:])
    return flightpath, rpm_std, state_speed



# generate an impulse signal
def getOneImpulse(len_samples, pulse_ind, loudness, pulse_form = 'rect', pulse_len = 1):
    impulse = np.zeros(len_samples)
    for ind in pulse_ind:
        if (pulse_form == 'rect'):
            for kk in range(int(pulse_len)):
                impulse[ind+kk] = 1.0*loudness
        elif (pulse_form == 'alternate'):
            for kk in range(int(pulse_len)):
                impulse[ind+kk]= (-1.0*loudness)**(kk+1)
    return impulse

# sigmoid function for state transition
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def generate_s_shape_curve(x1, x2, fs, change_time=0.1, sig_coef=5):
    # Generate S-shape curve
    s_shape_curve = x1 + (x2 - x1) * sigmoid(sig_coef * (np.linspace(-1, 1,  int(fs * change_time))))
    return s_shape_curve

# add some random micro modulation of drone rpm during fight
def get_micro_modulation(duration, sampling_rate,  modulation_range = 6, seg_dura_min = 0.02, seg_dura_max = 0.05):
    # micro-modulation
    freq_mod = np.zeros(int(sampling_rate*duration))
    k = 0
    while (k + seg_dura_max*sampling_rate) < len(freq_mod):
        # Randomly choose the duration for each segment
        seg_duration = np.random.uniform(seg_dura_min, seg_dura_max)
        seg_len = int(seg_duration*sampling_rate)
        mod = np.random.uniform(-modulation_range/2, modulation_range/2)
        # add frequency modulation
        freq_mod[k:k+seg_len] = freq_mod[k:k+seg_len] + mod
        k += seg_len
    return freq_mod


# getOneRotorSignal function to output the signal of one rotor
def getOneRotorSignal(rpm_matrix, rpm_std, freq_mod, sampling_rate, drone_loudness, signaltype='rect', change_time=0.1, s_shape_coef=5, pulse_len=1, random_number=0):
    time_vec = rpm_matrix[:,1]
    rpm_vec = rpm_matrix[:,0]
    outsig = np.array([])
    base_freq = np.array([])
    pulse_ind = []
    freq_std = np.array([])

    start_time_s = 0.0
    # compute number of pulses
    nr_of_blades = 2
    sec_per_min = 60
    freq0 = rpm_vec[0]*nr_of_blades/sec_per_min
    
    # calculate the real-time base frequency and frequency std in each state, also considering the state transition
    for kk, end_time_s in enumerate(time_vec):
        delta_time = end_time_s - start_time_s
        act_rpm = rpm_vec[kk]
        nr_of_pulses = nr_of_blades*act_rpm
        freq1 = nr_of_pulses/sec_per_min
        act_pulse_freq = freq1*np.ones(int(sampling_rate*delta_time))
        std = rpm_std[kk][0]*np.ones(int(sampling_rate*delta_time))
        # add the state transition
        act_pulse_freq[:int(sampling_rate*change_time)] = generate_s_shape_curve(freq0, freq1, sampling_rate, change_time, s_shape_coef)   
        base_freq = np.concatenate((base_freq, act_pulse_freq))
        freq_std = np.concatenate((freq_std, std))    
        start_time_s = end_time_s
        freq0 = freq1

    # random start time
    start_time_s = int(random_number*sampling_rate/base_freq[0])
    while start_time_s < len(base_freq):
        # add some variation, add micro-modulation
        real_freq = np.random.normal(base_freq[start_time_s], base_freq[start_time_s]*freq_std[start_time_s]) + freq_mod[start_time_s]
        ind = int(sampling_rate/real_freq)
        pulse_ind.append(start_time_s + ind)
        base_freq[start_time_s] = real_freq
        start_time_s += ind
      
    outsig = getOneImpulse(int(sampling_rate*end_time_s), pulse_ind[:-2], drone_loudness, signaltype, pulse_len)
    return outsig, base_freq



# equalizer settings according to the paper
def get_equalizer_settings(fs, window_length, drone_type):
    """
    Generate equalizer settings for different drone states and frequencies.

    Parameters:
    - fs: Sampling rate.
    - window_length: Length of the window for short-time Fourier transform (STFT).
    - drone_type: Drone model type.
    Returns:
    - equalizer_settings: Dictionary containing interpolated equalizer settings for hover, climb, sink, forward states.
    """
    # Create dictionary with interpolated E values
    equalizer_settings = {}
    drone_states = ['hover', 'climb', 'sink', 'forward']
    # Calculate E values
    for state in drone_states:
        E_values = EQ_SLOPE[drone_type] * (DRONES_INFO[drone_type][state] - Rref_VALUES[drone_type])
        # Linear interpolation for frequencies in STFT range
        f_stft = np.fft.rfftfreq(window_length, 1 / fs)
        interp_function = interp1d(EQ_FREQ, E_values, kind='linear', fill_value='extrapolate')
        interpolated_E = interp_function(f_stft)
        equalizer_settings[state] = interpolated_E     
    return equalizer_settings

# Rotational speed dependent emission
def apply_equalizer(signal, flight_state, sampling_rate, window_length, equalizer_settings_db):
    """
    Apply equalizer settings to a time-domain signal.

    Parameters:
    - signal: Original time-domain signal.
    - flight_state
    - sampling_rate: Sampling rate of the signal.
    - window_length: Length of the window for short-time Fourier transform (STFT).
    - equalizer_settings_db: Equalizer settings in dB.

    Returns:
    - modified_signal: Signal after applying equalizer settings with the same shape as the original signal.
    """
    # Perform short-time Fourier transform (STFT)
    _, _, Zxx = stft(signal, sampling_rate, nperseg=window_length)
    
    state_ind = 0
    state = flight_state[state_ind][0]
    # Convert equalizer settings from dB to amplitude
    equalizer_settings_amplitude = 10 ** (equalizer_settings_db[state] / 20)
    
    # Apply equalizer settings to frequency amplitudes
    for i in range(Zxx.shape[1]):
        Zxx[:,i] = Zxx[:,i] * equalizer_settings_amplitude
        # change states
        if i*sampling_rate*window_length/2 > flight_state[state_ind][1] and state_ind < len(flight_state)-1:
            state_ind += 1
            state = flight_state[state_ind][0]
            equalizer_settings_amplitude = 10 ** (equalizer_settings_db[state] / 20)

    # Perform inverse short-time Fourier transform (ISTFT)
    _, modified_signal = istft(Zxx, sampling_rate)

    return modified_signal[:signal.shape[0]]



# Adding all signals of rotors would lead to the full drone signal
def getFullDroneSignal(simple_noise_flag, flight_state, rpm_matrix, rpm_std, freq_mod, sampling_rate, drone_loudness, drone_type="M2", signaltype='pulse', white_noise=0.01, num_rotor=4, change_time=0.1, s_shape_coef=5, pulse_len=1):
    full_drone_signal = np.zeros(int(rpm_matrix[-1][-1]*sampling_rate))
    rpm_freqs =[]
    
    for i in range(num_rotor):
        random_number = random.random()
        one_rotor_signal, rpm_freq = getOneRotorSignal(rpm_matrix, rpm_std, freq_mod, sampling_rate, drone_loudness, signaltype, change_time, s_shape_coef, pulse_len, random_number)
        full_drone_signal = full_drone_signal + one_rotor_signal
        rpm_freqs.append(rpm_freq)
    rpm_freqs = np.array(rpm_freqs)
    full_drone_signal = full_drone_signal + np.random.normal(0, white_noise, int(rpm_matrix[-1][-1]*sampling_rate))
    
    # Rotational speed dependent emission
    # need reference rotational speed — slope parameter, depends on frequency
    if drone_type != 'F-4':
        window_length = 1024 #256
        equalizer_settings_db = get_equalizer_settings(sampling_rate, window_length, drone_type)
        full_drone_signal = apply_equalizer(full_drone_signal, flight_state, sampling_rate, window_length, equalizer_settings_db)
    
    if simple_noise_flag:
        filtered_signal = np.random.normal(0, drone_loudness, int(rpm_matrix[-1][-1]*sampling_rate))
    else:
        #Add a simple low-pass filter to simulate changes in the sound
        order = 2 
        cutoff_freq_per = random.choice([0.08, 0.1]) 
        b, a = signal.butter(order, cutoff_freq_per)  # 2th-order low-pass filter, and its cutoff frequency percentage
        filtered_signal = signal.lfilter(b, a, full_drone_signal) # modified_signal


    return filtered_signal, rpm_freqs

