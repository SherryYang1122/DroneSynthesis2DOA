import numpy as np
from geometry import reflection_distance, cart2sph
import scipy.signal as signal

def get_drone_position(flight_path, sampling_rate):
    positions = []
    for i in range(len(flight_path) - 1):
        start_point = flight_path[i]
        end_point = flight_path[i + 1]
        start_time = start_point[3]
        end_time = end_point[3]
        time_diff = end_time - start_time
        num_samples = int(time_diff * sampling_rate)
        for j in range(num_samples):
            alpha = j / num_samples
            interpolated_position = (1 - alpha) * start_point[:3] + alpha * end_point[:3]
            positions.append(interpolated_position)
    positions.append(flight_path[-1][:3])  # Adding the last position
    return np.array(positions)


# Microphone signal reception model -- with ground reflection & wall reflection
def receive_reflected_signal(t, drone_pos, mic_pos, wall_coeff, ground_coeff, mic2wall_sym, mic2ground_sym, speed_of_sound, drone_signal, sampling_rate, ground_ref_coeff, wall_ref_coeff):
    # Calculate the distance between the microphone and the drone
    mic_distance = np.linalg.norm(drone_pos - mic_pos)
    tau_1 = mic_distance / speed_of_sound
    
    # Calculate the distance to the ground reflection point
    microphone_distance_to_ground_reflection = reflection_distance(mic_pos, mic2ground_sym, drone_pos, ground_coeff)
    tau_2 = microphone_distance_to_ground_reflection / speed_of_sound
    
    # Calculate the distance to the wall reflection point
    microphone_distance_to_wall_reflection = reflection_distance(mic_pos, mic2wall_sym, drone_pos, wall_coeff)
    tau_3 = microphone_distance_to_wall_reflection / speed_of_sound
    
    # Calculate signal attenuation factors
    attenuation1 = 1.0 / (mic_distance**2)                           
    attenuation2 = ground_ref_coeff * 1.0 / (microphone_distance_to_ground_reflection**2)
    if microphone_distance_to_wall_reflection!= 0:
        attenuation3 = wall_ref_coeff * 1.0 / (microphone_distance_to_wall_reflection**2)
    else:
        attenuation3 = 0

    received_signal1 = (linear_interpolation_signal((t - tau_1)*sampling_rate, drone_signal) if ((t - tau_1)>=0) else 0) * attenuation1
    received_signal2 = 0 if tau_2 == 0 else (linear_interpolation_signal((t - tau_2)*sampling_rate, drone_signal) if ((t - tau_2)>=0) else 0) * attenuation2
    received_signal3 = 0 if tau_3 == 0 else (linear_interpolation_signal((t - tau_3)*sampling_rate, drone_signal)if ((t - tau_3)>=0) else 0) * attenuation3
    return received_signal1 + received_signal2 + received_signal3

def linear_interpolation_signal(x, signal):
    x1 = int(x)
    x2 = int(x) + 1
    y1 = signal[x1]
    y2 = signal[x2]
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def design_highshelf(fc, Q, gain, fs):
    """Calculates filter coefficients for a High Shelf filter.

    Parameters
    ----------
    fc : float
        Center frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    gain : expressed in db
        Linear gain for the shelved frequencies
    fs :  float
        Sample rate in Hz

    Returns
    -------
    b : ndarray
        "b" (feedforward) coefficients of the filter
    a : ndarray
        "a" (feedback) coefficients of the filter
    """
    # Convert gain from dB to amplitude
    gain = 10 ** (gain/20)
    A = np.sqrt(gain)
    wc = 2 * np.pi * fc / fs
    wS = np.sin(wc)
    wC = np.cos(wc)
    beta = np.sqrt(A) / Q

    a0 = ((A+1.0) - ((A-1.0) * wC) + (beta*wS))

    b = np.zeros(3)
    a = np.zeros(3)
    b[0] = A*((A+1.0) + ((A-1.0)*wC) + (beta*wS)) / a0
    b[1] = -2.0*A * ((A-1.0) + ((A+1.0)*wC)) / a0
    b[2] = A*((A+1.0) + ((A-1.0)*wC) - (beta*wS)) / a0

    a[0] = 1
    a[1] = 2.0 * ((A-1.0) - ((A+1.0)*wC)) / a0
    a[2] = ((A+1.0) - ((A-1.0)*wC)-(beta*wS)) / a0
    return b, a

def get_radiation_signal(x, drone_pos, mic_pos, sampling_rate, window_time):
    filtered_signal = np.zeros_like(x)
    i = 0
    window_len = int(window_time*sampling_rate)
    while (i+window_len) <= len(x):
        angle = cart2sph(mic_pos - drone_pos[i])
        theta = 90-angle[0]
        # equation and paramters from Heutschi's paper
        gain = -0.0011*theta*theta+0.194*np.abs(theta)-4.9
        b, a = design_highshelf(fc=500, Q=0.5, gain=gain, fs=sampling_rate)
        input = x[i:i+window_len]
        if i >= 3:
            zi = signal.lfiltic(b, a, y=filtered_signal[i-1:i-3:-1], x=x[i-1:i-3:-1])
            y, _ = signal.lfilter(b, a, input, zi=zi)
        else:
            y = signal.lfilter(b, a, input)
        filtered_signal[i:i+window_len] = y
        i += window_len
    if i < len(x):
        filtered_signal[i:] = x[i:]
    return filtered_signal

# get real directions of drone
def get_doa_real(drone_position, mic_pos_centre, frame_length, sampling_rate):
    # actual drone sph angles
    doa_cart = []
    window_size = int(sampling_rate * frame_length)  
    frame_num = drone_position.shape[0]//window_size 
    for i in range(frame_num):
        doa_cart.append(drone_position[i*window_size]-mic_pos_centre)
    return doa_cart