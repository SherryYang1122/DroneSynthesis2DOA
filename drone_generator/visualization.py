import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import soundfile as sf

def plot_microphone_array(mic_pos_matrix):
    # Create a 3D coordinate axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the microphone position
    for i, mic_pos in enumerate(mic_pos_matrix):
        label = 'Microphone {}'.format(i)
        ax.scatter(mic_pos[0], mic_pos[1], mic_pos[2], marker='o', label=label)
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set plot legend
    ax.legend()

    # Save the plot as an image file
    plt.savefig('example/microphone_array.png')


# plot drone flight and micphone array in 3D space
def plot_drone_mic_scenario(mic_pos_matrix, wall_coeff, flight_path):
    # Create a 3D coordinate axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the ground plane (z = 0)
    xx, yy = np.meshgrid(np.linspace(-30, 30, 10), np.linspace(-30, 30, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='gray', alpha=0.2)
    # Generate points for the wall, ax+by+cz+d=0
    a, b, c, d = wall_coeff
    if b!=0:
        x = np.linspace(-30, 30, 10)
        z = np.linspace(0, 15, 10)
        xx, zz = np.meshgrid(x, z)
        yy = (-d - a*xx - c*zz) / b
    else:
        y = np.linspace(-30, 30, 10)
        z = np.linspace(0, 15, 10)
        yy, zz = np.meshgrid(y, z)
        xx = (-d - b*yy - c*zz) / a
    # Plot the plane
    ax.plot_surface(xx, yy, zz, color='blue', alpha=0.3)

    # Plot the microphone position
    #for i, mic_pos in enumerate(mic_pos_matrix):
    i = 0
    mic_pos = mic_pos_matrix[i]
    label = 'Microphone array'
    ax.scatter(mic_pos[0], mic_pos[1], mic_pos[2], color='red', marker='o', label=label)

    # Plot the line connecting microphone and ground projection
    mic_pos_shift = mic_pos_matrix[0]
    ax.plot([mic_pos_shift[0], mic_pos_shift[0]], [mic_pos_shift[1], mic_pos_shift[1]], [0, mic_pos_shift[2]], color='black', linestyle='--')

    # Plot the drone positions
    drone_pos = flight_path[0]
    ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2], color='green', marker='x', label='Drone')
    # Extract coordinates and time
    coordinates = np.array([state[:3] for state in flight_path])
    time_points = np.array([state[3] for state in flight_path])
    for i in range(len(coordinates) - 1):
        if np.all(coordinates[i] == coordinates[i + 1]):
            ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], 'Hover', zdir='y', ha='center', fontsize=8, color='blue', weight='bold')
        else:
            ax.plot(coordinates[i:i+2, 0], coordinates[i:i+2, 1], coordinates[i:i+2, 2])
    # Mark coordinate points
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='red', marker='o')
    # Mark time points
    i = 0
    while i < len(time_points):
        if (i+1) < len(time_points):
            if list(coordinates[i])==list(coordinates[i+1]):
                ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], f'Time: {time_points[i]} -- {time_points[i+1]}', ha='left', fontsize=8)
                i = i + 2
            else:
                ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], f'Time: {time_points[i]}', fontsize=8)
                i = i + 1          
        else:
            ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], f'Time: {time_points[i]}', fontsize=8)
            i = i + 1

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Flight Path and Microphone Array')

    # Set plot legend
    ax.legend()
    # Show and save the plot
    plt.savefig('example/drone_mic_scenario.png')


# plot only drone flight in 3D space
def plot_drone_scenario(flightpath, output_folder):
    # Extract coordinates and time
    coordinates = np.array([state[:3] for state in flightpath])
    time_points = np.array([state[3] for state in flightpath])
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the path
    for i in range(len(coordinates) - 1):
        if np.all(coordinates[i] == coordinates[i + 1]):
            ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], 'Hover', zdir='y', ha='center', fontsize=8, color='blue', weight='bold')
        else:
            ax.plot(coordinates[i:i+2, 0], coordinates[i:i+2, 1], coordinates[i:i+2, 2])
    # Mark coordinate points
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='red', marker='o')
    # Mark time points
    i = 0
    while i < len(time_points):
        if (i+1) < len(time_points):
            if list(coordinates[i])==list(coordinates[i+1]):
                ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], f'Time: {time_points[i]} -- {time_points[i+1]}', ha='left', fontsize=8)
                i = i + 2
            else:
                ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], f'Time: {time_points[i]}', fontsize=8)
                i = i + 1          
        else:
            ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], f'Time: {time_points[i]}', fontsize=8)
            i = i + 1
    # Set plot attributes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Path')
    plt.savefig(output_folder+'/drone_scenario.png')
  

def plot_drone_frequency(rpm_freqs, num_rotor, duration, sampling_rate, output_folder):
    plt.figure(figsize=(10, 6))
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    plt.plot(t, np.transpose(rpm_freqs))
    plt.title('drone base frequency (rpm)')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    rotors = ['Rotor '+str(i+1) for i in range(num_rotor)]
    plt.legend(rotors)
    plt.savefig(output_folder+'/drone_rpm.png')


def plot_mic_array_siganls(t, signal):
    # Create subplots
    num_channels = signal.shape[0]
    fig, axs = plt.subplots(num_channels // 2, 2)
    # Plot each channel's signal
    for i, ax in enumerate(axs.flatten()):
        ax.plot(t, signal[i])
        title = 'Received Signal at the Microphone {}'.format(i)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
    # Adjust layout
    plt.tight_layout()
    plt.savefig('example/signal_from_microphones.png')


def plot_signal_spectrogram(signal, sampling_rate, output_folder):
    # Compute the signal's spectrogram
    win = 0.07
    nperseg = int(win*sampling_rate)
    # Compute the signal's spectrogram
    frequencies, times, spectrogram_data = spectrogram(signal, fs=sampling_rate, nperseg=nperseg)

    # Plot the spectrogram
    vmin, vmax = -100, -40
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data), shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
    #plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity (dB)')
    max_frequency = 3000
    plt.ylim(0, max_frequency)# Adjust the frequency range based on your signal
    plt.savefig(output_folder+'/spectrogram_drone.png')
    sf.write(output_folder+'/drone_audio.wav', signal, sampling_rate)

    time = np.arange(len(signal)) / sampling_rate
    # Plot the waveform
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal, color='b', alpha=0.7)
    # Labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Signal Waveform")
    plt.savefig(output_folder+'/audio_waveform_drone.png')

# plot spherical coordinates
def plot_sph_angles_time_axis(azimuth_angles, elevation_angles, azimuth_angles_est, elevation_angles_est, window_size, overlap, sampling_rate):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) 
    # x-axis represents time
    time_steps = [(i+1)*(window_size - overlap)/sampling_rate for i in range(len(elevation_angles_est))]

    # Plotting the scatter plot
    ax1.scatter(time_steps, elevation_angles[1:1+len(elevation_angles_est)], label='Actual Elevation Angles', color='blue', s=10)
    ax1.scatter(time_steps, elevation_angles_est, label='Estimated Elevation Angles', color='red',s=10)
    # Adding labels and title
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Elevation Angle')
    ax1.set_title('Elevation Angle Comparison')
    ax1.legend()

    # Plotting the scatter plot
    ax2.scatter(time_steps, azimuth_angles[1:1+len(elevation_angles_est)], label='Actual Azimuth Angles', color='blue', s=10)
    ax2.scatter(time_steps, azimuth_angles_est, label='Estimated Azimuth Angles', color='red', s=10)

    # Adding labels and title
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Azimuth Angle')
    ax2.set_title('Azimuth Angle Comparison')
    ax2.legend()
    # Displaying the plot
    plt.savefig('example/est_sph_angles_time_axis.png')


def plot_sph_angles(azimuth_angles, elevation_angles, azimuth_angles_est, elevation_angles_est):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  
    # real angles
    ax1.scatter(azimuth_angles, elevation_angles, marker='o', color='blue', s=10)
    ax1.set_xlabel('Azimuth Angle')
    ax1.set_ylabel('Elevation Angle')
    ax1.set_title('Real Angle Visualization')
    ax1.grid(True)

    # estimated angles
    ax2.scatter(azimuth_angles_est, elevation_angles_est, marker='o', color='blue', s=10)
    ax2.set_xlabel('Azimuth Angle')
    ax2.set_ylabel('Elevation Angle')
    ax2.set_title('Estimated Angle Visualization')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('example/real_and_est_sph_angles.png')

# fibonacci sphere visualization
def plot_sphere_points(points):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(points[0,:], points[1,:], points[2,:])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.savefig('example/fibonacci_sphere.png')