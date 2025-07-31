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
def plot_drone_scenario(flightpath):
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
    plt.savefig('example/drone_scenario.png')
  

def plot_drone_frequency(rpm_freqs, num_rotor, duration, sampling_rate):
    plt.figure(figsize=(10, 6))
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    plt.plot(t, np.transpose(rpm_freqs))
    plt.title('drone base frequency (rpm)')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    rotors = ['Rotor '+str(i+1) for i in range(num_rotor)]
    plt.legend(rotors)
    plt.savefig('example/drone_rpm.png')


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


def plot_signal_spectrogram(signal, sampling_rate):
    # Compute the signal's spectrogram
    win = 0.05
    nperseg = int(win*sampling_rate)
    # Compute the signal's spectrogram
    frequencies, times, spectrogram_data = spectrogram(signal, fs=sampling_rate, nperseg=nperseg)

    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity (dB)')
    max_frequency = 5000
    plt.ylim(0, max_frequency)# Adjust the frequency range based on your signal
    plt.savefig('example/spectrogram_drone.png')

    sf.write('example/drone_audio.wav', signal, sampling_rate)

# plot spherical coordinates
def plot_sph_angles_time_axis(azimuth_angles, elevation_angles, azimuth_angles_est, elevation_angles_est, frame_hop, savepath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) 
    # x-axis represents time
    time_steps = [(i+1)*frame_hop for i in range(len(elevation_angles_est))]

    # Plotting the scatter plot
    ax1.scatter(time_steps, elevation_angles[:len(elevation_angles_est)], label='Actual Elevation Angles', color='blue', s=1)
    ax1.scatter(time_steps, elevation_angles_est, label='Estimated Elevation Angles', color='red',s=1)
    # Adding labels and title
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Elevation Angle')
    ax1.set_title('Elevation Angle Comparison')
    ax1.legend()

    # Plotting the scatter plot
    ax2.scatter(time_steps, azimuth_angles[:len(elevation_angles_est)], label='Actual Azimuth Angles', color='blue', s=1)
    ax2.scatter(time_steps, azimuth_angles_est, label='Estimated Azimuth Angles', color='red', s=1)

    # Adding labels and title
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Azimuth Angle')
    ax2.set_title('Azimuth Angle Comparison')
    ax2.legend()
    # Displaying the plot
    plt.savefig(savepath+'/est_sph_angles_time_axis.png')


def plot_sph_angles(azimuth_angles, elevation_angles, azimuth_angles_est, elevation_angles_est, savepath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  
    # real angles
    ax1.scatter(azimuth_angles, elevation_angles, marker='o', color='blue', s=1)
    ax1.set_xlabel('Azimuth Angle')
    ax1.set_ylabel('Elevation Angle')
    ax1.set_title('Real Angle Visualization')
    ax1.grid(True)

    # estimated angles
    ax2.scatter(azimuth_angles_est, elevation_angles_est, marker='o', color='blue', s=1)
    ax2.set_xlabel('Azimuth Angle')
    ax2.set_ylabel('Elevation Angle')
    ax2.set_title('Estimated Angle Visualization')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(savepath+'/real_and_est_sph_angles.png')

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


def plot_doa_error_over_time(DOA_error, frame_hop, savepath):
    time_points = [(i+1)*frame_hop for i in range(len(DOA_error))]
    plt.figure(figsize=(10, 6))
    plt.scatter(time_points, DOA_error, label='DOA Error', color='red', s=2)
    plt.plot(time_points, [20] * len(time_points), linestyle='-', label='ER20', color='blue')

    plt.title('DOA Error Over Time')
    plt.xlabel('Time Points (seconds)')
    plt.ylabel('DOA Error')
    plt.legend()

    plt.grid(True)
    plt.savefig(savepath+'/doa_error_over_time.png')

def plot_doa_error_vs_distance(distances, doa_errors, snr_values, savepath):
    """
    Plot DOA Error vs. Distance scatter plot, where the color of the points is determined by SNR values.
    :param distances: Array of distances from the drone to the microphone array.
    :param doa_errors: Array of DOA errors (in degrees).
    :param snr_values: Array of SNR values.
    """
    distances = np.array(distances)
    doa_errors = np.array(doa_errors)
    if snr_values != []:
        snr_values = np.array(snr_values)
        # Create a color map
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(snr_values), vmax=max(snr_values))
        # SNR average
        unique_snr_values = np.unique(snr_values)
        average_doa_errors = [np.mean(doa_errors[snr_values == snr]) for snr in unique_snr_values]

        # Create scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(distances, doa_errors, c=snr_values, cmap=cmap, norm=norm, alpha=0.7, edgecolors='w', s=100)

        for snr, avg_doa in zip(unique_snr_values, average_doa_errors):
            #plt.scatter([], [], c='k', alpha=0.7, s=100, label=f'SNR: {snr}, Avg DOA Error: {avg_doa:.2f}')
            color = cmap(norm(snr))
            plt.scatter([], [], c=color, alpha=0.7, s=100, label=f'SNR: {snr}, Avg DOA Error: {avg_doa:.2f}')
        plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='upper left')
        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('SNR (dB)')
    else:
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(distances, doa_errors, alpha=0.7, s=100)

    # Add labels and title
    plt.xlabel('Distance to Microphone Array (m)')
    plt.ylabel('DOA Error (degree)')
    plt.title('DOA Error vs. Distance with Different SNR Levels')
    # Show plot
    plt.savefig(savepath+'/doa_error_vs_distance.png')


def plot_doa_error_vs_distance_real_snr(distances, doa_errors, snr_values, savepath):
    """
    Plot DOA Error vs. Distance scatter plot, where the color of the points is determined by SNR values.
    :param distances: Array of distances from the drone to the microphone array.
    :param doa_errors: Array of DOA errors (in degrees).
    :param snr_values: Array of SNR values.
    """
    distances = np.array(distances)
    doa_errors = np.array(doa_errors)
    if snr_values != []:
        snr_values = np.array(snr_values)
        # Create a color map
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(snr_values), vmax=max(snr_values))
        # SNR average
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(distances, doa_errors, c=snr_values, cmap=cmap, norm=norm, alpha=0.7, edgecolors='w', s=100)
        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('real SNR (dB)')
    else:
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(distances, doa_errors, alpha=0.7, s=100)

    # Add labels and title
    plt.xlabel('Distance to Microphone Array (m)')
    plt.ylabel('DOA Error (degree)')
    plt.title('DOA Error vs. Distance with Different SNR Levels')
    # Show plot
    plt.savefig(savepath+'/doa_error_vs_distance_real_snr.png')

def plot_signal_spectrogram(signal, sampling_rate, example_path):
    # Compute the signal's spectrogram
    win = 0.1 #0.05
    nperseg = int(win*sampling_rate)
    # Compute the signal's spectrogram
    frequencies, times, spectrogram_data = spectrogram(signal, fs=sampling_rate, nperseg=nperseg)
    # Plot the spectrogram
    vmin, vmax = -125, -95
    #vmin, vmax = -100, -40
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data), shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
    #plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity (dB)')
    max_frequency = 7000 #3000
    plt.ylim(0, max_frequency)# Adjust the frequency range based on your signal
    plt.savefig(f'{example_path}/spectrogram_drone.png')


def plot_drone_path_with_array_center_2d(drone_positions, mic_positions, example_path, frame_hop, interesting_points=None, interesting_period=None):
    """
    Plot the drone's flight path in 2D space and mark the microphone array center, 
    as well as the start and end points of the path.
    
    Parameters:
        drone_positions (list of list): Drone path coordinates, where each point is [x, y, z].
        mic_positions (list of list): List of microphone coordinates, each mic at [x, y, z].
        example_path (str): Path to save the resulting image.
    """
    
    # Convert drone and microphone positions to numpy arrays
    drone_positions = np.array(drone_positions)
    mic_positions = np.array(mic_positions)
    
    # Calculate the center of the microphone array (using x and y only)
    array_center = np.mean(mic_positions, axis=0)
    
    # Create a 2D plot
    fig, ax = plt.subplots()
 
    # Plot the drone flight path in 2D (only x and y coordinates)
    ax.plot(drone_positions[:, 0], drone_positions[:, 1], 
            color='b', marker='o', linestyle='-', label='Drone Path', linewidth=0.4, markersize=0.3)
    
    # Mark the start and end points (plot after the path so they are on top)
    ax.scatter(drone_positions[0, 0], drone_positions[0, 1], color='purple', s=20, label='Start Point', zorder=5)
    ax.scatter(drone_positions[-1, 0], drone_positions[-1, 1], color='orange', s=20, label='End Point', zorder=5)

    # Mark the array center
    ax.scatter(array_center[0], array_center[1], color='g', marker='x', s=50, label='Array Center', zorder=5)
    
    # Annotate the array center coordinates
    ax.text(array_center[0], array_center[1], 
            f'({array_center[0]:.2f}, {array_center[1]:.2f})', 
            color='black', ha='right', va='bottom')
    
    # Plot "interesting points" if any
    if interesting_points:
        for point in interesting_points:
            xyz, time = point
            ax.scatter(xyz[0], xyz[1], color='red', s=25, marker='*', zorder=5)  # Mark interesting points with a red star
            ax.text(xyz[0], xyz[1], f' {time}s, ({xyz[0]:.2f}, {xyz[1]:.2f})', color='red', fontsize=8, ha='right')
    if interesting_period:
        for period in interesting_period:
            start_idx = int(period[0]/frame_hop)
            end_idx = int(period[1]/frame_hop)
            if end_idx >= len(drone_positions[:, 0]):
                end_idx = len(drone_positions[:, 0])-1
            #ax.scatter(drone_positions[start_idx, 0], drone_positions[start_idx, 1], color='red', s=25, marker='*', zorder=5)
            ax.plot(drone_positions[start_idx:end_idx+1, 0], drone_positions[start_idx:end_idx+1, 1], 
                color='red', linestyle='-', linewidth=2, zorder=5)
            ax.text(drone_positions[start_idx, 0], drone_positions[start_idx, 1], f' {period[0]}s', color='red', fontsize=8, ha='right')
            ax.text(drone_positions[end_idx, 0], drone_positions[end_idx, 1], f' {period[1]}s', color='red', fontsize=8, ha='right')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Ensure equal scaling on both x and y axes
    #ax.set_aspect('equal', 'box')
    
    # Add a legend
    ax.legend()
    
    # Set the title
    ax.set_title("2D Drone Flight Path with Array Center")
    
    # Save the plot as an image file
    plt.savefig(example_path+"/RealDroneFlight2D.png")
