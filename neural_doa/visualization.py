# visualization.py
# This code provides visualization functions for DOA estimation results and acoustic analysis.
# (c) 2025, X. Yang, Fraunhofer IDMT, Germany, MIT License
# version 1.0, August 2025

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import soundfile as sf
from IPython.display import Audio
import os

def plot_microphone_array(mic_pos_matrix):
    # Create a 3D coordinate axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mic_array = mic_pos_matrix[0]

    # Plot the microphone position
    z_values = [mic[2] for mic in mic_array]
    for i, mic_pos in enumerate(mic_array):
        label = 'Microphone {}'.format(i)
        ax.scatter(mic_pos[0], mic_pos[1], mic_pos[2], marker='o', label=label)
        ax.plot([mic_pos[0], mic_pos[0]], [mic_pos[1], mic_pos[1]], [min(z_values), mic_pos[2]], color='grey', linestyle='--')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Microphone array example')

    # Set plot legend
    ax.legend()

    # Save the plot as an image file
    plt.savefig('drone_classical_output/microphone_array.png')
    #plt.show()


def plot_drone_scenario(mic_pos_matrix, wall_coeff, flight_path):
    # Create a 3D coordinate axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x and y values from the coordinates list.
    x_values = []
    y_values = []
    z_values = []
    for mic_array in mic_pos_matrix:
        for mic in mic_array:
            x_values.append(mic[0])
            y_values.append(mic[1])
            z_values.append(mic[2])
    for waypoint in flight_path:
        x_values.append(waypoint[0])
        y_values.append(waypoint[1])
        z_values.append(waypoint[2])

    # Generate points for the wall
    a, b, c, d = wall_coeff
    margin = 5
    x = np.linspace(min(x_values)-margin, max(x_values)+margin, 10)
    margin_height = 1
    z = np.linspace(0, max(z_values)+margin_height, 10)
    xx, zz = np.meshgrid(x, z)
    yy = (-d - a*xx - c*zz) / b
    # Plot the wall plane
    ax.plot_surface(xx, yy, zz, color='blue', alpha=0.3)

    # Plot the ground plane (z = 0)
    y_values.append(np.max(yy))
    y_values.append(np.min(yy))

    xx, yy = np.meshgrid(np.linspace(min(x_values)-margin, max(x_values)+margin, 10), np.linspace(min(y_values)-margin, max(y_values)+margin, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='gray', alpha=0.2)


    # Plot the microphone position
    for i, mic_pos in enumerate(mic_pos_matrix):
        label = 'Microphone array {}'.format(i)
        mic0 = mic_pos[0]
        ax.scatter(mic0[0], mic0[1], mic0[2], marker='o', label=label)

    # Plot the drone position
    drone_pos = flight_path[0]
    ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2], color='green', marker='x', label='Drone')

    # Plot the line connecting microphone and ground projection
    for mic_pos in mic_pos_matrix:
        mic0 = mic_pos[0]
        ax.plot([mic0[0], mic0[0]], [mic0[1], mic0[1]], [0, mic0[2]], color='black', linestyle='--')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot legend
    ax.legend()

    # Show and save the plot
    plt.savefig('drone_classical_output/drone_scenario.png')
    #plt.show()


def plot_mic_array_siganls(t, signal):
    # Create subplots
    num_channels = signal.shape[0]
    fig, axs = plt.subplots(num_channels // 2, 2, figsize=(15, 5*(num_channels // 2)))

    # Plot each channel's signal
    for i, ax in enumerate(axs.flatten()):
        ax.plot(t, signal[i])
        title = 'Received signal at Microphone {}'.format(i)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    # Adjust layout
    # plt.tight_layout()
    plt.subplots_adjust()
    plt.savefig('drone_classical_output/signal_from_one_microphone_array.png')
    #plt.show()


def plot_signal_spectrogram(signal, sampling_rate):
    # Compute the signal's spectrogram
    frequencies, times, spectrogram_data = spectrogram(signal, fs=sampling_rate)

    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity (dB)')
    max_frequency = 10000
    plt.ylim(0, max_frequency)# Adjust the frequency range based on your signal
   
    plt.savefig('drone_classical_output/spectrogram_one_microphone.png')
    #plt.show()

    #wav.write('received_mic_signal.wav', sampling_rate, signal)
    sf.write('drone_classical_output/received_mic_signal.wav', signal, sampling_rate)
    Audio(signal, rate=sampling_rate)


def plot_sph_angles_time_axis(azimuth_angles, elevation_angles, azimuth_angles_est, elevation_angles_est, window_size, overlap, sampling_rate):
    fig, ax = plt.subplots(len(azimuth_angles), 2, figsize=(15, 5.5*len(azimuth_angles))) 
    ax = np.array(ax).reshape(len(azimuth_angles), 2)
    # x-axis represents time
    time_steps = [(i+1)*(window_size - overlap)/sampling_rate for i in range(len(elevation_angles_est[0]))]
    for i in range(len(azimuth_angles)):
        # Plotting the scatter plot
        ax[i,0].scatter(time_steps, elevation_angles[i][:len(elevation_angles_est[0])], label='Actual Elevation Angles', color='blue', s=10)
        ax[i,0].scatter(time_steps, elevation_angles_est[i], label='Estimated Elevation Angles', color='red',s=10)
        # Adding labels and title
        ax[i,0].set_xlabel('Time')
        ax[i,0].set_ylabel('Elevation Angle')
        ax[i,0].set_title(f'Elevation Angle Comparison in Mircphone Array {i}')
        ax[i,0].legend()

        # Plotting the scatter plot
        ax[i,1].scatter(time_steps, azimuth_angles[i][:len(elevation_angles_est[0])], label='Actual Azimuth Angles', color='blue', s=10)
        ax[i,1].scatter(time_steps, azimuth_angles_est[i], label='Estimated Azimuth Angles', color='red', s=10)
        # Adding labels and title
        ax[i,1].set_xlabel('Time')
        ax[i,1].set_ylabel('Azimuth Angle')
        ax[i,1].set_title(f'Azimuth Angle Comparison in Mircphone Array {i}')
        ax[i,1].legend()

    # Displaying the plot
    #plt.subplots_adjust()
    plt.savefig('drone_classical_output/est_sph_angles_time_axis.png')


def plot_sph_angles(azimuth_angles, elevation_angles, azimuth_angles_est, elevation_angles_est):
    fig, ax = plt.subplots(len(azimuth_angles), 2, figsize=(15, 5*len(azimuth_angles))) 
    ax = np.array(ax).reshape(len(azimuth_angles), 2)
    for i in range(len(azimuth_angles)):
        # real angles
        ax[i, 0].scatter(azimuth_angles[i], elevation_angles[i], marker='o', label='Real Angles', s=10)
        ax[i, 0].scatter(azimuth_angles_est[i], elevation_angles_est[i], marker='o', label='Estimated Angles', s=10)
        ax[i, 0].set_xlabel('Azimuth Angle')
        ax[i, 0].set_ylabel('Elevation Angle')
        ax[i, 0].set_title(f'Angle Visualization in Mircphone Array {i}')
        ax[i, 0].grid(True)
        ax[i, 0].legend()

        # Convert known points
        known_points = [sph2cart(el, az) for az, el in zip(azimuth_angles[i], elevation_angles[i])]
        # Convert estimated points
        estimated_points = [sph2cart(el, az) for az, el in zip(azimuth_angles_est[i], elevation_angles_est[i])]
        # Plot known points
        ax[i,1] = fig.add_subplot(len(azimuth_angles), 2, 2+i*2, projection='3d')
        known_x, known_y, known_z = zip(*known_points)
        ax[i,1].scatter(known_x, known_y, known_z, marker='o', label='Real Points')
        # Plot estimated points
        estimated_x, estimated_y, estimated_z = zip(*estimated_points)
        ax[i,1].scatter(estimated_x, estimated_y, estimated_z, marker='^', label='Estimated Points')
        # Set axis labels
        ax[i,1].set_xlabel('X')
        ax[i,1].set_ylabel('Y')
        ax[i,1].set_zlabel('Z')
        ax[i,1].set_title(f'3D Plot of Unit Directional Vectors in Mircphone Array {i}')
        # Show legend
        ax[i,1].legend()
    #plt.tight_layout()
    plt.savefig('drone_classical_output/real_and_est_sph_angles.png')


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
    plt.savefig('drone_classical_output/fibonacci_sphere.png')


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
    plt.savefig(os.path.join(savepath, 'doa_error_vs_distance.png'))

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
    plt.savefig(os.path.join(savepath, 'doa_error_vs_distance_real_snr.png'))

def plot_localization_result(result, result_name, num_classes, max_error, column_num, max_subfig, output_folder):
    num = len(result)
    if num == 0:
        return None
    else:
        num_fig = int((num-1)/max_subfig) + 1
    for i in range(num_fig):
        if num >= (i+1)*max_subfig:
            num_subfig = max_subfig
        else:
            num_subfig = num - i*max_subfig
        fig, ax = plt.subplots((num_subfig-1)//column_num + 1, column_num, figsize=(18, 5.5*((num_subfig-1)//column_num + 1))) 
        for j in range(num_subfig):
            if num_subfig > column_num:
                ax1 = ax[j//column_num, j%column_num]
            else:
                ax1 = ax[j]
            predicted = result[i*max_subfig+j][0]
            expected = result[i*max_subfig+j][1]
            DOAerror = result[i*max_subfig+j][2]
            sample_name = result[i*max_subfig+j][3]
            ax1.plot(predicted, linestyle='-', label=f'DL Pred', alpha=0.5)
            ax1.plot(expected, linestyle='-', label=f'Exp', alpha=0.5)
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Labels (Prediction / Expectation)', color='k')
            ax1.tick_params(axis='y', labelcolor='k')
            ax1.set_ylim(-1, num_classes+1)
            ax1.legend(loc='upper left')
            # Add a second y-axis for DOA error
            ax2 = ax1.twinx()
            ax2.plot(DOAerror, linestyle='None', marker='o', color='m', markersize=3, label='DOAError')
            ax2.set_ylabel('DOA Error', color='m')
            ax2.tick_params(axis='y', labelcolor='m')
            # Add a horizontal line for error
            ax2.axhline(y=max_error, linestyle='--', color='r', label=f'ER{max_error}')
            ax2.set_ylim(0, max(max(DOAerror), max_error)+10)
            ax2.legend(loc='upper right')
            # Add title and legend
            ax1.set_title(f'{sample_name} for the drone flight {i*max_subfig+j + 1}')
            # Add legend
            #fig.tight_layout()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/doa_result_{result_name}{i}.png")
        plt.close(fig)

def plot_localization_result_rg(result, result_name, max_error, max_subfig, output_folder, window_period, overlap_rate):
    num = len(result)
    if num == 0:
        return None
    else:
        num_fig = int((num-1)/max_subfig) + 1
    for i in range(num_fig):
        if num >= (i+1)*max_subfig:
            num_subfig = max_subfig
        else:
            num_subfig = num - i*max_subfig
        column_num = 5
        fig, ax = plt.subplots(num_subfig, column_num, figsize=(22, 4*(num_subfig))) 
        for j in range(num_subfig):
            if num_subfig == 1:
                ax1 = ax[0]
                ax2 = ax[1]
                ax3 = ax[2]
                ax4 = ax[3]
                ax5 = ax[4]
            else:
                ax1 = ax[j, 0]
                ax2 = ax[j, 1]
                ax3 = ax[j, 2]
                ax4 = ax[j, 3]
                ax5 = ax[j, 4]
            DOAerror = result[i*max_subfig+j][0]
            labels_list = np.array(result[i*max_subfig+j][1])
            #predicted_values = result[i*max_subfig+j][2].cpu()
            predicted_values = np.array([tensor.cpu() for tensor in result[i*max_subfig+j][2]])
            sample_name = result[i*max_subfig+j][3]
            time_points = np.arange(len(DOAerror)) * window_period * (1 - overlap_rate)
            ax1.plot(time_points, DOAerror, linestyle='None', marker='o', color='m', markersize=2, label=f'DOAError: {round(np.mean(DOAerror),2)} degree')
            ax1.set_ylabel('DOA Error', color='m')
            ax1.set_xlabel('Time (s)')
            ax1.tick_params(axis='y', labelcolor='m')
            # Add a horizontal line for error
            ax1.axhline(y=max_error, linestyle='--', color='r', label=f'ER{max_error}')
            ax1.set_ylim(0, max(max(DOAerror), max_error)+10)
            ax1.legend(loc='upper right')
            # Add title and legend
            ax1.set_title(f'{sample_name} for the drone flight {i*max_subfig+j + 1}')
            # Add legend
            #fig.tight_layout()
            ax2.plot(time_points, labels_list[:, 0], label='True X')
            ax2.plot(time_points, predicted_values[:, 0], label='Predicted X', linestyle='None', marker='o', markersize=1, alpha=0.5) #linestyle='dashed'
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('X Value')
            ax2.set_ylim(-1.05, 1.05)
            ax2.legend()
            ax2.set_title('X Over Time')
            ax3.plot(time_points, labels_list[:, 1], label='True Y')
            ax3.plot(time_points, predicted_values[:, 1], label='Predicted Y', linestyle='None', marker='o', markersize=1, alpha=0.5)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Y Value')
            ax3.set_ylim(-1.05, 1.05)
            ax3.legend()
            ax3.set_title('Y Over Time')
            ax4.plot(time_points, labels_list[:, 2], label='True Z')
            ax4.plot(time_points, predicted_values[:, 2], label='Predicted Z', linestyle='None', marker='o', markersize=1, alpha=0.5)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Z Value')
            ax4.set_ylim(-1.05, 1.05)
            ax4.legend()
            ax4.set_title('Z Over Time')

            true_azi = np.arctan2(labels_list[:, 1], labels_list[:, 0])  
            true_azi_deg = np.degrees(true_azi) 
            true_azi_deg = np.mod(true_azi_deg, 360)
            pred_azi = np.arctan2(predicted_values[:, 1], predicted_values[:, 0])  
            pred_azi_deg = np.degrees(pred_azi) 
            pred_azi_deg = np.mod(pred_azi_deg, 360)
            ax5.plot(time_points, true_azi_deg, label='True Azimuth (degree)')
            ax5.plot(time_points, pred_azi_deg, label='Predicted Azimuth (degree)', linestyle='--')
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Azimuth (degree)')
            ax5.set_title('True vs Predicted Azimuth')
            ax5.legend()

        plt.tight_layout()
        plt.savefig(f"{output_folder}/doa_result_{result_name}{i}.png")
        plt.close(fig)

def plot_localization_result_rg_2d(result, result_name, max_error, max_subfig, output_folder, AUDIO_DIR_TEST, window_period, overlap_rate):
    num = len(result)
    if num == 0:
        return None
    else:
        num_fig = int((num-1)/max_subfig) + 1
    for i in range(num_fig):
        if num >= (i+1)*max_subfig:
            num_subfig = max_subfig
        else:
            num_subfig = num - i*max_subfig
        column_num = 3
        fig, ax = plt.subplots(num_subfig, column_num, figsize=(12.85, 3.4*(num_subfig))) 
        for j in range(num_subfig):
            if num_subfig == 1:
                ax1 = ax[0]
                ax2 = ax[1]
                ax3 = ax[2]
            else:
                ax1 = ax[j, 0]
                ax2 = ax[j, 1]
                ax3 = ax[j, 2]
            DOAerror = result[i*max_subfig+j][0]
            labels_list = np.array(result[i*max_subfig+j][1])
            #predicted_values = result[i*max_subfig+j][2].cpu()
            predicted_values = np.array([tensor.cpu() for tensor in result[i*max_subfig+j][2]])
            sample_name = result[i*max_subfig+j][3]
            time_points = np.arange(len(DOAerror)) * window_period * (1 - overlap_rate)
            # Add legend
            #fig.tight_layout()
            ax1.plot(time_points, labels_list[:, 0], label='True X', linestyle='None', marker='o', markersize=1)
            ax1.plot(time_points, predicted_values[:, 0], label='Predicted X', linestyle='None', marker='o', markersize=1, alpha=0.5)          
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('X Value')
            ax1.set_ylim(-1.05, 1.05)
            ax1.legend()
            ax1.set_title('X over Time')
            ax2.plot(time_points, labels_list[:, 1], label='True Y', linestyle='None', marker='o', markersize=1)
            ax2.plot(time_points, predicted_values[:, 1], label='Predicted Y', linestyle='None', marker='o', markersize=1, alpha=0.5)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Y Value')
            ax2.set_ylim(-1.05, 1.05)
            ax2.legend()
            ax2.set_title('Y over Time')

            sample_name = result[i*max_subfig+j][3]
            srp_phat_csv = f'{AUDIO_DIR_TEST}/{sample_name}/{sample_name}.csv'
            DOAerror_srp = []
            true_azi = np.arctan2(labels_list[:, 1], labels_list[:, 0])  
            true_azi_deg = np.degrees(true_azi) 
            true_azi_deg = np.mod(true_azi_deg, 360)
            pred_azi = np.arctan2(predicted_values[:, 1], predicted_values[:, 0])  
            pred_azi_deg = np.degrees(pred_azi) 
            pred_azi_deg = np.mod(pred_azi_deg, 360)
            ax3.plot(time_points, true_azi_deg, label='True Azimuth', color="black", linestyle="-", linewidth=1.5) #linestyle='None', marker='o', markersize=1)
            ax3.plot(time_points, pred_azi_deg, label=f'Neural SRP', linestyle="None", marker='o', markersize=1, alpha=0.3) #linestyle='None', marker='o', markersize=1, alpha=0.3), color="blue"， (DOA error {round(np.mean(DOAerror),2)}°)
            if os.path.isfile(srp_phat_csv):
                with open(srp_phat_csv, 'r', newline='') as file:
                        doa_est_srp = csv.reader(file)
                        doa_est_srp = list(doa_est_srp)[10:]
                        doa_est_srp = [list(map(float, row)) for row in doa_est_srp] 
                        azi_est_srp = [row[5] for row in doa_est_srp] 
                for k in range(min(len(doa_est_srp), len(labels_list))):
                    error = angle_between_vectors(np.array(doa_est_srp[k][1:3]), np.array(labels_list[k, :2]))
                    DOAerror_srp.append(error)
                ax3.plot(time_points, azi_est_srp[:len(time_points)], label=f'SRP-PHAT', linestyle="None", marker='*', markersize=2.5, alpha=0.7) #linestyle='None', marker='o', markersize=2, alpha=0.3), color="red"， (DOA error {round(np.mean(np.array(DOAerror_srp)),2)}°)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Azimuth (°)')
            ax3.set_title(f'DOA Estimation: True vs Predicted Azimuth')
            ax3.legend()

        plt.tight_layout()
        plt.savefig(f"{output_folder}/doa_result_{result_name}{i}.png")
        plt.close(fig)

def plot_3d_scatter(ax, labels, title):
    x, y, z = labels[:, 0], labels[:, 1], labels[:, 2]
    ax.scatter(x, y, z, marker='o', c='r', s=0.8, alpha=0.15) #c='r'
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_zlim([z.min() - 0.1, z.max() + 0.1])
    ax.set_title(title)

def plot_2d_azi_ele(ax, labels, title):
    azimuth, elevation = xyz_to_azi_ele(labels)
    ax.scatter(azimuth, elevation, c='b', s=0.8, alpha=0.15) #c='b'
    ax.set_xlabel('Azimuth')
    ax.set_ylabel('Elevation')
    ax.set_title(title)

def xyz_to_azi_ele(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    azimuth = np.degrees(np.arctan2(y, x))
    azimuth[azimuth < 0] += 360
    elevation = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z))
    return azimuth, elevation


def angle_between_vectors(vector1, vector2):
    """
    Calculate the angle (in degrees) between two vectors.

    Parameters:
        vector1: The first vector in numpy array format.
        vector2: The second vector in numpy array format.

    Returns:
        The angle (in degrees) between the two vectors.
    """
    # Calculate the dot product of the vectors
    dot_product = np.dot(np.array(vector1), np.array(vector2))
    # Calculate the lengths of the vectors
    length_vector1 = np.linalg.norm(vector1)
    length_vector2 = np.linalg.norm(vector2)
    # Calculate the cosine of the angle
    cosine_angle = dot_product / (length_vector1 * length_vector2)
    # Calculate the angle in radians
    if cosine_angle>=1:
        cosine_angle = 1
    if cosine_angle<=-1:
        cosine_angle = -1
    angle_radians = np.arccos(cosine_angle)
    # Convert radians to degrees
    angle_degree = np.degrees(angle_radians)

    return angle_degree