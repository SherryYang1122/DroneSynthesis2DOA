from torch.utils.data import Dataset
import torchaudio
import numpy as np
import os
import torch
import librosa
import csv


half_elevation = 90
resampling_rate = 8000
cutoff_frequency = 5000
max_difference_samples = 36 

class DOADatasetMulti(Dataset):

    def __init__(self, data_folder, preprocessing_method, input_feature, hyperparams, device):
        # Get the current working directory
        abspath = os.path.dirname(os.path.abspath(__file__))
        self.data_folder = os.path.join(abspath, data_folder)
        self.items = [f for f in os.listdir(self.data_folder) if not f.startswith('.DS_Store')]

        self.window_period = hyperparams['frame_length']
        self.overlap_rate = hyperparams['frame_length'] - hyperparams['frame_hop']
        self.device = device
        self.preprocessing_method = preprocessing_method
        self.input_feature = input_feature
        self.class_mapping = []            

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        doa_real = self._get_flight_path(index)
        doa_resolution = doa_real[1][0] - doa_real[0][0]
        signal_orig, fs = torchaudio.load(audio_sample_path)
        if self.preprocessing_method == "resampling":
            signal = librosa.resample(signal_orig.numpy(), orig_sr=fs, target_sr=resampling_rate)
            signal = torch.from_numpy(signal).to(device=self.device, dtype=torch.float32)
            sampling_rate = resampling_rate
        else:
            signal = signal_orig.to(self.device)
            sampling_rate = fs

        drone_angles = []
        window_size = int(sampling_rate * self.window_period)  
        overlap = int(window_size * self.overlap_rate)
        bin_num = int(window_size/2)+1
        # Overlap
        num_frames = int((signal.shape[1] - window_size) / (window_size - overlap)) + 1
        start_index = 10 
        if self.preprocessing_method == "low_pass":
            bin_num = int(bin_num * cutoff_frequency * 2/sampling_rate)

        if self.input_feature == "GCC_PHAT" or self.input_feature =='GCC_PHAT_beta' or self.input_feature == "GCC_PHAT_mask" or self.input_feature == "GCC_PHAT_beta_mask":
            sig_out = torch.zeros(num_frames-start_index, 1, int(signal.shape[0]*(signal.shape[0]-1)/2), max_difference_samples*2, device=self.device)
        elif self.input_feature == "GCC_PHAT_MC":
            sig_out = torch.zeros(num_frames-start_index, int(signal.shape[0]), int(signal.shape[0]), max_difference_samples*2, device=self.device)
        elif self.input_feature == "phase_mag":
            sig_out = torch.zeros(num_frames-start_index, 1, signal.shape[0]*2, bin_num, device=self.device)
        else:
            sig_out = torch.zeros(num_frames-start_index, 1, signal.shape[0], bin_num, device=self.device)
        labels = torch.zeros(num_frames-start_index, 3, device=self.device)
        real_pos = []
        
        freqs = np.fft.fftfreq(window_size, d=1/sampling_rate)  # Generate frequency bins
        # Band-pass 
        low_cut = 250
        high_cut = 7000
        # Create a mask for desired frequency range
        mask = (np.abs(freqs) >= low_cut) & (np.abs(freqs) <= high_cut)
        mask = torch.from_numpy(mask).to(self.device) 
        for i in range(num_frames-start_index):
            start_idx = (i+start_index) * (window_size - overlap)
            end_idx = start_idx + window_size
            x = signal[:, start_idx:end_idx].to(self.device)
            if self.input_feature == "GCC_PHAT" or self.input_feature =='GCC_PHAT_beta' or self.input_feature =='GCC_PHAT_mask' or self.input_feature == "GCC_PHAT_beta_mask":
                x = torch.fft.fft(x, dim=-1)
                smoothing = 1e-10
                cc = []
                for j in range(x.shape[0]):
                    for k in range(j + 1, x.shape[0]):
                        # Cross power spectrum
                        cross_correlation = x[k] * torch.conj(x[j])
                        # Prevent division by zero and add a smoothing term.
                        if self.input_feature =='GCC_PHAT_beta' or self.input_feature == "GCC_PHAT_beta_mask":
                            beta = 0.7
                            gcc_phat_spectrum = cross_correlation / (torch.abs(cross_correlation) ** beta + smoothing)
                        else:
                            gcc_phat_spectrum = cross_correlation / (torch.abs(cross_correlation) + smoothing)
                        if self.input_feature == "GCC_PHAT_mask" or self.input_feature == "GCC_PHAT_beta_mask":
                            gcc_phat_spectrum = gcc_phat_spectrum * mask
                                
                        m_gcc_phat = torch.fft.ifft(gcc_phat_spectrum)
                        m_gcc_phat = torch.fft.fftshift(m_gcc_phat)
                        cc.append(m_gcc_phat[bin_num-max_difference_samples:bin_num+max_difference_samples])
                cc = torch.stack(cc)
                cc = torch.nn.functional.normalize(cc)
                sig_out[i, 0, :, :] = cc
            elif self.input_feature == "GCC_PHAT_MC":
                sig_out[i] = self.calculate_uninformed(x, max_difference_samples)
            elif self.input_feature == "phase":
                x = torch.fft.fft(x, dim=-1)
                x = x[:, 0:int(window_size/2+1)]
                x = x.angle()
                x = torch.tensor(np.unwrap(x.cpu().numpy(), axis=-1), device=self.device)
                x = torch.nn.functional.normalize(x)
                sig_out[i, 0, :, :] = x[:,:bin_num]
            elif self.input_feature == "phase_mag":
                x = torch.fft.fft(x, dim=-1)
                x = x[:, 0:int(window_size/2+1)]
                magnitude = torch.abs(x)
                magnitude = torch.nn.functional.normalize(magnitude)
                phase = torch.angle(x)
                phase = torch.tensor(np.unwrap(phase.cpu().numpy(), axis=-1), device=self.device)
                phase = torch.nn.functional.normalize(phase)
                sig_out[i, 0, :, :] = torch.cat((magnitude, phase), dim=0)
            elif self.input_feature == "none":
                x = torch.abs(x)
                x = torch.nn.functional.normalize(x)
                sig_out[i, 0, :, :] = x[:,:bin_num]
            dir_angle = np.array(doa_real[int((i+start_index)*(window_size - overlap)/(doa_resolution*sampling_rate))][4:6])
            drone_angles.append(dir_angle)
            real_pos.append(np.array(doa_real[int((i+start_index)*(window_size - overlap)/(doa_resolution*sampling_rate))][1:4]))
        sig_out = sig_out.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        return sig_out, audio_sample_path.split(".")[0], real_pos


    def _get_audio_sample_path(self, index):
        filename = self.items[index]
        return os.path.join(self.data_folder, filename+"/"+filename+".wav")


    def _get_flight_path(self, index):
        filename = self.items[index]
        doa_real_sample = os.path.join(self.data_folder, filename+"/"+"DOATruePos.csv")
        with open(doa_real_sample, 'r', newline='') as file:
            doa_real = csv.reader(file)
            doa_real = list(doa_real)[1:]
            doa_real = [list(map(float, row)) for row in doa_real] 
        return doa_real

    def get_num_classes(self):
        return self.num_classes

    def get_class_mapping(self):
        return self.class_mapping
    
    def get_fib_dirs(self):
        return self.fib_dirs
    
    def calculate_uninformed(self, frame, max_difference_samples):
        num_channels = frame.shape[0]
        frame_length = frame.shape[1]
        principal_length = frame.shape[1]
        unity_magnitudes = torch.ones(size=(int(principal_length / 2 + 1),), device=self.device)
        window = torch.hann_window(frame_length, device=self.device)
        frame = torch.mul(frame, window)
        
        x_spec = torch.fft.rfft(frame, dim=-1)
        x_angle = x_spec.angle()
        m_cross_phase_weighted = torch.zeros(num_channels, num_channels, int(principal_length / 2 + 1),
                                             device=self.device,
                                             dtype=torch.complex64)
        # now with magnitude weighting by direct signal
        for ii in range(num_channels):
            for jj in range(num_channels):
                # Subtract in order to account for conjugation
                gcc_phase = x_angle[ii, :] - x_angle[jj, :]
                m_cross_phase_weighted[ii, jj, :] = torch.polar(unity_magnitudes, gcc_phase)

        m_gcc_phat_weighted = torch.fft.irfft(m_cross_phase_weighted, dim=-1).real
        m_gcc_phat_weighted = torch.fft.fftshift(m_gcc_phat_weighted, dim=-1)
        m_gcc_phat_weighted = m_gcc_phat_weighted[:, :, int(principal_length / 2 - max_difference_samples):
                                                        int(principal_length / 2 + max_difference_samples)]
        return m_gcc_phat_weighted
    
def cart2sph(vector):
    # Calculate elevation angle, vector = [x,y,z]
    elevation_angle = np.degrees(np.arctan2(np.sqrt(vector[0]**2 + vector[1]**2), vector[2]))
    # Calculate azimuth angle
    azimuth_angle = np.degrees(np.arctan2(vector[1], vector[0]))
    azimuth_angle = (azimuth_angle + 360) if azimuth_angle<0 else azimuth_angle
    return [elevation_angle, azimuth_angle]

# Convert azimuth and elevation angles to Cartesian coordinates
def sph2cart(elevation_angle, azimuth_angle):
    elevation = np.radians(elevation_angle)
    azimuth = np.radians(azimuth_angle)
    x = np.sin(elevation) * np.cos(azimuth)
    y = np.sin(elevation) * np.sin(azimuth)
    z = np.cos(elevation)
    return x, y, z

def fibonacci_sphere(n):  
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:,0] = radius * np.cos(theta)
    points[:,1] = radius * np.sin(theta)
    points[:,2] = z
    return points

def half_fibonacci_sphere(n):
    full_sphere_points = fibonacci_sphere(2 * n)
    upper_hemisphere_points = [(x, y, z) for x, y, z in full_sphere_points if z >= 0]
    return np.array(upper_hemisphere_points[:n])

def find_nearest_direction(dir_angle, directions):
    min_distance = -100
    nearest_direction = None
    target_unit_dir = sph2cart(dir_angle[0], dir_angle[1])
    for i, direction in enumerate(directions):
        dot_product = sum(a * b for a, b in zip(target_unit_dir, direction))
        if dot_product > min_distance:
            min_distance = dot_product
            nearest_direction = direction
            label = i
    return label, nearest_direction 