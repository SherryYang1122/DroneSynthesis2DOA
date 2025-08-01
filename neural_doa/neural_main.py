# neural_main.py
# This code provides the main training and testing pipeline for neural DOA estimation models.
# (c) 2025, X. Yang, Fraunhofer IDMT, Germany, MIT License
# version 1.0, August 2025

import numpy as np
import torch
import argparse
import os
from torch import nn
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torch.utils.data
import warnings
from torch.utils.tensorboard import SummaryWriter
import json
import shutil
import pickle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import yaml

from seld_net_accdoa import SELD_net
from DOADatasetMulti import DOADatasetMulti
import train
from neural_srp import NeuralSrp
from visualization import plot_doa_error_vs_distance, plot_doa_error_vs_distance_real_snr, plot_localization_result_rg, plot_localization_result_rg_2d, plot_3d_scatter, plot_2d_azi_ele, angle_between_vectors

BATCH_SIZE = 1024 
num_workers = 0
EPOCHS = 300 
LEARNING_RATE = 0.001 
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
points_num = 500 # Number of points in half fibonacci sphere for classical SRP-PHAT
elevation = 90 # half shphere
azimuth_degree = 360
n_fibonacci = 100


def predict_rg(model, input, mic_pairs=None): # regression task 
    model.eval()
    hidden_state = None
    if mic_pairs == None:
        with torch.no_grad():
            output, _ = model(input, lengths=torch.tensor([input.shape[0]]), hidden_state = hidden_state)
            output = output[0]
    else:
        with torch.no_grad():
            output = []
            output, _ = model(input, mic_pairs.to(torch.float32), lengths=torch.tensor([input.shape[0]]), hidden_state=hidden_state) 
            output = output[0]    
    return output

# padding for the sequences
def collate_fn(batch):
    sequences, targets = zip(*batch)
    seq_lengths = torch.tensor([len(seq) for seq in sequences])
    sequences = pad_sequence(sequences, batch_first=True)
    targets = pad_sequence(targets, batch_first=True)
    return sequences, targets, seq_lengths

def collate_fn_srp(batch):
    sequences, mic_pairs, targets = zip(*batch)
    seq_lengths = torch.tensor([len(seq) for seq in sequences])
    sequences = pad_sequence(sequences, batch_first=True)
    targets = pad_sequence(targets, batch_first=True)
    return sequences, mic_pairs, targets, seq_lengths


def print_total_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")


def extract_labels(dataset):
    labels = []
    for item in dataset:
        labels.append(item[-1])
    labels = torch.cat(labels, dim=0)
    return labels

class SimpleDataset(Dataset):
    def __init__(self, result, labels):
        self.result = result
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.result[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class SRP_Dataset(Dataset):
    def __init__(self, result, mic_pairs, labels):
        self.result = result
        self.labels = labels
        if isinstance(mic_pairs, (list, tuple)):
            mic_pairs = np.array(mic_pairs)
        self.mic_pairs = torch.tensor(mic_pairs, dtype=torch.float32) 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.result[idx], dtype=torch.float32), self.mic_pairs, torch.tensor(self.labels[idx], dtype=torch.float32)


def process_mic_positions(json_file):
    """
    Reads microphone configuration from a JSON file, extracts the microphone coordinates, 
    and generates pairs of microphone coordinates.

    Parameters:
        json_file (str): Path to the JSON file containing microphone configurations.

    Returns:
        mic_pairs (numpy.ndarray): num_pairs x 6 array of concatenated coordinate pairs.
    """
    # 1. Read the JSON file
    with open(json_file, 'r') as f:
        mic_data = json.load(f)

    # 2. Extract microphone positions into an N x 3 matrix
    mic_positions = mic_data[0]["mic_pos"]
    mic_positions = np.array(mic_positions)  # Convert to N x 3 numpy array
    num = mic_positions.shape[0]

    # 3. Generate concatenated coordinates for each pair of microphones
    mic_pairs = []
    for j in range(num):
        for k in range(j + 1, num):
            # Retrieve coordinates of mic j and mic k
            mic1 = mic_positions[j]
            mic2 = mic_positions[k]
            # Concatenate as [x1, y1, z1, x2, y2, z2]
            mic_pairs.append(np.concatenate((mic1, mic2)))

    mic_pairs = np.array(mic_pairs)  # Convert to num_pairs x 6 array

    return mic_pairs

def split_data(features, labels, max_time_steps, audio_name_list=None, real_pos_list=None):
    new_features = []
    new_labels = []
    new_audio_names = [] if audio_name_list is not None else None
    new_real_pos = [] if real_pos_list is not None else None

    for i in range(len(features)):
        feature = features[i]
        label = labels[i]
        time_steps = len(feature)  # Get the current feature's time steps

        # Get audio name and position if lists are provided
        audio_name = audio_name_list[i] if audio_name_list is not None else None
        real_pos = real_pos_list[i] if real_pos_list is not None else None

        # Calculate number of segments needed (ceiling division)
        num_segments = (time_steps + max_time_steps -1) // max_time_steps

        for segment_idx in range(num_segments):
            start = segment_idx * max_time_steps
            end = min(start + max_time_steps, time_steps)  # End position, ensuring it doesn’t exceed original length
            
            # Slice feature and label according to max_time_steps
            new_feature_segment = feature[start:end]
            new_label_segment = label[start:end]
            new_features.append(new_feature_segment)
            new_labels.append(new_label_segment)

            # If audio_name_list and real_pos_list exist, handle slicing
            if new_audio_names is not None:
                new_audio_names.append(f"{audio_name}_{segment_idx + 1}")  # Append segment ID to audio name
            if new_real_pos is not None:
                new_real_pos.append(real_pos[start:end])  # Slice real position list to match feature segment

    return new_features, new_labels, new_audio_names, new_real_pos


def calculate_frame_stats(labels, window_period = 0.032, overlap_rate = 0.0):
    frame_counts = [len(sample) for sample in labels]  

    total_frames = sum(frame_counts)
    total_flight_time = total_frames*(window_period-overlap_rate) / 60
    min_dur = round(min(frame_counts)*(window_period-overlap_rate),2)
    max_dur = round(max(frame_counts)*(window_period-overlap_rate),2)
    print("total_frames:", total_frames, "total_flight_time (min):", round(total_flight_time,2))
    print(f"one flight time is between {min_dur} and {max_dur} seconds")



def main():
    # Create a directory to store data if it doesn't exist
    exp = args.exp
    preprocessing_method = args.preprocessing_method
    input_feature = args.input_feature
    nn_model = args.nn_model

    output_folder = f"{result_dir}/DOAOutput{nn_model}"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/{args.data_file}_{args.input_feature}", exist_ok=True)
    os.makedirs(f"{output_folder}/{args.data_file_test}_{args.input_feature}", exist_ok=True)
    data_file = args.data_file
    data_file_test = args.data_file_test
    AUDIO_DIR_TRAIN = f"{result_dir}/{data_file}"
    AUDIO_DIR_TEST = f"{result_dir}/{data_file_test}"

    is_training = args.is_training #True
    is_continue = False

    if nn_model == "neural_srp":
        max_time_steps = 250 #500 
        BATCH_SIZE_SAMPLE = 8 #8 
    else:
        max_time_steps = 100000 
        BATCH_SIZE_SAMPLE = 32
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")


    if is_training:
        prep_data_folder = f"{result_dir}/preprocessing_data/{data_file}"
        os.makedirs(prep_data_folder, exist_ok=True)
        result_path = f"{prep_data_folder}/input_{preprocessing_method}_{input_feature}.pkl"
        if nn_model == "SELD_ACCDOA" or nn_model == "neural_srp":
            labels_path = f"{prep_data_folder}/real_pos.pkl"
            
        if os.path.exists(result_path) and os.path.exists(labels_path):
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            with open(labels_path, 'rb') as f:
                labels = pickle.load(f)
        else:
            dataset = DOADatasetMulti(AUDIO_DIR_TRAIN, preprocessing_method, input_feature, hyperparams, device=device)
            sig_out_list = []
            audio_sample_list = []
            real_pos_list = []
            for sample in dataset:
                sig_out_list.append(sample[0].cpu().numpy())
                audio_sample_list.append(sample[1])
                real_pos_list.append(np.array(sample[2]))
            with open(f"{prep_data_folder}/input_{preprocessing_method}_{input_feature}.pkl", 'wb') as f:
                pickle.dump(sig_out_list, f)
            with open(f"{prep_data_folder}/audio_name.pkl", 'wb') as f:
                pickle.dump(audio_sample_list, f)
            with open(f"{prep_data_folder}/real_pos.pkl", 'wb') as f:
                pickle.dump(real_pos_list, f)
            if nn_model == "SELD_ACCDOA" or nn_model == "neural_srp":
                labels = real_pos_list
            result = sig_out_list

        calculate_frame_stats(labels, window_period, overlap_rate)
        result, labels, _, _ = split_data(result, labels, max_time_steps)

        if nn_model == "SELD_ACCDOA" or nn_model == "neural_srp":
            # change xyz to unit vector
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    vector = np.array(labels[i][j])
                    norm = np.linalg.norm(vector)
                    if norm != 0:
                        labels[i][j] = vector / norm
                    else:
                        labels[i][j] = vector 
            input_shape = result[0].shape
            if nn_model == "neural_srp":
                json_file = f"{result_dir}/mic_config.json"
                mic_pairs = process_mic_positions(json_file)
                dataset =  SRP_Dataset(result, mic_pairs, labels)
            else:
                dataset =  SimpleDataset(result, labels)

            n_total = len(dataset)
            n_train = int(n_total * TRAIN_RATIO/(TRAIN_RATIO+VAL_RATIO))
            n_val = n_total - n_train
            train_data = Subset(dataset, list(range(n_train)))
            val_data = Subset(dataset, list(range(n_train, n_train + n_val)))
            print("number of training samples: ", n_train)
            print("number of validation samples: ", n_val)
            # visualization 
            train_labels = extract_labels(train_data)
            val_labels = extract_labels(val_data)
            # train data
            fig = plt.figure(figsize=(14, 10))
            ax1 = fig.add_subplot(221, projection='3d')
            plot_3d_scatter(ax1, train_labels, "Train Data 3D Scatter Plot")
            ax2 = fig.add_subplot(222)
            plot_2d_azi_ele(ax2, train_labels, "Train Data Azimuth and Elevation")
            ax3 = fig.add_subplot(223, projection='3d')
            plot_3d_scatter(ax3, val_labels, "Validation Data 3D Scatter Plot")
            ax4 = fig.add_subplot(224)
            plot_2d_azi_ele(ax4, val_labels, "Validation Data Azimuth and Elevation")
            # save
            plt.tight_layout()
            plt.savefig(f"{output_folder}/{args.data_file}_{args.input_feature}/train_validation_data_visualization.png")
        
        # creating cnn and pushing it to CPU/GPU(s)
        if nn_model == "SELD_ACCDOA":
            cnn = SELD_net(
                input_shape = input_shape,
                n_conv_filters=64,
                pooling_size=2,
                conv_dropout=0.2,
                rnn_type="gru",
                num_rnn_layers=2,
                hidden_size=128,
                rnn_dropout = 0.0,
                bidirectional=False, #True,
                linear_dim=128,
                fc_dropout=0.0,
                sequence_length=8,
                n_doa_dims=3,
                n_classes=1,
                n_tracks=1)
        elif nn_model == "neural_srp":
            with open('params_srp.json') as json_file:
                params_srp = json.load(json_file)
            cnn = NeuralSrp(input_shape[-1], params_srp["neural_srp"], n_max_dataset_sources=1)


        if nn_model == "SELD_ACCDOA":
            train_data = sorted(train_data, key=lambda x: x[0].shape[0], reverse=True)
            train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE_SAMPLE, shuffle=False, collate_fn=collate_fn)
            val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE_SAMPLE, shuffle=False, collate_fn=collate_fn)
        elif nn_model == "neural_srp":
            train_data = sorted(train_data, key=lambda x: x[0].shape[0], reverse=True)
            train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE_SAMPLE, shuffle=False, collate_fn=collate_fn_srp)
            val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE_SAMPLE, shuffle=False, collate_fn=collate_fn_srp)

        writer = SummaryWriter()
        if nn_model == "SELD_ACCDOA":
            loss_fn = nn.MSELoss()
        elif  nn_model == "neural_srp":
            loss_fn = nn.MSELoss()
           
        if is_continue:
            sd = torch.load(f"{output_folder}/{nn_model}.pth")
            cnn.load_state_dict(sd)
        cnn.to(device)
        print_total_parameters(cnn)

        optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
        # train model
        loss_list, val_list = train.train(cnn, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS, writer)

        # save model
        torch.save(cnn.state_dict(), f"{output_folder}/{args.data_file}_{args.input_feature}/{nn_model}.pth")
        print(f"Trained feed forward net using {data_file} dataset saved at {nn_model}.pth")
        writer.close()
        torch.save(loss_list, f"{output_folder}/{args.data_file}_{args.input_feature}/loss_list.pt")
        torch.save(val_list, f"{output_folder}/{args.data_file}_{args.input_feature}/val_list.pt")
  

    # -------------------------------------------------------------------------------------------
    # Always test
    prep_data_folder = f"{result_dir}/preprocessing_data/{data_file_test}"
    os.makedirs(prep_data_folder, exist_ok=True)
    sig_path = f"{prep_data_folder}/input_{preprocessing_method}_{input_feature}.pkl"
    audio_name_path = f"{prep_data_folder}/audio_name.pkl"
    real_pos_path = f"{prep_data_folder}/real_pos.pkl"
    if nn_model == "SELD_ACCDOA" or nn_model == "neural_srp":
        labels_path = f"{prep_data_folder}/real_pos.pkl"


    if os.path.exists(sig_path) and os.path.exists(labels_path) and os.path.exists(audio_name_path) and os.path.exists(real_pos_path):
        with open(sig_path, 'rb') as f:
            sig_out_list = pickle.load(f)
        with open(labels_path, 'rb') as f:
            labels_list = pickle.load(f)
        with open(audio_name_path, 'rb') as f:
            audio_name_list = pickle.load(f)
        with open(real_pos_path, 'rb') as f:
            real_pos_list = pickle.load(f)
    else:
        dataset = DOADatasetMulti(AUDIO_DIR_TEST, preprocessing_method, input_feature, hyperparams, device=device)
        sig_out_list = []
        audio_name_list = []
        real_pos_list = []
        for sample in dataset:
            sig_out_list.append(sample[0].cpu().numpy())
            audio_name_list.append(sample[1])
            real_pos_list.append(np.array(sample[2]))
        with open(f"{prep_data_folder}/input_{preprocessing_method}_{input_feature}.pkl", 'wb') as f:
            pickle.dump(sig_out_list, f)
        with open(f"{prep_data_folder}/audio_name.pkl", 'wb') as f:
            pickle.dump(audio_name_list, f)
        with open(f"{prep_data_folder}/real_pos.pkl", 'wb') as f:
            pickle.dump(real_pos_list, f)

    if nn_model == "SELD_ACCDOA" or nn_model == "neural_srp":
        labels_list = [[] for i in range(len(real_pos_list))]
        # change xyz to unit vector
        for i in range(len(real_pos_list)):
            for j in range(len(real_pos_list[i])):
                vector = np.array(real_pos_list[i][j], dtype=np.float64)
                norm = np.linalg.norm(vector)
                if norm != 0:
                    labels_list[i].append(vector / norm)
                else:
                    labels_list[i].append(vector)
        
    # label distribution 
    if nn_model == "SELD_ACCDOA" or nn_model == "neural_srp":
        input_shape = sig_out_list[0].shape
        test_labels = []
        for label in labels_list:
            test_labels.append(torch.tensor(label))
        test_labels = torch.cat(test_labels, dim=0)
        # visualization 
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        plot_3d_scatter(ax1, test_labels, "Test Data 3D Scatter Plot")
        ax2 = fig.add_subplot(122)
        plot_2d_azi_ele(ax2, test_labels, "Test Data Azimuth and Elevation")
        # save
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{args.data_file_test}_{args.input_feature}/test_data_visualization.png")
        plt.clf()
    calculate_frame_stats(labels_list, window_period, overlap_rate)

    if nn_model == "SELD_ACCDOA":
        cnn = SELD_net(
            input_shape = input_shape,
            n_conv_filters=64,
            pooling_size=2,
            conv_dropout=0.2,
            rnn_type="gru",
            num_rnn_layers=2,
            hidden_size=128,
            rnn_dropout = 0.0,
            bidirectional=False, #True,
            linear_dim=128,
            fc_dropout=0.0,
            sequence_length=8,
            n_doa_dims=3,
            n_classes=1,
            n_tracks=1)
    elif nn_model == "neural_srp":
        with open('params_srp.json') as json_file:
            params_srp = json.load(json_file)
        cnn = NeuralSrp(input_shape[-1], params_srp["neural_srp"], n_max_dataset_sources=1)
     
    sd = torch.load(f"{output_folder}/{args.data_file}_{args.input_feature}/{nn_model}.pth", map_location=torch.device(device))
    cnn.load_state_dict(sd)
    cnn.to(device)
    

    n_true20 = 0
    max_error = 20

    loss_list = torch.load(f"{output_folder}/{args.data_file}_{args.input_feature}/loss_list.pt")
    val_list = torch.load(f"{output_folder}/{args.data_file}_{args.input_feature}/val_list.pt")

    frame_num_total = 0
    doa_errors = []
    doa_errors_sum = []
    ACC20 = []
    distances = []
    snr_values = []
    real_bandpass_snr_values = []
    real_snr_values = []
    good_result = []
    bad_result = []
    if nn_model == "neural_srp":
        json_file = f"{result_dir}/mic_config.json"
        mic_pairs = process_mic_positions(json_file)
    for input, labels, audio_sample_path, real_pos in zip(sig_out_list, labels_list, audio_name_list, real_pos_list):
        sample_name = audio_sample_path.split('/')[-1]
        if nn_model == "SELD_ACCDOA":
            predicted = predict_rg(cnn, torch.tensor(input).to(device))
        elif nn_model == "neural_srp":
            predicted = predict_rg(cnn, torch.tensor(input).to(device), torch.tensor(mic_pairs).to(device))

        frame_num_total += len(predicted)
        DOAerror = []
        dis = []
        n_true20_each = 0
        for index, pred in enumerate(predicted):
            pos = real_pos[index]
            if nn_model == "SELD_ACCDOA" or nn_model == "neural_srp":
                if args.only_azi:
                    angle_error = angle_between_vectors(pred.cpu()[:2], pos[:2])
                else:
                    angle_error = angle_between_vectors(pred.cpu(), pos)
            if angle_error <= max_error:
                n_true20 += 1
                n_true20_each += 1
            DOAerror.append(angle_error)
            dis.append(np.linalg.norm(pos))

        # read drone metadata
        try:
            with open(os.path.join(os.path.join(AUDIO_DIR_TEST, sample_name), "metadata.json"), 'r') as file:
                drone_data = json.load(file)[0]
                if "SNR" in drone_data:
                    snr_values.append(drone_data["SNR"])
                if "real_bandpass_SNR" in drone_data:
                    real_bandpass_snr_values.append(drone_data["real_bandpass_SNR"])
                if "real_SNR" in drone_data:
                    real_snr_values.append(drone_data["real_SNR"])
        except FileNotFoundError:
            pass
        doa_errors.append(np.mean(np.array(DOAerror)))
        doa_errors_sum.append(np.sum(np.array(DOAerror)))
        distances.append(np.mean(np.array(dis)))
        ACC20.append(n_true20_each/len(predicted))
        
        if nn_model == "SELD_ACCDOA" or nn_model == "neural_srp":
            if np.mean(np.array(DOAerror)) <= max_error:
                good_result.append([DOAerror, labels, predicted, sample_name])
            else:
                bad_result.append([DOAerror, labels, predicted, sample_name])   

    is_training = False
    ACC20_weighted = n_true20 / frame_num_total 
    ACC20_avg = np.mean(np.array(ACC20))
    DOAerror_avg = np.mean(np.array(doa_errors))
    DOAerror_weighted = np.sum(np.array(doa_errors_sum))/frame_num_total
    if not is_training:
        plt.figure()
        line_train = plt.plot(loss_list, 'b', label='training')
        line_val = plt.plot(val_list, 'r', label='validation')
        if abs(loss_list[-1]) < 1e-3:
            loss_final =  f'{loss_list[-1]:.4e}' 
            val_final =  f'{val_list[-1]:.4e}' 
        else:
            loss_final = f'{loss_list[-1]:.4f}'  
            val_final =  f'{val_list[-1]:.4f}'
        plt.annotate(loss_final, xy=(len(loss_list)-1, loss_list[-1]), xytext=(len(loss_list)-1, loss_list[-1] + 0.0001),
                     fontsize=10, ha='right', va='bottom') 
        plt.annotate(val_final, xy=(len(val_list)-1, val_list[-1]), xytext=(len(val_list)-1, val_list[-1] + 0.0001),
                     fontsize=10, ha='left', va='bottom')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if nn_model == "SELD_ACCDOA" or nn_model == "neural_srp":
            plt.title(f"{nn_model}, \n DOA Error:{DOAerror_avg:.2f} degrees, ACC{max_error}:{ACC20_avg*100:.2f}%; (weighted, {DOAerror_weighted:.2f}/{ACC20_weighted*100:.2f}%)")
            plt.savefig(f"{output_folder}/{args.data_file_test}_{args.input_feature}/CRNN_{input_feature}.png")            
        plt.clf()

    max_subfig = 200
    output_folder_test = f"{output_folder}/{args.data_file_test}_{args.input_feature}/TestDOAResult"
    if os.path.exists(output_folder_test):
        shutil.rmtree(output_folder_test)
    os.makedirs(output_folder_test, exist_ok=True)
    if nn_model == "SELD_ACCDOA" or nn_model == "neural_srp":
        max_subfig = int(max_subfig/4)
        if args.only_azi:
            plot_localization_result_rg_2d(good_result, "good", max_error, max_subfig, output_folder_test, AUDIO_DIR_TEST, window_period, overlap_rate)
            plot_localization_result_rg_2d(bad_result, "bad", max_error, max_subfig, output_folder_test,  AUDIO_DIR_TEST, window_period, overlap_rate)
        else:
            plot_localization_result_rg(good_result, "good", max_error, max_subfig, output_folder_test, window_period, overlap_rate)
            plot_localization_result_rg(bad_result, "bad", max_error, max_subfig, output_folder_test, window_period, overlap_rate)
    

    print(f"testing using {nn_model}, {data_file_test} dataset with {input_feature}")
    print(f"DL DOA Error: {DOAerror_avg} degrees")
    print(f"DL ACC{max_error}: {ACC20_avg * 100}%")
    print(f"DL ERR{max_error}: {(1-ACC20_avg) * 100}%")
    print(f"DL DOA Error(weighted): {DOAerror_weighted} degrees")
    print(f"DL ACC{max_error} (weighted): {ACC20_weighted * 100}%")
    plot_doa_error_vs_distance(distances, doa_errors, snr_values, f"{output_folder}/{args.data_file_test}_{args.input_feature}")
    plot_doa_error_vs_distance_real_snr(distances, doa_errors, real_snr_values, f"{output_folder}/{args.data_file_test}_{args.input_feature}") 
    snr_to_doa_errors = {}
    for doa_error, snr_value in zip(doa_errors, snr_values):
        if snr_value not in snr_to_doa_errors:
            snr_to_doa_errors[snr_value] = []
        snr_to_doa_errors[snr_value].append(doa_error)
    snr_to_mean_doa_error = {snr: np.mean(errors) for snr, errors in snr_to_doa_errors.items()}
    for snr, mean_doa_error in snr_to_mean_doa_error.items():
        print(f"SNR: {snr}, Mean DOA Error: {mean_doa_error:.2f} degrees")
    print("total number of test labels: ", frame_num_total)
    print("total flight time (minutes): ", round(frame_num_total*(window_period-overlap_rate)/60,2))

    print("done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drone Localization Neural Network')
    # Add argument for the path to the training and testing data file
    parser.add_argument('--exp', type=str, default='exp_test', help='Path to the dataset file')
    parser.add_argument('--data_file', type=str, default='MicArrayData', help='Folder to the training dataset file')
    parser.add_argument('--data_file_test', type=str, default='MicArrayDataTest', help='Folder to the testing dataset file')
    # Add argument for the output method
    parser.add_argument('--preprocessing_method', choices=['low_pass', 'resampling', 'none'], default='none',
                        help='Input preprocessing method: high_pass, resampling, none')
    parser.add_argument('--input_feature', type=str, choices=['phase', 'GCC_PHAT', 'GCC_PHAT_beta', 'GCC_PHAT_MC', 'GCC_PHAT_mask', 'none', 'phase_mag', 'GCC_PHAT_beta_mask'], default='phase',
                        help='Input feature to use. Choices are: phase, GCC_Phat, none. Default is phase.')
    parser.add_argument('--nn_model', type=str, choices=['SELD_ACCDOA', 'neural_srp'], default='SELD_ACCDOA',
                        help='Choose the neural network model')
    parser.add_argument('--only_azi', action='store_true', help='Only consider azi degree during test, ignore ele degree')
    parser.add_argument('--is_training', action='store_true', help='Enable training mode.')

    args = parser.parse_args()
    
    warnings.filterwarnings("ignore")

    # Get the current working directory
    abspath = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory path
    parent_dir = os.path.dirname(abspath)
    result_dir = os.path.join(parent_dir, "exps", args.exp)
    os.makedirs(result_dir, exist_ok=True)

    # Load hyperparameters from yaml
    hyperpara_path = os.path.join(result_dir, 'exp_config.yaml')   
    with open(hyperpara_path, 'r') as yamlfile:
        hyperparams = yaml.safe_load(yamlfile)
    SAMPLING_RATE = hyperparams['sampling_rate']
    SPEED_OF_SOUND = hyperparams['speed_of_sound']
    window_period = hyperparams['frame_length']
    overlap_rate = hyperparams['frame_length'] - hyperparams['frame_hop']

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()

   



