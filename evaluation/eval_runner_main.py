# eval_runner_main.py
# This code provides the main evaluation runner for DOA estimation performance analysis.
# (c) 2025, X. Yang, Fraunhofer IDMT, Germany, MIT License
# version 1.0, August 2025

import numpy as np
import argparse
import warnings
import os
import json
import csv
import yaml
import librosa

from geometry import cart2sph
from visualization import plot_sph_angles_time_axis, plot_sph_angles, plot_doa_error_over_time, plot_doa_error_vs_distance, plot_signal_spectrogram, plot_doa_error_vs_distance_real_snr


# evaluation part
def main():
    # Get the current working directory
    abspath = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory path
    parent_dir = os.path.dirname(abspath)
    result_dir = os.path.join(parent_dir, "exps", args.exp)
    os.makedirs(result_dir, exist_ok=True)
    alg_dir = os.path.join(parent_dir, "exps", args.exp, args.eval_alg)
    os.makedirs(alg_dir, exist_ok=True)

    eval_alg = args.eval_alg
    est_path = "DOA_"+eval_alg
    if args.beta:
        est_path = est_path + "_beta"
    if args.mask:
        est_path = est_path + "_mask"

    doa_est_path = os.path.join(result_dir, est_path)
    doa_real_path = os.path.join(result_dir, args.dataset)

    if args.example:
        # get an example for visulation
        example_path = os.path.join(result_dir, "eval_example")
        example_path = os.path.join(example_path, args.sample_name)
        os.makedirs(example_path, exist_ok=True)
        with open(os.path.join(doa_est_path, args.sample_name + ".csv"), 'r', newline='') as file:
            doa_est = csv.reader(file)
            doa_est = list(doa_est)[1:]
            doa_est = [list(map(float, row)) for row in doa_est] 
        # load mic array
        mic_jason = "mic_config.json"
        # Open the JSON file and load its contents
        with open(os.path.join(result_dir, mic_jason), 'r') as file:
            mic_data = json.load(file)    

        # get real doa
        doa_real_dir = os.path.join(doa_real_path, args.sample_name)
        doa_real_sample = os.path.join(doa_real_dir, "DOATruePos.csv")
        drone_wav = os.path.join(doa_real_dir, f"{args.sample_name}.wav")
        signal, sample_rate = librosa.load(drone_wav, sr=None, mono=False)
        num_channels = 1 if signal.ndim == 1 else signal.shape[0]
        print(f"sampling rate: {sample_rate} Hz")
        print(f"channel number: {num_channels}")

        with open(doa_real_sample, 'r', newline='') as file:
            doa_real = csv.reader(file)
            doa_real = list(doa_real)[1:]
            doa_real = [list(map(float, row)) for row in doa_real] 
        real_sph = []
        est_sph =[]
        DOAerror = []
        real_xyz = []
        ER20_num = 0
        for i in range(min(len(doa_est), int(len(doa_real)*resolution/frame_hop)-1)):
            est_sph.append(doa_est[i][4:])
            real_sph.append(doa_real[int(i*frame_hop/resolution)][4:])
            real_xyz.append(doa_real[int(i*frame_hop/resolution)][1:4])
            error = angle_between_vectors(doa_est[i][1:4], doa_real[int(i*frame_hop/resolution)][1:4])
            DOAerror.append(error)
            if error > 20:
                ER20_num += 1 
        # visualization
        interesting_timestamp = [] #[300,1000,1250,1500]
        interesting_period = []
        interesting_points = []
        for time in interesting_timestamp:
            interesting_points.append([doa_real[int(time/resolution)][1:4],time])
        # visualization
        ele_real = [angle[0] for angle in real_sph]
        azi_real = [angle[1] for angle in real_sph]
        ele_est = [angle[0] for angle in est_sph]
        azi_est = [angle[1] for angle in est_sph]
        plot_sph_angles_time_axis(azi_real, ele_real, azi_est, ele_est, frame_hop, example_path)
        plot_sph_angles(azi_real, ele_real, azi_est, ele_est, example_path)
        plot_doa_error_over_time(DOAerror, frame_hop, example_path)
        plot_signal_spectrogram(signal[0], sample_rate, example_path)
        print("The DOAerror of " + args.sample_name + ": " + str(np.mean(np.array(DOAerror))) + " degrees")
        print("The error rate ER20 of " + args.sample_name + ": " + str(ER20_num/len(doa_est)) )



    else:
        DOA_eval = []
        snr_values = []
        real_bandpass_snr_values = []
        distances = []
        doa_errors = []
        for sampel_csv in os.listdir(doa_est_path):
            if sampel_csv.endswith('.csv'):
                # get estimated doa
                with open(os.path.join(doa_est_path, sampel_csv), 'r', newline='') as file:
                    doa_est = csv.reader(file)
                    doa_est = list(doa_est)[1:]
                    doa_est = [list(map(float, row)) for row in doa_est] 
                sample_name = sampel_csv[:-4]
                # get real doa
                doa_real_sample = os.path.join(doa_real_path, sample_name)
                doa_real_sample = os.path.join(doa_real_sample, "DOATruePos.csv")
                with open(doa_real_sample, 'r', newline='') as file:
                    doa_real = csv.reader(file)
                    doa_real = list(doa_real)[1:]
                    doa_real = [list(map(float, row)) for row in doa_real] 
                DOAerror = []
                dis = []
                ER20_num = 0
                start_index = 10 #2
                for i in range(start_index, min(len(doa_est), int(len(doa_real)*resolution/frame_hop)-1)):
                    if args.only_azi:
                        error = angle_between_vectors(np.array(doa_est[i][1:3]), np.array(doa_real[int(i*frame_hop/resolution)][1:3]))
                    else:
                        error = angle_between_vectors(np.array(doa_est[i][1:4]), np.array(doa_real[int(i*frame_hop/resolution)][1:4]))
                    pos = np.array(doa_real[int(i*frame_hop/resolution)][1:4])
                    dis.append(np.linalg.norm(pos))
                    DOAerror.append(error)
                    if error > 20:
                        ER20_num += 1    

                eval = {
                        "name": sample_name,
                        "frame_num": len(DOAerror),
                        "DOAerror": np.mean(np.array(DOAerror)),
                        "ER_20": ER20_num/len(DOAerror)
                    }
                DOA_eval.append(eval)
                distances.append(np.mean(np.array(dis)))
                doa_errors.append(np.mean(np.array(DOAerror)))

                # read drone metadata
                metadata_path = os.path.join(os.path.join(doa_real_path, sample_name), "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as file:
                        drone_data = json.load(file)[0]
                    if "SNR" in drone_data:
                        snr_values.append(drone_data["SNR"])
                    if "real_bandpass_SNR" in drone_data:
                        real_bandpass_snr_values.append(drone_data["real_bandpass_SNR"])

        # calculate the overall DOAerror of dataset
        doa_error_all = 0
        ER20_num_all = 0
        frame_all = 0
        ACC20_ave = []
        frame_min = 10000000
        frame_max = 0
        for eval_sample in DOA_eval:
            frame_all += eval_sample["frame_num"]
            if eval_sample["frame_num"] < frame_min:
                frame_min = eval_sample["frame_num"]
            if eval_sample["frame_num"] > frame_max:
                frame_max = eval_sample["frame_num"]
            doa_error_all += eval_sample["frame_num"]*eval_sample["DOAerror"]
            ER20_num_all += eval_sample["frame_num"]*eval_sample["ER_20"]
            ACC20_ave.append(1-eval_sample["ER_20"])
        overall_eval = {
            "frame_total": frame_all,
            "total flight time (minutes)": round(frame_all*frame_hop/60.0, 2),
            "one_flight_min(s)": frame_min*frame_hop,
            "one_flight_max(s)": frame_max*frame_hop,
            "DOAerror": np.mean(np.array(doa_errors)),
            "ACC_20": np.mean(np.array(ACC20_ave)),
            "ERR_20": 1-np.mean(np.array(ACC20_ave)),
            "DOAerror(weighted)": doa_error_all/frame_all,
            "ACC_20(weighted)": 1-ER20_num_all/frame_all,
            "ERR_20(weighted)": ER20_num_all/frame_all
        }
        DOA_eval.append(overall_eval)
        print(f"Results for {args.dataset} with {est_path}")
        print("The DOAerror in this dataset is " + str(overall_eval["DOAerror"]) + " degrees")
        print("The accuracy rate ACC20 is " +str(overall_eval["ACC_20"]))
        print("The accuracy rate ER20 is " +str(1-overall_eval["ACC_20"]))
        print("The DOAerror (weighted) in this dataset is " + str(overall_eval["DOAerror(weighted)"]) + " degrees")
        print("The accuracy rate ACC20 (weighted) is " +str(overall_eval["ACC_20(weighted)"]))
        print(f"total frame number: {frame_all}")
        print(f"total flight time: {round(frame_all*frame_hop/60.0,2)} minutes")
        print(f"one flight time is between {frame_min*frame_hop} and {frame_max*frame_hop} seconds")
        # save error of all samples
    
        json_name = args.eval_alg
        if args.beta:
            json_name = json_name + "_beta"
        if args.mask:
            json_name = json_name + "_mask"
        json_name = json_name +'_eval.json'
        with open(os.path.join(alg_dir, json_name), "w") as json_file:
            json.dump(DOA_eval, json_file, indent=4) 
        
        if snr_values != []:
            # plot results
            plot_doa_error_vs_distance(distances, doa_errors, snr_values, alg_dir)
            plot_doa_error_vs_distance_real_snr(distances, doa_errors, real_bandpass_snr_values, alg_dir)
            snr_to_doa_errors = {}
            for doa_error, snr_value in zip(doa_errors, snr_values):
                if snr_value not in snr_to_doa_errors:
                    snr_to_doa_errors[snr_value] = []
                snr_to_doa_errors[snr_value].append(doa_error)
            snr_to_mean_doa_error = {snr: np.mean(errors) for snr, errors in snr_to_doa_errors.items()}
            for snr, mean_doa_error in snr_to_mean_doa_error.items():
                print(f"SNR: {snr}, Mean DOA Error: {mean_doa_error:.2f} degrees")                   

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
    dot_product = np.dot(vector1, vector2)
    # Calculate the lengths of the vectors
    length_vector1 = np.linalg.norm(vector1)
    length_vector2 = np.linalg.norm(vector2)
    # Calculate the cosine of the angle
    cosine_angle = dot_product / (length_vector1 * length_vector2)
    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_angle)
    # Convert radians to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimation')
    parser.add_argument('--exp', type=str, default='exp_test', help='Enter experiment ID') 
    parser.add_argument('--dataset', type=str, default='MicArrayData', help='Path to the wav data for localization')   
    parser.add_argument('--eval_alg', choices=['srp_phat'], default='srp_phat', help='Localization algorithm choice')
    parser.add_argument('--beta', action='store_true', help='Only consider adding beta to srp phat')
    parser.add_argument('--mask', action='store_true', help='Only consider adding mask in frequqency domain')
    # give one example for visualization, all results saved in "example" folder
    parser.add_argument('--example', action='store_true', help='Enable give an example of one sample')
    parser.add_argument('--sample_name', type=str, default="sample_0.wav", help='Choose the sample') 
    parser.add_argument('--only_azi', action='store_true', help='Only consider azi degree during test, ignore ele degree')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    # Get the current working directory
    abspath = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory path
    parent_dir = os.path.dirname(abspath)
    result_dir = os.path.join(parent_dir, "exps", args.exp)
    os.makedirs(result_dir, exist_ok=True)

    hyperpara_path = os.path.join(result_dir, 'exp_config.yaml')
    with open(hyperpara_path, 'r') as yamlfile:
        hyperparams = yaml.safe_load(yamlfile)           
    # Access hyperparameters
    speed_of_sound = hyperparams['speed_of_sound'] # Speed of sound in air (m/s)
    sampling_rate = hyperparams['sampling_rate']
    # real time localizaton: frame parameters 
    frame_hop = hyperparams['frame_hop']

    # real drone doa resolution
    resolution = hyperparams['resolution']

    main()
