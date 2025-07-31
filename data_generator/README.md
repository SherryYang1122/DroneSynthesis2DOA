# Train and Test Data Generator

## Overview

Train and Test Data Generator is a tool designed for generating mixed signal datasets for training and testing purposes. It allows users to create datasets based on synthetic drone signals, environmental noise, ground/wall settings, and microphone array data with Signal-to-Noise Ratio (SNR) settings, mimicking real-world scenarios.

## Usage

To use the Train and Test Data Generator, users need to provide the following inputs:

- Drone audio and flight path data
- Background noise files

These data or file can be generated and saved in **Data/exp...** folder.

An example of how to run the script to generate 20 samples is as follows:

```bash
python data_generator_main.py --num 10 --exp exp3 
```


The generated dataset will be saved in a folder named such as **Data/exp.../MicArrayData**. For each sample, there are a wav file, a json file and a csv file for real-time real positions in it.

 
## Input Variables

The following input variables can be adjusted to customize the dataset generation process:

- `--num`: Number of samples in the dataset (default: 5)
- `--env_path`: Path to the background noise folder 
- `--drone_data`: Path to the drone data file (default='DroneAudioData')
- `--output`: Simulated microphone dataset are saved in this folder (default='MicArrayData')
- `--wall`: Optional argument to specify whether there is wall reflection (default: False)

## 📝 Citation
If you use any files of this repository, please cite:

Yang, X., Naylor, P. A., Doclo, S., Bitzer, J., 
"NEURAL DRONE LOCALIZATION EXPLOITING SIGNAL SYNTHESIS OF REAL-WORLD AUDIO DATA", Eusipco 2025, Italy

