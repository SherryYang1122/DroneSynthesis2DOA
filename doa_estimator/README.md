# doa_estimator



## Usage

To use the doa_classical, users need to provide the following inputs:

- Microphone array configuration data

These data or file can be generated from other subprojects which are saved in **Data/exp../**.

An example of how to run the script to position is as follows:

```bash
python doa_runner_main.py --exp exp.. --algorithm srp_phat
```

The results are saved in this folder **Data/exp../DOA_xxx** such as **Data/exp../DOA_srp_phat** which depends on the algorithm.

## 📝 Citation
If you use any files of this repository, please cite:

Yang, X., Naylor, P. A., Doclo, S., Bitzer, J., 
"NEURAL DRONE LOCALIZATION EXPLOITING SIGNAL SYNTHESIS OF REAL-WORLD AUDIO DATA", Eusipco 2025, Italy
