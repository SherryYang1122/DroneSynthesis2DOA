# mic_array

## Getting microphone arrays

This section describes the process of generating multiple microphone arrays. Users can input parameters for multiple microphone arrays in `Data/exp../xxx.yaml` file, including the microphone type (tetra or octahedron or individual), the center position of the array, rotation angles around the X-axis, Y-axis, and Z-axis (in degrees), and microphone white noise intensity (dB). If the type is tetra, the input also requires the length of the side  (in meters). Some yaml examples are shown for reference in this part (mic_config1/2/3.yaml).

After saving the parameters, run the script mic_generator.py as an example:
```bash
python  mic_generator_main.py --yaml_file xxx.yaml --exp exp3
```

You will get the specific coordinates of each microphone in the array, which are saved in its corresponding `Data/exp../xxx.json` file for later use.

## Tips
1. In one `xxx.yaml` file, the user can input multiple microphone arrays (you can see 'mic_config3.yaml' as reference)
2. There are three types of arrays so far. 'individual' means the user can define the positions of microphones in array directly (you can see 'mic_config2.yaml' as reference).
3. RotX, RotY, and RotZ perform rotation operations around the X, Y, and Z axes (in degrees), respectively, rotating the given 3D object's vertex coordinates around the corresponding axis. If the value is positive, the rotation direction follows the right-hand rule; if negative, it's the opposite. If the values for RotX, RotY, and RotZ are not specified in the YAML file, they default to 0 (no rotation).

## 📝 Citation
If you use any files of this repository, please cite:

Yang, X., Naylor, P. A., Doclo, S., Bitzer, J., 
"NEURAL DRONE LOCALIZATION EXPLOITING SIGNAL SYNTHESIS OF REAL-WORLD AUDIO DATA", Eusipco 2025, Italy


