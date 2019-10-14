# music_generator
The project of stat7503


To do list: 
- convert data from MIDI into pianorow format (using pretty-midi library) 
- trial different architectures for music generation (LSTM, VAE, GAN) 

### how to use data preprocessing
navigate into data_clean folder, then type following code on terminal
```bash
conda env create -f environment.yml
```
The code above will create a same environment as this project.

For generating cleaned data, type following code on terminal
```bash
python3 preprocessing.py ~/path_of_your_midi_files
```

processed data will be stored with name piano_roll.npy file, in order to
load this data, type following code in python to load it back to numpy array
```python
arr = np.load('piano_roll.npy')
```