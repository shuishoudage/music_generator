# music_generator
The project of stat7503


To do list: 
- filter out midi file based on tempos, key, pitch for later use (using pretty-midi library) 
- try different architectures for music generation (LSTM, VAE, GAN) 

### how to use data preprocessing
navigate into data_clean folder, then type following code on terminal
```bash
conda env create -f environment.yml
```
The code above will create a same environment as this project.

For generating cleaned data, type following code on terminal
```bash
source activate music
python3 preprocessing.py ~/path_of_your_midi_files
```

processed data will be stored in a directory called `filterdata` just under
current root folder 