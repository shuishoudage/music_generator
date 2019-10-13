from typing import List, Tuple
import pretty_midi
import matplotlib.pyplot as plt
import librosa.display
import os
from os import listdir, walk
from os.path import isfile, isdir, join
from sys import argv
import traceback
import logging
import numpy as np


# def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
#     # Use librosa's specshow function for displaying the piano roll
#     librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
#                              hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
#                              fmin=pretty_midi.note_number_to_hz(start_pitch))


# if len(argv) != 2:
#     raise ValueError("folder name must be provided")
folder = '/Users/PTT/Downloads/data'


def cli_arg_parser(argv: List[str]) -> str:
    if len(argv) != 2:
        raise ValueError(f"path of folder must be provided")
    if isdir(argv[1]):
        path = os.path.abspath(argv[1])
        return path
    else:
        raise ValueError(f"provided path is not a folder")


def dist_tempo(pms: List[pretty_midi.PrettyMIDI]) -> Tuple[float, float]:
    tempos = []
    for pm in pms:
        tempos.append(pm.estimate_tempo())
    tem_vec = np.array(tempos)
    return tem_vec.mean(), tem_vec.std()


def file_filter(folder: str) -> List[pretty_midi.PrettyMIDI]:
    pms: List[pretty_midi.PrettyMIDI] = []
    for (dirPath, _, files) in walk(folder):  # type: ignore
        for file in files:
            # get the absoluted path of file
            path = join(dirPath, file)
            try:
                pm = pretty_midi.PrettyMIDI(path)
                # only handle files contain one key and one tempo
                if len(pm.key_signature_changes) == 1 and len(pm.time_signature_changes) == 1:
                    pms.append(pm)
            except:  # skip all parsing exceptions
                pass
    return pms


if __name__ == "__main__":
    try:
        folder = cli_arg_parser(argv)
        pms = file_filter(folder)
        dist_tempo(pms)
    except Exception as err:
        logging.error(traceback.format_exc())
        exit(1)
# print(pm.key_signature_changes)
# print(pretty_midi.key_number_to_key_name(5))

# plt.figure(figsize=(12, 4))
# plot_piano_roll(pm, 0, 128)
# plt.show()
