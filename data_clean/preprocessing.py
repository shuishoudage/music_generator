from typing import List, Tuple, Dict, Any
from collections import Counter
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
from shutil import copyfile
import shutil


# Ideas behind the preprocessing class
#
# 1. only use those midi with one tempo and one key, since some midi music
# have key and tempo changes inside. Which might make some unpredictable result
#
# 2. list distribution for all keys contained in the corpus. Only select those
# most frequent appeared. (different keys may increase training difficulty)
#
# 3. only select similar tempo music, based on the mean and std of tempos,
# simple one will be left boundary = mean - std, right boundary = mean + std
#
# 4. find the mean of highest and lowest pitch in the corpus. filter out those not
# the range. We have pitch range from 0-128, no meaning cover two extreme sides.
class FileReport(object):
    """
    This class is mainly for generating meta information for our report
    """

    def __init__(self,
                 tempos: List[float],
                 freq_key: Dict[int, int],
                 min_pitch: List[int],
                 max_pitch: List[int]):
        self.tempos = tempos
        self.freq_key = freq_key
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

    def aggregation_report(self):
        """
        two important variable are min_pitch and max_pitch,
        since they will be used to decode from pitch to audio
        """
        temp_mean = np.array(self.tempos).mean()
        temp_std = np.array(self.tempos).std()
        most_freq_key = self.getMostFreqValue(self.freq_key)
        min_pitch = int(np.array(self.min_pitch).mean())
        max_pitch = int(np.array(self.max_pitch).mean())
        return temp_mean, temp_std, most_freq_key, min_pitch, max_pitch

    def plots(self):
        # implement later on
        pass

    def getMostFreqValue(self, keys: Dict[int, int], reversed=True) -> int:
        return sorted(keys.items(), key=lambda kv: kv[1], reverse=reversed)[0][0]


class Preprocess(object):
    def __init__(self, path: str):
        self.path = path
        self.fileFilter()

    def generateMidiFileReport(self) -> FileReport:
        """
        meta information like tempos, keys, pitches will be generated for
        filtering the midi files
        """
        tempos = []
        keys = []
        max_pitchs = []
        min_pitchs = []
        for pm in self.pms:
            try:
                tempos.append(pm.estimate_tempo())
                key = pm.key_signature_changes[0].key_number
                keys.append(key)
                min_pitch, max_pitch = self.getMinMaxPitch(pm)
                max_pitchs.append(max_pitch)
                min_pitchs.append(min_pitch)
            except:
                pass
        self.report = FileReport(tempos, dict(
            Counter(keys)), min_pitchs, max_pitchs)
        return self.report

    def getMinMaxPitch(self, pm: pretty_midi.PrettyMIDI):
        """
        find the min and max pitch inside a midi file
        """
        notes = [
            note.pitch for instrument in pm.instruments for note in instrument.notes
        ]
        return min(notes), max(notes)

    def SaveFilterMIDIfiles(self):
        """
        according generated meta data info to filter out those not in range
        """
        report = self.generateMidiFileReport()
        temp_mean, temp_std, key, left_boundary, right_boundary = report.aggregation_report()
        piano_roll_paths = []
        for pm, path in zip(self.pms, self.paths):
            try:
                tempo = pm.estimate_tempo()
                min_pitch, max_pitch = self.getMinMaxPitch(pm)
                if self.isTempoInRange(tempo, temp_mean, temp_std) \
                    and self.isPitchInRange(min_pitch, max_pitch, left_boundary, right_boundary) \
                        and self.isKeyMatch(pm.key_signature_changes[0].key_number, key):
                    savedPath = os.path.join(os.getcwd(), 'filterData')
                    if not os.path.exists(savedPath):
                        os.makedirs(savedPath, exist_ok=True)
                    shutil.move(
                        path, os.path.join(os.getcwd(), 'filterData', os.path.basename(path)))
            except:
                pass

    def isTempoInRange(self, tempo: float, mean: float, std: float) -> bool:
        """
        a helper function that can be used check if a midi file's tempo in range
        """
        if tempo > (mean - std) and tempo < (mean + std):
            return True
        return False

    def isKeyMatch(self, key: int, grand_truth_key: int) -> bool:
        if key == grand_truth_key:
            return True
        return False

    def isPitchInRange(self, low_pitch: int,
                       high_pitch: int,
                       left_boundary: int,
                       right_boundary: int) -> bool:
        if low_pitch >= left_boundary and high_pitch <= right_boundary:
            return True
        return False

    def fileFilter(self):
        """
        first filtering that only allow one tempo and one key inside a midi file
        """
        self.pms: List[pretty_midi.PrettyMIDI] = []
        self.paths: List[str] = []
        for (dirPath, _, files) in walk(self.path):  # type: ignore
            for file in files:
                # get the absoluted path of file
                path = join(dirPath, file)
                try:
                    pm = pretty_midi.PrettyMIDI(path)
                    # only handle files contain one key and one tempo
                    if len(pm.key_signature_changes) == 1 \
                            and len(pm.time_signature_changes) == 1:
                        self.pms.append(pm)
                        self.paths.append(path)
                except:  # skip all parsing exceptions
                    pass


def cliArgParser(argv) -> Any:
    if len(argv) != 2:
        raise ValueError(f"path of folder must be provided")
    if isdir(argv[1]):
        path = os.path.abspath(argv[1])
        return path
    else:
        raise ValueError(f"provided path is not a folder")


if __name__ == "__main__":
    try:
        path = cliArgParser(argv)
        p = Preprocess(path)
        p.SaveFilterMIDIfiles()
    except Exception as err:
        print(traceback.format_exc())
        exit(1)
