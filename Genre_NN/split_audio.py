import pandas as pd
from pydub import AudioSegment
import math
import os
import ffmpeg

classical_data = pd.read_csv('classical_dataset.csv')
# print (classical_data)

wavs = list(classical_data['audio_filename'])
for i in range(0, len(wavs)):
    wavs[i] = wavs[i][5:]
    wavs[i] = "wavs/" + wavs[i]

mp3s = []
for i in range(len(wavs)):
    mp3_name = str.rstrip(wavs[i], "wav")
    mp3_name = str.lstrip(mp3_name, "wavs")
    mp3_name = "mp3" + mp3_name + "mp3"
    mp3s.append(mp3_name)

mypath = os.path.abspath(__file__)
print(mypath)
mydir = os.path.dirname(mypath)

class SplitWavAudioMubin():
    def __init__(self, folder, filename, path):
        self.folder = folder
        self.filename = filename
        self.filepath = path
        
        self.audio = AudioSegment.from_file(self.filepath, format="mp3")
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="mp3")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')

# fp = os.path.join(mydir, mp3s[0])
# new_folder = "audio"
# new_name = ""
# print(fp, new_folder, new_name)
# sound = AudioSegment.from_mp3(fp)
# split_wav = SplitWavAudioMubin(new_folder, new_name, fp)
# split_wav.multiple_split(min_per_split=.5)