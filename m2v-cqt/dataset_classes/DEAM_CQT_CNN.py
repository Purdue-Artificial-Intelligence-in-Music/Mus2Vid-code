import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os

def display_cqt(chroma_cq, hop_length):
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    img = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time',hop_length=hop_length)
    ax.set(title='chroma_cqt')
    fig.colorbar(img, ax=ax)

class DEAM_CQT_Dataset_For_CNN(torch.utils.data.Dataset):
    def __init__(self, annot_path: str, audio_path: str, save_files: bool, transform_path: str, transform_name: str, transform_func=librosa.feature.chroma_cqt, start_s=15, dur=30, train=True):
        super(DEAM_CQT_Dataset_For_CNN, self).__init__()

        self.LEN_WINDOW = 4

        self.annot_path = annot_path
        self.audio_path = audio_path
        self.transform_path = transform_path
        self.transform_name = transform_name
        self.train = train
        self.transform_func = transform_func
        self.save_files = save_files

        self.annot_df = pd.read_csv(annot_path)
        
        self.start_s = start_s
        self.dur = dur

        self.n_items = len(self.annot_df)
        if train:
            self.annot_df.drop(list(range(0, int(0.1* self.n_items))), inplace=True)
        else:
            self.annot_df.drop(list(range(int(0.1* self.n_items),  self.n_items)), inplace=True)
        self.annot_df = self.annot_df.reset_index()
        self.n_items = len(self.annot_df)
        self.num_perms = self.dur - self.LEN_WINDOW + 1

    def __len__(self):
        return self.n_items * self.num_perms
    
    def get_path(self, index):
        if self.train:
            train_str = "train_"
        else:
            train_str = "test_"
        return self.transform_path + self.transform_name + "_" + train_str + str(index) + ".pt"
    
    def calculate_transform(self, index: int, start: int, end: int, save_files = False):
        path = self.get_path(index)
        if os.path.exists(path):
            chroma_cq = torch.load(path)
        else:
            song_id = self.annot_df.loc[index, "song_id"]
            song_path = "".join([self.audio_path, "/", str(song_id), ".mp3"])
            y, sr = librosa.load(song_path)
            HOP_LENGTH = int(sr/2) + 1
            y = y[self.start_s*sr:(self.start_s + self.dur)*sr]
            chroma_cq = self.transform_func(y=y, sr=sr, hop_length=HOP_LENGTH)
            chroma_cq = torch.tensor(chroma_cq).double()
            chroma_cq = torch.transpose(chroma_cq, 0, 1)
            if save_files:
                torch.save(chroma_cq, path)

        chroma_cq = chroma_cq[start:end]

        return chroma_cq
    
    def get_annots(self, index: int, start: int, end: int, len: int):
        annots = self.annot_df.loc[index].to_numpy()
        annots = annots[2 + start : 2 + end - self.LEN_WINDOW]
        annots = torch.tensor(annots).double()
        return annots
    
    def save_transforms(self):
        for index in range(self.__len__()):
            self.calculate_transform(index)

    def __getitem__(self, index: int):
        file_num = int(np.floor(index / self.num_perms))
        start = index % self.num_perms
        end = start + self.LEN_WINDOW
        chroma_cq = self.calculate_transform(file_num, start, end, save_files=self.save_files)
        annots = self.get_annots(file_num, start, end, chroma_cq.shape[0])

        return chroma_cq, annots
    
    def specshow(self, index: int):
        song_id = self.annot_df.loc[index, "song_id"]
        song_path = "".join([self.audio_path, "/", str(song_id), ".mp3"])
        y, sr = librosa.load(song_path)
        HOP_LENGTH = int(sr/2) + 1
        y = y[self.start_s*sr:(self.start_s + self.dur)*sr]
        chroma_cq = self.transform_func(y=y, sr=sr, hop_length=HOP_LENGTH)
        librosa.display.specshow(data=chroma_cq, sr=sr, hop_length=HOP_LENGTH, y_axis='chroma', x_axis='time',)
