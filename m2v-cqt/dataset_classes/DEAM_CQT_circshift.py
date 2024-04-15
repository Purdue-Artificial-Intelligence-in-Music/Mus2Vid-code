import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from dataset_classes.DEAM_CQT import DEAM_CQT_Dataset

def display_cqt(chroma_cq, hop_length):
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    img = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time',hop_length=hop_length)
    ax.set(title='chroma_cqt')
    fig.colorbar(img, ax=ax)

class DEAM_CQT_Dataset_With_CircShift(DEAM_CQT_Dataset):
    def __init__(self, annot_path: str, audio_path: str, save_files: bool, transform_path: str, transform_name: str, transform_func=librosa.feature.chroma_cqt, start_s=15, dur=30, train=True):
        super().__init__(annot_path, audio_path, save_files, transform_path, transform_name, transform_func, start_s, dur, train)

        test_transf = super().calculate_transform(index=1, save_files=False)
        self.transform_width = test_transf.shape[1]
        self.df_size = len(self.annot_df)

    def __len__(self):
        return self.df_size * self.transform_width

    def calculate_transform(self, index: int, save_files = False):
        new_index = index % self.df_size
        transf = super().calculate_transform(index=new_index, save_files=save_files)
        roll_val = int(np.floor(index / self.df_size))
        return transf.roll(roll_val, dims=(1))
    
    def get_annots(self, index: int, len: int):
        new_index = index % self.df_size
        annots = super().get_annots(index=new_index, len=len)
        roll_val = int(np.floor(index / self.df_size))
        return annots.roll(roll_val, dims=(0))

    def __getitem__(self, index: int):
        chroma_cq = self.calculate_transform(index, save_files=self.save_files)
        annots = self.get_annots(index, chroma_cq.shape[0])

        return chroma_cq, annots
