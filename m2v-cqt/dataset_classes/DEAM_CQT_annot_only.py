import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from dataset_classes.DEAM_CQT import *

def display_cqt(chroma_cq, hop_length):
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    img = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time',hop_length=hop_length)
    ax.set(title='chroma_cqt')
    fig.colorbar(img, ax=ax)

class DEAM_CQT_Dataset_Annotations_Only(DEAM_CQT_Dataset):
    def __init__(self, annot_path: str, audio_path: str, save_files: bool, transform_path: str, transform_name: str, transform_func=librosa.feature.chroma_cqt, start_s=15, dur=30, train=True):
        super().__init__(annot_path, audio_path, save_files, transform_path, transform_name, transform_func, start_s, dur, train)

    def calculate_transform(self, index: int, save_files = False):
        return torch.empty((0))
    
    def get_annots(self, index: int, len=60):
        annots = self.annot_df.loc[index].to_numpy()
        annots = annots[2:len + 2]
        annots = torch.tensor(annots).double()
        return annots
    
    def __getitem__(self, index: int):
        chroma_cq = self.calculate_transform(index, save_files=self.save_files)
        annots = self.get_annots(index)

        return chroma_cq, annots