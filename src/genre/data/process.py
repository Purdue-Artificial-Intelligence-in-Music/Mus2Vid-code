import pandas as pd
import re
import os
import librosa
import pickle, lzma
import joblib
from src.genre.data.util import *
from src.genre.data.splitwavs import SplitWavAudio

def process_audio() -> None:
    """Converts raw maestro audio (from zip file in google drive) to 3 minute clips with genre labels.
    Saves clip filepaths and labels as .xz files in data/interim.
    Saves list of genres using joblib to be used in model later
    """
    maestro_filepaths = get_filepaths()
    split_audio(maestro_filepaths) # no return, creates data/processed/maestro/audio folder

    (audio_clips, clip_genres) = match_clips(get_labels())

    # Save lists using pickle for easier access later
    with lzma.open(f"{INTERIM_DATA_DIR}audio_clips.xz", "wb") as f:
        pickle.dump(audio_clips, f)
    with lzma.open(f"{INTERIM_DATA_DIR}clip_genres.xz", "wb") as f:
        pickle.dump(clip_genres, f)


def get_filepaths() -> list:
    """Generates a list of filepaths to maestro wav files

    Returns
    -------
    wavs: list
        list of audio filepaths
    """
    classical_data = pd.read_csv(ANNOTATIONS_FILE)

    wavs = list(classical_data['audio_filename'])
    for i in range(0, len(wavs)):
        wavs[i] = RAW_AUDIO_DIR + wavs[i]

    return wavs

def split_audio(wavs) -> None:
    """_summary_

    Parameters
    ----------
    wavs : _type_
        _description_
    """
    if not os.path.exists(PROCESSED_AUDIO_DIR): os.mkdir(PROCESSED_AUDIO_DIR)

    for fp in wavs:
        new_name = fp
        new_name = re.sub(f"{RAW_AUDIO_DIR}[0-9]+/", "", new_name) # resultant directory won't have the year folders present in maestro
        split_wav = SplitWavAudio(PROCESSED_AUDIO_DIR, new_name, fp)
        split_wav.multiple_split(min_per_split=3)

    return

def get_labels() -> tuple[list, list]:
    """_summary_

    Returns
    -------
    wavs: list
        maestro wav filepaths without year folders
    genre_ints: list
        integer representations of genre labels for each audio file
    """
    classical_data = pd.read_csv(ANNOTATIONS_FILE)

    wavs = list(classical_data['audio_filename'])
    genres = list(classical_data['Genre'])
    for i in range(0, len(wavs)):
        wavs[i] = re.sub("[0-9]+/", "", wavs[i]) # Remove the year folders because the clips created above are all in one folder

    genre_list = [] # List of classifications (strings)
    for i in genres:
        if not i in genre_list:
            genre_list.append(i)

    genre_ints = [] # Classifications (as ints) of each maestro file
    for i in range(len(genres)):
        j = 0
        while (not(genres[i] == genre_list[j])):
            j += 1
        genre_ints.append(j)

    # save genre_list for the model
    if (not(os.path.exists(f"{INTERIM_DATA_DIR}genre_list.jb"))):
        joblib.dump(genre_list, f"{INTERIM_DATA_DIR}genre_list.jb")

    return wavs, genre_ints

def get_len(wav):
    """Measures the duration of a wav file

    Parameters
    ----------
    wav: string
        audio filepath

    Returns
    -------
    length: int
        duration of audio file
    """
    length = librosa.get_duration(path=(PROCESSED_AUDIO_DIR + wav))

    return length

def match_clips(wavs, genre_ints) -> tuple[list,list]:
    """matches shortened clips from maestro to genres in the annotations file

    Parameters
    ----------
    wavs: list
        Names of full audio files without year folders from maestro annotations
    genre_ints: list
        Genres of full audio files

    Returns
    ------
    audio_clips: list

    """
    audio_clips = os.listdir(PROCESSED_AUDIO_DIR) # list of filepaths in the folder of 3 minute clips we just created
    clip_genres = [] # list of corresponding genres to each audio file

    for clip in audio_clips:
        match_str = re.sub("^[0-9]+_", "", clip) # remove the clip number to match short clips to their corresponding full songs, which have labels
        wav_index = wavs.index(match_str) # index (in wav array) of the full song
        clip_genres.append(genre_ints[wav_index]) # access tje genre of the full song and assign it to the clip

    # remove tracks less than 1 minute seconds long
    for i in range(len(audio_clips)):
        print(str(i) + "/" + str(len(audio_clips)))
        if (get_len(audio_clips[i]) < 60):
            print("short")
            audio_clips.pop(i)
            clip_genres.pop(i)
        if (i == len(audio_clips) - 1): # len(audio_clips) changes as short tracks are removed. This avoids going out of the bounds of the list
            break

    return audio_clips, clip_genres

if __name__ == "__main__":
    process_audio()