import os
import pandas as pd
import librosa
import opensmile
import joblib 
from utilities import AUDIO_DIR, FEATURES_DIR, FEATURES_EXT, CHUNK_SIZE


##### librosa #####

def extract_librosa_features(audio_filepath):
    waveform, sample_rate = librosa.load(audio_filepath)

    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
    rms = librosa.feature.rms(y=waveform)
    tempo = librosa.feature.tempo(y=waveform, sr=sample_rate)
    onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(waveform)
    chromagram = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
    pitches, magnitudes = librosa.piptrack(y=waveform, sr=sample_rate)

    return [mfcc, rolloff, centroid, rms, tempo, onset_env, zcr, chromagram, pitches, magnitudes]


def get_librosa_features_list(audio_filepaths):
    librosa_features_list = []

    size = len(audio_filepaths)
    for i, audio_filepath in enumerate(audio_filepaths):
        if i % CHUNK_SIZE == 0:
            print(f"{i}/{size}")
        if i == CHUNK_SIZE: # TODO
            break # TODO

        librosa_features_list.append(extract_librosa_features(audio_filepath))

    return pd.DataFrame(
        data=librosa_features_list,
        columns=["mfcc", "rolloff", "centroid", "rms", "tempo", "onset_env", "zcr", "chromagram", "pitches", "magnitudes"]
    )


def save_librosa_features_list(audio_filepaths, filename="librosa"):
    librosa_features_list = get_librosa_features_list(audio_filepaths)

    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)
    joblib.dump(librosa_features_list, f"{FEATURES_DIR}/{filename}.{FEATURES_EXT}")


##### opensmile #####

def extract_opensmile_features(smile, audio_filepath):
    # Get smile features, convert from df to list, and convert from 2D list to 1D list
    return sum(smile.process_file(audio_filepath).values.tolist(), [])


def get_opensmile_features_list(audio_filepaths):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    opensmile_features_list = [] # list of smile features for each clip

    size = len(audio_filepaths)
    for i, audio_filepath in enumerate(audio_filepaths):
        if i % CHUNK_SIZE == 0:
            print(f"{i}/{size}")
        
        opensmile_features_list.append(extract_opensmile_features(smile, audio_filepath))

    return pd.DataFrame(data=opensmile_features_list)


def save_opensmile_features_list(audio_filepaths, filename="opensmile"):
    opensmile_features_list = get_opensmile_features_list(audio_filepaths)

    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)
    joblib.dump(opensmile_features_list, f"{FEATURES_DIR}/{filename}.{FEATURES_EXT}")


if __name__ == "__main__":
    from utilities import get_audio_filepaths

    audio_filepaths = get_audio_filepaths()
    save_opensmile_features_list(audio_filepaths)
    save_librosa_features_list(audio_filepaths)
