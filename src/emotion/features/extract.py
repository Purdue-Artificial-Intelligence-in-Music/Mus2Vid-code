import os
import pandas as pd
import librosa
import opensmile
import joblib 
from src.emotion.features.util import FEATURES_DIR, FEATURES_EXT, CHUNK_SIZE


def extract_librosa_features(audio_filepaths):
    if os.path.exists(f"{FEATURES_DIR}/librosa.{FEATURES_EXT}"):
        return joblib.load(f"{FEATURES_DIR}/librosa.{FEATURES_EXT}")

    librosa_features = []

    size = len(audio_filepaths)
    for i, audio_filepath in enumerate(audio_filepaths):
        if i % CHUNK_SIZE == 0:
            print(f"{i}/{size}")
        if i == CHUNK_SIZE: # TODO
            break # TODO

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

        librosa_features.append(
            [mfcc, rolloff, centroid, rms, tempo, onset_env, zcr, chromagram, pitches, magnitudes]
        )

    if not os.path.exists(FEATURES_DIR): os.mkdir(FEATURES_DIR)
    joblib.dump(librosa_features, f"{FEATURES_DIR}/librosa.{FEATURES_EXT}")

    return pd.DataFrame(
        data=librosa_features,
        columns=["mfcc", "rolloff", "centroid", "rms", "tempo", "onset_env", "zcr", "chromagram", "pitches", "magnitudes"]
    )


def extract_opensmile_features(audio_filepaths):
    if os.path.exists(f"{FEATURES_DIR}/opensmile.{FEATURES_EXT}"):
        return joblib.load(f"{FEATURES_DIR}/opensmile.{FEATURES_EXT}")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    opensmile_features_list = [] # list of smile features for each clip

    size = len(audio_filepaths)
    for i, audio_filepath in enumerate(audio_filepaths):
        if i % CHUNK_SIZE == 0:
            print(f"{i}/{size}")

        # Get smile features, convert from df to list, and convert from 2D list to 1D list
        opensmile_features_list.append(sum(smile.process_file(audio_filepath).values.tolist(), []))

    if not os.path.exists(FEATURES_DIR): os.mkdir(FEATURES_DIR)
    joblib.dump(opensmile_features_list, f"{FEATURES_DIR}/opensmile.{FEATURES_EXT}")

    return pd.DataFrame(data=opensmile_features_list)


if __name__ == "__main__":
    from src.emotion.features.util import get_audio_filepaths

    audio_filepaths = get_audio_filepaths()
    print(extract_librosa_features(audio_filepaths))
    print(extract_opensmile_features(audio_filepaths))
