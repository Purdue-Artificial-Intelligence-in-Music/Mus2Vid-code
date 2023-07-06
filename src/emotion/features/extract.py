import os
import pandas as pd
import librosa
import opensmile
import joblib 
from utilities import AUDIO_DIR, FEATURES_DIR, FEATURES_EXT, CHUNK_SIZE


def extract_librosa_features(song_id_list):
    features_list = []

    size = len(song_id_list)
    for i, song_id in enumerate(song_id_list):
        if i % CHUNK_SIZE == 0:
            print(f"{i}/{size}")
        if i == CHUNK_SIZE: # TODO
            break # TODO

        waveform, sample_rate = librosa.load(f"{AUDIO_DIR}/{song_id}.wav")

        mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
        rms = librosa.feature.rms(y=waveform)
        tempo = librosa.feature.tempo(y=waveform, sr=sample_rate)
        onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
        zcr = librosa.feature.zero_crossing_rate(waveform)
        chromagram = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sample_rate)
        features_list.append([mfcc, rolloff, centroid, rms, tempo, onset_env, zcr, chromagram, pitches, magnitudes])

    librosa_features = pd.DataFrame(
        data=features_list,
        columns=["mfcc", "rolloff", "centroid", "rms", "tempo", "onset_env", "zcr", "chromagram", "pitches", "magnitudes"]
    )

    # TODO save features by chunk
    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)
    joblib.dump(librosa_features, f"{FEATURES_DIR}/librosa.{FEATURES_EXT}")

def extract_opensmile_features(song_id_list):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features_list = [] # list of smile features for each clip

    size = len(song_id_list)
    for i, file in enumerate(song_id_list):
        if i % CHUNK_SIZE == 0:
            print(f"{i}/{size}")
        
        # get smile features
        smile_features = smile.process_file(f"{AUDIO_DIR}/{file}.wav")
        # convert from df to list
        smile_features = smile_features.values.tolist()
        # convert from 2d list to 1d list
        smile_features = sum(smile_features, [])
        features_list.append(smile_features)

    opensmile_features = pd.DataFrame(
        data=features_list
    )
    
    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)
    joblib.dump(opensmile_features, f"{FEATURES_DIR}/opensmile.{FEATURES_EXT}")


if __name__ == "__main__":
    from utilities import get_song_id_list

    song_id_list = get_song_id_list()
    extract_opensmile_features(song_id_list)
