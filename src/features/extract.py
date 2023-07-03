import pandas as pd
import librosa
# import opensmile
import pickle

AUDIO_PATH = "./data/processed/audio/"
CHUNK_SIZE = 10

def extract_librosa_features(song_id_list):
    features_list = []
    size = len(song_id_list)
    iter = 0

    for song_id in song_id_list:
        if iter % CHUNK_SIZE == 0:
            print(f"{iter}/{size}")
        iter += 1

        waveform, sample_rate = librosa.load(AUDIO_PATH + f"{song_id}.mp3")

        mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
        rms = librosa.feature.rms(y=waveform)
        tempo = librosa.feature.tempo(y=waveform, sr=sample_rate)
        onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
        zcr = librosa.feature.zero_crossing_rate(waveform)
        chromagram = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sample_rate)
        features_list.extend([mfcc, rolloff, centroid, rms, tempo, onset_env, zcr, chromagram, pitches, magnitudes])

    librosa_features = pd.DataFrame(
        data=features_list,
        columns=["mfcc", "rolloff", "centroid", "rms", "tempo", "onset_env", "zcr", "chromagram", "pitches", "magnitudes"]
    )

    # TODO save features by chunk
    with open("librosa_features.pkl", "wb") as f:
        pickle.dump(librosa_features, f)

# FIXME fix opensmile features
# def extract_opensmile_features(song_id_list):
#     smile = opensmile.Smile(
#         feature_set=opensmile.FeatureSet.emobase,
#         feature_level=opensmile.FeatureLevel.Functionals,
#     )
#     features_list = [] # list of smile features for each clip
#     size = len(song_id_list)
#     iter = 0
#     for file in song_id_list:
#         if iter % CHUNK_SIZE == 0:
#             print(f"{iter}/{size}")
        
#         # get smile features
#         filepath = AUDIO_PATH + str(file) + ".wav"
#         smile_features = smile.process_file(filepath)
#         # convert from df to list
#         smile_features = smile_features.values.tolist()
#         # convert from 2d list to 1d list
#         smile_features = sum(smile_features, [])
#         features_list.append(smile_features)

#     opensmile_features = pd.DataFrame(
#         data=features_list
#     )
    
#     with open("opensmile_features.pkl", "wb") as f:
#         pickle.dump(opensmile_features, f)

if __name__ == "__main__":
    from utilities import get_song_id_list

    song_id_list = get_song_id_list()
    extract_librosa_features(song_id_list)
    # FIXME
    # extract_opensmile_features(song_id_list)