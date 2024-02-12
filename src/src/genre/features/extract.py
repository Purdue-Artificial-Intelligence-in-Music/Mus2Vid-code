import opensmile
import numpy as np
import pandas as pd
import pickle, lzma
import pretty_midi as pm
import os
from src.features.util import *
from src.data.process import process_audio

def get_features() -> None:
    if(not(os.path.exists(f"{INTERIM_DATA_DIR}audio_clips.xz") and os.path.exists(f"{INTERIM_DATA_DIR}clip_genres.xz"))):
        process_audio()
    
    with lzma.open(f"{INTERIM_DATA_DIR}audio_clips.xz", "rb") as f:
        audio_clips = pickle.load(f)
    with lzma.open(f"{INTERIM_DATA_DIR}clip_genres.xz", "rb") as f:
        clip_genres = pickle.load(f)

    smile_features = get_smile_feats(audio_clips)

    if(not(os.path.exists(f"{INTERIM_DATA_DIR}matched_midi.xz"))):
        get_matched_midi(audio_clips, clip_genres)

    with lzma.open(f"{INTERIM_DATA_DIR}matched_midi.xz", "rb") as f:
        matched_midi = pickle.load(f)

    midi_features = extract_midi_features(matched_midi)

    labeled_features = np.concatenate((smile_features, midi_features), axis=1) # smile features don't have genre labels, midis do. This way the last column is genre

    # Save feature matrix using pickle for easier access later
    with lzma.open(f"{INTERIM_DATA_DIR}labeled_features.xz", "wb") as f:
        pickle.dump(labeled_features, f)

    return 

def get_smile_feats(audio):
    """Create a feature matrix of openSMILE features from the emobase set (988 total features) for all of the audio clips

    Parameters
    ----------
    audio: list
        filepaths to audio clips

    Returns
    -------
    feature_array: np.array
        array of features for each audio clip. 988 columns for each of 988 features. len(audio) rows (one row for every audio clip).
    """
    # create smile object for feature extraction
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    all_smiles = [] # list of smile features for each clip
    iters = 0
    for file in audio:
        iters += 1
        print(str(iters) + "/" + str(len(audio)))
        
        # get smile features
        smile_feats = smile.process_file(PROCESSED_AUDIO_FOLDER + file)
        # convert from df to list
        smile_feats = smile_feats.values.tolist()
        # convert from 2d list to 1d list
        smile_feats = sum(smile_feats, [])
        all_smiles.append(smile_feats)

    feature_array = np.asarray(all_smiles)
    
    return feature_array

def normalize_features(features):
    """Normalizes the features to the range [-1, 1].

    Parameters
    ----------
    features: list of floats
        The array of features.

    Returns
    -------
    list of floats
        normalized features
    """
    # Normalize each feature based on its specific range
    tempo = (features[0] - 150) / 300
    num_sig_changes = (features[1] - 2) / 10
    resolution = (features[2] - 260) / 400
    time_sig_1 = (features[3] - 3) / 8
    time_sig_2 = (features[4] - 3) / 8
    melody_complexity = (features[5] - 0) / 10
    melody_range = (features[6] - 0) / 80

    # Normalize pitch class histogram
    pitch_class_hist = [((f - 0) / 100) for f in features[7:-1]]

    # Return the normalized feature vector
    return [tempo, num_sig_changes, resolution, time_sig_1, time_sig_2, melody_complexity, melody_range] + pitch_class_hist

def get_midi_features(midi_obj):# , all_chords = generate_chord_list()):
    """Extracts specific features from a PrettyMIDI object given its path using the pretty_midi library.

    Parameters
    ----------
        midi_obj: pm.PrettyMIDI 
            the PrettyMIDI object

    Returns
    -------
        features: list of floats
            The extracted features
    """
    
    # tempo: the estimated tempo of the audio file
    try:
        tempo = midi_obj.estimate_tempo()
    except:
        tempo = 0

    # num_sig_changes: the number of time signature changes in the audio file
    num_sig_changes = len(midi_obj.time_signature_changes)

    # resolution: the time resolution of the audio file (in ticks per beat)
    resolution = midi_obj.resolution


    # Extract time signature information
    ts_changes = midi_obj.time_signature_changes
    ts_1, ts_2 = 4, 4
    if len(ts_changes) > 0:
        ts_1 = ts_changes[0].numerator
        ts_2 = ts_changes[0].denominator
    
    # Extract melody-related features
    # melody: a pitch class histogram of the audio file
    melody = midi_obj.get_pitch_class_histogram()
    # melody_complexity: the number of unique pitch classes in the melody
    melody_complexity = np.sum(melody > 0)
    # melody_range: the range of pitch classes in the melody
    melody_range = np.max(melody) - np.min(melody)
    # OPTIONAL feature melody_contour: the temporal evolution of pitch content in the audio file
    # melody_contour = librosa.feature.tempogram(y=file.fluidsynth(fs=16000), sr=16000, hop_length=512)
    # melody_contour = np.mean(melody_contour, axis=0)
    # chroma: a chroma representation of the audio file
    chroma = midi_obj.get_chroma()
    # pitch_class_hist: the sum of the chroma matrix along the pitch axis
    pitch_class_hist = np.sum(chroma, axis=1)

    # Chord detection functions
    # notes = consolidate_notes(midi_obj)
    # chords = calculate_song_chords(notes, all_chords=all_chords)
    # key = estimate_key(chords)
    # changes = chord_changes(chords, midi_obj)
    # grams = n_grams(chords, 3)

    features = normalize_features([tempo, num_sig_changes, resolution, ts_1,
                            ts_2, melody_complexity, melody_range] + list(pitch_class_hist))
    # features.append(key)
    # features.append(changes)
    # features.append(grams)

    return features

def extract_midi_features(midi_df):
    """Extracts features and labels from MIDI objects listed in the DataFrame and concatenates the
    features with their labels into a matrix.

    Parameters
    ----------
        path_df: pd.DataFrame
            A DataFrame with pretty-midi objects and their composition years.

    Returns
    -------
        all_features: numpy.ndarray 
            A matrix of features along with labels.
    """
    all_features = []  # List to store all extracted features
    iters = 0
    for index, row in midi_df.iterrows():
        print(iters)
        iters += 1

        midi_obj = row['midi_obj']
        obj_features = get_midi_features(midi_obj) #all_chords)
        obj_features.append(row['genre'])
        all_features.append(obj_features)
    # Return the numpy array of all extracted features along with corresponding years
    return np.array(all_features)

if __name__ == "__main__":
    get_features()
