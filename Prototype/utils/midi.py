import numpy as np
import pretty_midi
import tensorflow as tf
from basic_pitch.inference import predict as bp_predict
from basic_pitch import ICASSP_2022_MODEL_PATH
BASIC_PITCH_MODEL = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

def predict(filestr):
    """
    Makes a Basic Pitch prediction with the global parameters above given an input audio file.
    
    Parameters:
        filestr (str): The path to the input audio file.
        
    Returns:
        PrettyMIDI object containing predicted MIDI notes.
    """
    # Run prediction
    STD_ONSET = 0.3
    STD_FRAME = 0.2
    STD_MIN_NOTE_LEN = 50
    STD_MIN_FREQ = None
    STD_MAX_FREQ = 3000
    model_output, midi_data, note_events = bp_predict(
        filestr,
        BASIC_PITCH_MODEL,
        STD_ONSET,
        STD_FRAME,
        STD_MIN_NOTE_LEN,
        STD_MIN_FREQ,
        STD_MAX_FREQ
    ) # midi_data is the PrettyMIDI object corresponding to the prediction
    return midi_data

def normalize_features(features):
    """
    Normalizes the features to the range [-1, 1].

    Parameters:
        features (list of float): The array of features.

    Returns:
        list of float: Normalized features.
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
    return [tempo, resolution, time_sig_1, time_sig_2, melody_complexity, melody_range] + pitch_class_hist

def get_features(midi_path):
    """
    Extracts specific features from a PrettyMIDI object given its path using the pretty_midi library.
    Handle any potential errors with MIDI files appropriately.

    Parameters:
        midi_path: path to an audio file

    Returns:
        list of float: The extracted features.
    """
    
    # tempo: the estimated tempo of the audio file
    midi_obj = predict(midi_path)

    tempo = midi_obj.estimate_tempo()

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
    # chords = calculate_song_chords(midi_obj)
    # changes = chord_changes(chords, midi_obj)
    # grams = n_grams(chords, 3)

    features = normalize_features([tempo, num_sig_changes, resolution, ts_1,
                            ts_2, melody_complexity, melody_range])
    features.append(list(pitch_class_hist))
    # features.append(chords)
    # features.append(changes)
    # features.append(grams)
    
    features = np.asarray(features, dtype=object)
    features = np.expand_dims(features, axis = 0)
    print(features.shape)

    return features
