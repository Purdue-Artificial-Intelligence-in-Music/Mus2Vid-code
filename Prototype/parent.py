import pandas as pd
import pickle
import numpy as np
import pretty_midi
import tensorflow as tf
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
BASIC_PITCH_MODEL = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

# Parameters
STD_ONSET = 0.3
STD_FRAME = 0.2
STD_MIN_NOTE_LEN = 50
STD_MIN_FREQ = None
STD_MAX_FREQ = 3000

def predict(filestr):
    """
    Makes a Basic Pitch prediction with the global parameters above given an input audio file.
    
    Parameters:
        filestr (str): The path to the input audio file.
        
    Returns:
        PrettyMIDI object containing predicted MIDI notes.
    """
    # Run prediction
    model_output, midi_data, note_events = predict(
        filestr,
        BASIC_PITCH_MODEL,
        STD_ONSET,
        STD_FRAME,
        STD_MIN_NOTE_LEN,
        STD_MIN_FREQ,
        STD_MAX_FREQ
    ) # midi_data is the PrettyMIDI object corresponding to the prediction
    return midi_data

with open('./pickles/matched_midi.pkl', 'rb') as f:
    matched_midi_df = pickle.load(f)

with open('./pickles/labeled_features.pkl', 'rb') as f:
    labeled_features = pickle.load(f)

with open('./pickles/my_model.pkl', 'rb') as f:
    model = pickle.load(f)

def get_genres(path):
    """
    Stores the genre labels into a pandas data frame.
    
    Parameters:
        path (str): The path to the genre label file.
        
    Returns:
        pandas.DataFrame: A data frame containing the genres and MIDI IDs.
    """
    ids = []
    genres = []
    
    with open(path) as f:
        for line in f:
            # Skip lines starting with '#'
            if not line.startswith('#'):
                # Splits the line by the tab character ('\t') and unpacks the resulting values 
                # into variables x and y. The strip() function removes leading and trailing whitespace from the line.
                x, y, *_ = line.strip().split("\t")
                # Appends the value of x (track ID) to the ids list.
                ids.append(x)
                # Appends the value of y (genre) to the genres list.
                genres.append(y)
    
    # Constructs a data frame with two columns, "Genre" and "TrackID", using a dictionary. 
    # The "Genre" column contains the genres stored in the genres list, and the "TrackID" column 
    # contains the track IDs stored in the ids list.
    genre_df = pd.DataFrame(data={"Genre": genres, "TrackID": ids})

    return genre_df

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

def get_features(midi_obj):
    """
    Extracts specific features from a PrettyMIDI object given its path using the pretty_midi library.
    Handle any potential errors with MIDI files appropriately.

    Parameters:
        midi_obj: the PrettyMIDI object

    Returns:
        list of float: The extracted features.
    """
    
    # tempo: the estimated tempo of the audio file
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

    return normalize_features([tempo, num_sig_changes, resolution, ts_1,
                    ts_2, melody_complexity, melody_range] + list(pitch_class_hist)) # + list(melody_contour))
    
# genre_path: path of the unzipped "CD1" file
genre_path = "./datasets/msd_tagtraum_cd1.cls"
# creates the genres data frame
genre_df = get_genres(genre_path)

# Get unique genre labels
label_list = list(set(genre_df.Genre))

# Create a dictionary mapping genre labels to their index
label_dict = {lbl: label_list.index(lbl) for lbl in label_list}

midi_path = "test.mid"
midi_features = np.asarray(get_features(midi_path))
midi_features = np.expand_dims(midi_features, axis = 0)

prediction = model.predict(midi_features)

print(prediction)

genre = np.argmax(prediction)
print(label_list[genre])

with open('./pickles/my_model.pkl', 'rb') as f:
    model = pickle.load(f)
