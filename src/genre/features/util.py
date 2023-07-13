from basic_pitch.inference import predict as bp_predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import tensorflow as tf
import pandas as pd
import pretty_midi as pm
BASIC_PITCH_MODEL = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

RAW_AUDIO_DIR = "./data/raw/maestro-v3.0.0/"
PROCESSED_AUDIO_DIR = "./data/processed/maestro/audio"
ANNOTATIONS_FILE = "./data/raw/maestro-v3.0.0/maestro-annotations.csv"
INTERIM_DATA_DIR = "./data/interim/genre/"
SELECTOR_EXT = ".selector"

def predict(filestr):
    """Makes a Basic Pitch prediction with the global parameters above given an input audio file.

    Parameters
    ----------
    filestr: string
        The path to the input audio file.

    Returns
    -------
    midi_data: pm.PrettyMIDI
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

def get_matched_midi(audio, genres) -> None:
    """Creates pretty-midi objects from input .wav files
    Matches each MIDI object with a composition year based on the years list
    Saves results to interim data folder

    Parameters
    ----------
    audio: list
        list of audio filepaths.
    genres : list
        list of genres for each .wav file
    """
    prettys = [] # list of pretty_midi objects

    for i in range(len(audio)):
        midi_obj = predict(PROCESSED_AUDIO_DIR + audio[i]) # uses basic pitch to convert wav to MIDI
        prettys.append(midi_obj)

    df = pd.DataFrame({'midi_obj': prettys, 'genre': genres})

    with lzma.open(f"{INTERIM_DATA_DIR}matched_midi.xz", "wb") as f:
        pickle.dump(matched_midi_df, f)

    return

if __name__ == "__main__":
    pass
