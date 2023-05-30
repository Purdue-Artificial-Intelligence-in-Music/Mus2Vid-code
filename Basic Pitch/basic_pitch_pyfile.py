# imports
import tensorflow as tf
import pretty_midi as pm
import matplotlib.pyplot as plt
import mido as m
import numpy as np
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
BASIC_PITCH_MODEL = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

# Parameters
STD_ONSET = 0.3
STD_FRAME = 0.2
STD_MIN_NOTE_LEN = 50
STD_MIN_FREQ = None
STD_MAX_FREQ = 3000

# Grab the file path
filestr = input("Enter the file path to the mp3 file to convert to video")

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