import pickle
import torch
import pretty_midi
import pandas as pd
import numpy as np
import tensorflow as tf
from basic_pitch.inference import predict as bp_predict
from basic_pitch import ICASSP_2022_MODEL_PATH
BASIC_PITCH_MODEL = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
from diffusers import StableDiffusionPipeline
stable_diffusion_model_id = "runwayml/stable-diffusion-v1-5"
# from ..Basic_Pitch.chordfuncs import *


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

midi_path = 'Prototype/blind comp render E-PIANO ONLY.mp3'
midi_object = predict(midi_path)
midi_features = np.asarray(get_features(midi_object))
midi_features = np.expand_dims(midi_features, axis = 0)

model = tf.keras.models.load_model('Max NN\my_model.h5')

comp_year = model.predict(midi_features)

print(comp_year)

prompt = ''
if (comp_year < 1400):
    prompt = 'A painting in the style of Giotto di Bondone of a musician in an italian town square'
elif (comp_year < 1600):
    prompt = 'A painting in the style of Leonardo Da Vinci of a lute player performing for the royal family'
elif (comp_year < 1750):
    prompt = 'A painting in the style of Johannes Vermeer of the girl with a pearl earing playing piano'
elif (comp_year < 1830):
    prompt = 'A painting in the style of Jacque-Louis David of two knights dueling in the French country'
elif (comp_year < 1920):
    prompt = 'A painting in the style of Claude Monet of a peaceful koi pond in a park in Argentina'
else:
    prompt = 'A painting in the style of Andy Worhol of a cello player on stage'

print(prompt)

pipe = StableDiffusionPipeline.from_pretrained(stable_diffusion_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def get_pic(prompt, inference = 50, guidance_scale = 7.5, num_images_per_prompt = 3, seed = 0):
    """
    Creates an image using stable diffusion pipeline
    Parameters:
        prompt: string of the prompt
        inference: number of inference step, around 50 for a high quality image
        guidance scale: a way to increase the adherence to the conditional signal that guides the generation (text, in this case) as well as overall sample quality
    Returns:
        output: stable diffusion pipeline output
    """
    generator_list = []
    for i in range(num_images_per_prompt):
        generator_list.append(torch.Generator("cuda").manual_seed(seed + i))

    return pipe(
        prompt,
        num_inference_steps = inference,
        guidance_scale = guidance_scale,
        num_images_per_prompt = num_images_per_prompt,
        generator = generator_list
    )

def display_images(pipe):
    for i in range(len(pipe[0])):
        image = pipe.images[i]
        image.show()
        
image = get_pic(prompt)
display_images(image)