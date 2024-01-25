import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import threading
import time
import random

## take float values and output emotion word
## values based on data csv file
## Some emotions based on VA model is not on there so I used the closest synonym
def get_emotion_from_values(arousal, valence):
    thresholds = {
        (5.74, 4.09): "surprised",
        (7.22, 6.37): "excited",  ## used jumpy
        (4.55, 2): "joyous",
        (5.98, 3.98): "happy",  ## used Jaunty
        (2.05, 2.28): "content",
        (1.72, 2.37): "relaxed",
        (3.4, 2.34): "calm",  ## used carefree
        (5.47, 6.03): "sleepy",  ## used wistful
        (4.04, 5.83): "bored",
        (5.91, 6.6): "sad",
        (6.26, 6.82): "depressed",  ## used grievous
        (6.43, 5.77): "distressed",  ## used troubled
        (8.06, 7.5): "angry",
        (7.2, 6.47): "afraid",  ## used aghast
    }

    closest_distance = float("inf")
    closest_emotion = None

    for threshold, emotion in thresholds.items():
        arousal_threshold, valence_threshold = threshold
        distance = abs(arousal - arousal_threshold) + abs(valence - valence_threshold)
        if distance < closest_distance:
            closest_distance = distance
            closest_emotion = emotion

    return closest_emotion

## use the emotions on the VA model
## output text for the different emotions for overall prompt generation
def get_modifiers(emotion_input):
    word = ""
    lighting = ""
    color = ""

    if emotion_input == "surprised":
        word = "surprise and astonishment"
        lighting = "bright areas with dark shadows for lighting"
        color = "vibrant and bold colors"

    if emotion_input == "excited":
        word = "excitement, restlessness, and agitation"
        lighting = "lively atmosphere, energetic lighting"
        color = "vibrant and saturated colors"

    if emotion_input == "joyous":
        word = "joy and pleasure"
        lighting = "soft and diffused lighting"
        color = "pastel shades, accents of vibrant colors"

    if emotion_input == "happy":
        word = "happiness"
        lighting = "sun lighting at golden hour"
        color = "warm and bright colors"

    if emotion_input == "content":
        word = "content"
        lighting = "soft and warm lighting"
        color = "neutral and earthy tones"

    if emotion_input == "relaxed":
        word = "relaxation and tranquility"
        lighting = "soft, low intensity lighting"
        color = "cool and muted tones"

    if emotion_input == "calm":
        word = "calm, peace"
        lighting = "warm, dim, ambient lighting"
        color = "cool and muted tones, pastel shades"

    if emotion_input == "sleepy":
        word = "sleepiness and relaxation"
        lighting = "moon lighting"
        color = "subdued, warm neutrals"

    if emotion_input == "bored":
        word = "boredom"
        lighting = "flat, uniform lighting"
        color = "dull colors"

    if emotion_input == "sad":
        word = "sadness and melancholy"
        lighting = "shadows and low lighting"
        color = "dark, deep colors"

    if emotion_input == "depressed":
        word = "depression and woefulness"
        lighting = "harsh shadows and minimal lighting"
        color = "dark desaturated tones"

    if emotion_input == "distressed":
        word = "distress and agitation"
        lighting = "flickering and unstable lighting"
        color = "dark, intense colors"

    if emotion_input == "angry":
        word = "anger and aggressiveness"
        lighting = "harsh and intense lighting"
        color = "high contrast color combinations"

    if emotion_input == "afraid":
        word = "fear and distress"
        lighting = "dim and eerie lighting"
        color = "dark and muted hues"

    return word, lighting, color

## function for different type of genre with date
def get_subject(subgenre):
    prompt = ""
    if subgenre == "Baroque":
        prompt = "Baroque Classical Music, from the years 1600-1750."
    elif subgenre == "Classical":
        prompt = "Classical music, from the years 1750-1830."
    elif subgenre == "Romantic":
        prompt = "Romantic Classical Music, from the years 1830-1920."
    elif subgenre == "20th Century":
        prompt = "Modern Classical Music, from the years 1900-2000."

    return prompt


## connect all the text into one prompt to send to SD
def get_prompt(subgenre, valence, arousal):
    subject = get_subject(subgenre)
    emotion = get_emotion_from_values(arousal, valence)
    emotions, lighting, colors = get_modifiers(emotion)
    prompt = "Generate an artwork inspired by " + subject + "Convey a sense of " + emotions + ". Use exclusively " + lighting + ". Utilize a color pallete dominated by " + colors + "."
    return prompt

"""
This class is a thread class that generates prompts procedurally in real time.
"""


class PromptGenerationThread(threading.Thread):
    """
    This function is called when a PromptGenerationThread is created.
    Parameters:
        name: the name of the thread
    Returns: nothing
    """

    def __init__(self, name, genre_thread, emotion_thread, audio_thread):
        super(PromptGenerationThread, self).__init__()
        self.name = name
        self.prompt = "Black screen"
        self.genre_thread = genre_thread
        self.emotion_thread = emotion_thread
        self.audio_thread = audio_thread
        self.stop_request = False

    """
    When the thread is started, this function is called which repeatedly generates new prompts.
    Parameters: nothing
    Returns: nothing
    """

    def run(self):
        while not self.stop_request:
            if (not self.audio_thread.input_on or
                    (self.genre_thread is None or self.genre_thread.genre_output is None or
                     self.emotion_thread is None or self.emotion_thread.emo_values is None)):
                self.prompt = "Black screen"
            else:
                self.prompt = get_prompt(self.genre_thread.genre_output, self.emotion_thread.emo_values[0],
                                           self.emotion_thread.emo_values[1])
            time.sleep(0.2)
