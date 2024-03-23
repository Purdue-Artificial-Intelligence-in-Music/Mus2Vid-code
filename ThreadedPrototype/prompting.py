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
    actions = ""
    action = ""

    if emotion_input == "surprised":
        word = "surprise and astonishment"
        lighting = "bright areas with dark shadows for lighting"
        color = "vibrant and bold colors"
        actions = ["group of men with hand to their chest looking at each other astonished","a conductor leading an orchestra with his hands raised","a group of women gasping in awe"]
        action = random.choice(actions)
    if emotion_input == "excited":
        word = "excitement, restlessness, and agitation"
        lighting = "lively atmosphere, energetic lighting"
        color = "vibrant and saturated colors"
        actions = ["a group of men and women clapping along to the music", "a group of women and men dancing in a big hall", "an orchestra playing in a big concert hall filled with a large audience"]
        action = random.choice(actions)
    if emotion_input == "joyous":
        word = "joy and pleasure"
        lighting = "soft and diffused lighting"
        color = "pastel shades, accents of vibrant colors"
        actions = ["women and men dancing with their hands raised", "group of five men sitting at a table eating a big feast and a group of five men play instruments next to the table", "girls running by a georgian colonial house"]
        action = random.choice(actions)
    if emotion_input == "happy":
        word = "happiness"
        lighting = "sun lighting at golden hour"
        color = "warm and bright colors"
        actions = ["women and men dancing with their hands raised", "group of five men sitting at a table eating a big feast and a group of five men play instruments next to the table", "girls running by a georgian colonial house"]
        action = random.choice(actions)
    if emotion_input == "content":
        word = "content"
        lighting = "soft and warm lighting"
        color = "neutral and earthy tones"
        actions = ["two men writing a composition and another man sitting by the harpsichord", "child playing violin for his teacher","music class"]
        action = random.choice(actions)

    if emotion_input == "relaxed":
        word = "relaxation and tranquility"
        lighting = "soft, low intensity lighting"
        color = "cool and muted tones"
        actions = ["women sitting in a field of flowers playing the cello","man and women walking in a garden","a women sleeping on the bed and a man playing the cello next to the bed"]
        action = random.choice(actions)

    if emotion_input == "calm":
        word = "calm, peace"
        lighting = "warm, dim, ambient lighting"
        color = "cool and muted tones, pastel shades"
        actions = ["women sitting in a field of flowers playing the cello","man and women walking in a garden","a women sleeping on the bed and a man playing the cello next to the bed"]
        action = random.choice(actions)
    if emotion_input == "sleepy":
        word = "sleepiness and relaxation"
        lighting = "moon lighting"
        color = "subdued, warm neutrals"
        actions = ["women sitting in a field of flowers playing the cello","man and women walking in a garden","a women sleeping on the bed and a man playing the cello next to the bed"]
        action = random.choice(actions)

    if emotion_input == "bored":
        word = "boredom"
        lighting = "flat, uniform lighting"
        color = "dull colors"
        actions = ["an empty music hall filled with chairs and instruments","showcasing a lone musician absentmindedly tapping their fingers on a violin,the instrument resting limply against his shoulder as he gazes off into the distance with a vacant expression", "a lone figure sits on a bench, absentmindedly tossing pebbles into a nearby pond, her expression distant and disinterested"]
        action = random.choice(actions)
    if emotion_input == "sad":
        word = "sadness and melancholy"
        lighting = "shadows and low lighting"
        color = "dark, deep colors"
        actions = ["portraying a lone violinist with bowed head and slumped shoulders, her melancholic melody filling the empty concert hall with a haunting lament", "capturing a pianist seated at the grand piano, his fingers hesitantly hovering over the keys, eyes cast downward in sorrow as he plays a mournful melody that echoes through the empty hall", "depicting an empty stage with scattered sheet music, abandoned instruments, and dimly lit candles, evoking a sense of desolation and loss in the absence of performers and audience alike"]
        action = random.choice(actions)
    if emotion_input == "depressed":
        word = "depression and woefulness"
        lighting = "harsh shadows and minimal lighting"
        color = "dark desaturated tones"
        actions = ["portraying a solitary figure slouched over the piano, his hands resting on the keys without playing, the weight of despair evident in their hunched posture and downcast gaze", "showing a lone cellist seated on a chair, their instrument lying neglected beside them, while they stare blankly into the distance, lost in the depths of their melancholy", "portraying an empty stage with a single spotlight illuminating a solitary violin, its strings untouched and the bow lying abandoned nearby,symbolizing the silent despair of a musician overwhelmed by sadness"]
        action = random.choice(actions)
    if emotion_input == "distressed":
        word = "distress and agitation"
        lighting = "flickering and unstable lighting"
        color = "dark, intense colors"
        actions = ["depicting a conductor frantically waving their baton amidst a dissonant cacophony of instruments, their expression fraught with anxiety as they struggle to regain control of the chaotic performance","capturing a violinist onstage, their fingers gripping the instrument tightly as they play with frenzied intensity, their furrowed brow and clenched jaw betraying the turmoil within as the haunting notes pierce the air", "portraying a conductor frantically gesturing amidst scattered sheet music and overturned chairs,the chaos reflecting their inner turmoil"]
        action = random.choice(actions)
    if emotion_input == "angry":
        word = "anger and aggressiveness"
        lighting = "harsh and intense lighting"
        color = "high contrast color combinations"
        actions = ["depicting a conductor slamming their baton onto the podium, their face contorted with rage as they command the orchestra with forceful gestures", "showing a musician furiously tearing their sheet music apart, their instrument abandoned as they storm off the stage in a fit of rage", "featuring a conductor glaring fiercely at the orchestra, their clenched fists and tense posture conveying their simmering fury"]
        action = random.choice(actions)
    if emotion_input == "afraid":
        word = "fear and distress"
        lighting = "dim and eerie lighting"
        color = "dark and muted hues"
        actions = ["depicting a lone violinist frozen mid-performance, their hands trembling and eyes darting nervously as they struggle to maintain composure", "portraying a hushed audience, their faces etched with worry as they exchange fearful whispers and steal nervous glances at the empty stage, anticipation of the unknown filling that air","depicting a lone figure in the corner of the music hall, their eyes wide with fear as they clutch onto their instrument, the eerie silence echoing their apprehension"]
        action = random.choice(actions)
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
    emotions, lighting, colors, action = get_modifiers(emotion)
    prompt_one = subject + "," + emotions + "," + lighting + "," + colors + "," + action
    prompt_two = subject + "of" + emotion + "emotion," + lighting + "," + colors + "," + action
    prompt_three = "generate an artwork inspired by" + subject + ",convey a sense of" + emotion + "emotion," + "use exclusively" + lighting + ", utilize a color palette dominated by" + colors + "," + action
    prompts = [prompt_one, prompt_two,prompt_three]
    prompt = random.choice(prompts)
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
        self.prompt = None
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
                self.prompt = "Blank screen"
            else:
                self.prompt = get_prompt(self.genre_thread.genre_output, self.emotion_thread.emo_values[0],
                                           self.emotion_thread.emo_values[1])
            time.sleep(0.2)
