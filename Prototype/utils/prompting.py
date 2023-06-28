import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

## use the emotions on the VA model
## output text for the different emotions for overall prompt generation
def get_emotion(emotion_input):
    emotion_prompt = []
    if emotion_input == "surprised":
        word = "surprised,astonished"
        lighting = "bright areas with dark shadows"
        color = "vibrant and bold colors"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "excited":
        word = "excited,restless,agitated"
        lighting = "lively atmosphere,energetic lighting"
        color = "vibrant and saturated colors"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "joyous":
        word = "joyous"
        lighting = "soft and diffused lighting"
        color = "pastel shades, accents of vibrant colors"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "happy":
        word = "happy"
        lighting = "sun lighting,golden hour"
        color = "warm and bright colors"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "content":
        word = "content"
        lighting = "soft and warm lighting"
        color = "neutral and earthy tones"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "relaxed":
        word = "relaxed,tranquil"
        lighting = "soft lighting,low intensity"
        color = "cool and muted tones"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "calm":
        word = "calm,peaceful"
        lighting = "ambient lighting,warm and dim"
        color = "cool and muted tones,pastel shades"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "sleepy":
        word = "sleepy"
        lighting = "moon lighting"
        color = "subdued,warm neutrals"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "bored":
        word = "bored"
        lighting = "flat,uniform lighting"
        color = "dull colors"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "sad":
        word = "sad,melancholy"
        lighting = "shadows and low lighting"
        color = "dark,deep colors"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "depressed":
        word = "depressed"
        lighting = "harsh shadows,minimal illumination"
        color = "dark desaturated tones"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "distressed":
        word = "distressed"
        lighting = "flickering and unstable lighting"
        color = "dark,intense colors"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "angry":
        word = "angry,aggressive"
        lighting = "harsh and intense lighting"
        color = "high contrast color combinations"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "afraid":
        word = "afraid,fearful"
        lighting = "dim and eerie lighting"
        color = "dark and muted hues"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)

    return emotion_prompt

def perspectiveRandom():
    modifiers = []
    angle_modifier = ["extreme close-up", "close-up", "medium shot,waist and torso shot", "long shot, wide shot, full body", "extreme long shot, from a distance"]
    lens_modifier = ["fast shutter speed, 1/1000 sec shutter","slow shutter speed, long exposure","tilt shift photography", "macro lens, macro photo,Sigma 105mm f/2.8", "wide angle lens, 15mm, ultra-wide shot", "4k"]
    artstyle_modifier = ["woodcut painting", "charcoal sketch","watercolor","acrylic on canvas","colored pencil,detailed","oil painting","airbrush", "digital painting","low poly, unreal engine,Blender render","isometric 3D, highest quality render","claymation"]
    scale_modifier = ["grand","high-scale","monumental","imposing","delicate","elegant","intricate","decorative","opulent","ornamental"]

    angle_random = random.choice(angle_modifier)
    lens_random = random.choice(lens_modifier)
    artstyle_random = random.choice(artstyle_modifier)
    scale_random = random.choice(scale_modifier)
    
    modifiers.append(angle_random)
    modifiers.append(lens_random)
    modifiers.append(artstyle_random)
    modifiers.append(scale_random)
    return modifiers

## function for different type of genre with date
def get_genre(subgenre):
    prompt = ''
    if subgenre == 'Baroque':
        prompt = 'Baroque Classical Music, 1600-1750'
    elif subgenre ==  'Classical':
        prompt = 'Classical music, 1750-1830'
    elif subgenre == 'Romantic':
        prompt = 'Romantic Classical Music, 1830-1920'
    elif subgenre ==  '20th Century':
        prompt = 'Classical Music, 1900-2000'

    return prompt

## connect all the text into one prompt to send to SD 

def get_prompt(subgenre): # add valence and arousal eventually
    modify = perspectiveRandom()
    genre = get_genre(subgenre)
    emotion = get_emotion("sleepy")
    result = [genre,emotion,modify]
    modify_str = [', '.join(item) if isinstance(item, list) else item for item in result]
    prompt = ', '.join(str(item) for item in modify_str)
    return prompt