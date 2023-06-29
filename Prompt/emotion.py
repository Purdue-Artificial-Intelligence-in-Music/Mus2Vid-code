## use the emotions on the VA model
## output text for the different emotions for overall prompt generation


## take float values and output emotion word 
## values based on normalized data csv file 
## Some emotions based on VA model is not on there so I used the closest synonym

def get_emotion_from_values(arousal,valence):
    thresholds = {
        (0.2017937219730941,-0.2763358778625955): "surprised",
        (0.6442451420029891,0.41984732824427495): "excited",
        (0.15396113602391637,-0.9145038167938931): "joyous",
        (0.2735426008968609,-0.30992366412213745): "happy",
        (-0.9013452914798207,-0.8290076335877863): "content",
        (-0.9999999999999999,-0.8015267175572519): "relaxed",
        (-0.4977578475336324,-0.8106870229007634): "calm",
        (0.12107623318385641,0.31603053435114514): "sleepy",
        (-0.30642750373692085,0.2549618320610687): "bored",
        (0.2526158445440956,0.49007633587786237): "sad",
        (0.3572496263079221,0.5572519083969467): "depressed",
        (0.4080717488789236,0.23664122137404564): "distressed",
        (0.8953662182361732,0.7648854961832063): "angry",
        (0.638266068759342,0.45038167938931295): "afraid"
        
    }
    
    ## fix this part of code 

    closest_distance = float('inf')
    closest_emotion = None

    for threshold, emotion in thresholds.items():
        arousal_threshold, valence_threshold = threshold
        distance = abs(arousal - arousal_threshold) + abs(valence - valence_threshold)
        if distance < closest_distance:
            closest_distance = distance
            closest_emotion = emotion

    return closest_emotion


## Use emotion word to output text for prompt 
def emotion(emotion_input):
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
        color = "subdued and warm neutrals"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "bored":
        word = "bored"
        lighting = "uniform lighting"
        color = "dull colors"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "sad":
        word = "sad,melancholy"
        lighting = "shadows and low lighting"
        color = "dark and deep colors"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "depressed":
        word = "depressed"
        lighting = "harsh shadows,minimal illumination"
        color = "dark unsaturated tones"
        emotion_prompt.append(word)
        emotion_prompt.append(lighting)
        emotion_prompt.append(color)
    if emotion_input == "distressed":
        word = "distressed"
        lighting = "flickering and unstable lighting"
        color = "dark and intense colors"
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


def emotion_from_values(arousal,valence):
    generated_emotion = get_emotion_from_values(arousal,valence)
    emotion_prompt = emotion(generated_emotion)
    return emotion_prompt


## Example 
##result = emotion_from_values(.2,-.2)
##print(result) 





