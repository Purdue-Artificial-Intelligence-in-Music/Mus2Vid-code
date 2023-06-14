

def get_prompt(subgenre):
    prompt = ''

    match subgenre:
        case 'Baroque':
            prompt = 'Baroque Classical Music, 1600-1750, close-up, tilt shift photography, oil painting, grand, sad, melancholy, shadows and low lighting, dark and deep colors'
        case 'Classical':
            prompt = 'Classical music, 1750-1830, single subject, close up, detailed face, muted colors, female subject in ballroom, dramatic lighting, ornate'
        case 'Romantic':
            prompt = 'Romantic Classical Music, 1830-1920, extreme long shot, extreme wide shot, from a distance, Fast shutter speed, 1/1000 sec shutter, charcoal sketch, ornate, calm, peaceful, ambient lighting, cool and muted tones'
        case '20th Century':
            prompt = 'Classical Music, 1900-2000, waist and torso shot, Slow shutter speed, long exposure, digital painting, abstract, excited, restless, agitated, vibrant and lively atmosphere, bright lightning, vibrant and saturated colors'

    return prompt