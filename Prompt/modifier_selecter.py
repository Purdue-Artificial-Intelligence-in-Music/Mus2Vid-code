import random

def perspectiveRandom():
    angle_modifier = ["extreme close-up", "close-up", "medium shot", "waist and torso shot", "long shot", "wide shot", "full body", "extreme long shot", "from a distance"]
    lens_modifier = ["fast shutter speed", "1/1000 sec shutter","slow shutter speed", "long exposure","tilt shift photography", "macro lens", "macro photo", "Sigma 105mm f/2.8", "wide angle lens", "ultra-wide shot", "4k"]
    artstyle_modifier = ["woodcut painting", "charcoal sketch","watercolor","acrylic on canvas","colored pencil", "oil painting", "airbrush", "digital painting","low poly,unreal engine,Blender render","isometric 3D, highest quality render"]


    angle_random = random.choice(angle_modifier)
    lens_random = random.choice(lens_modifier)
    artstyle_random = random.choice(artstyle_modifier)


    return [angle_random, lens_random, artstyle_random]

random_modifiers = perspectiveRandom()
print(random_modifiers)
