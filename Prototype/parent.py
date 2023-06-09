import sys
from utils.midi import *
from utils.diffusion import *
from utils.prompting import *

def generate_picture(midi_path, image_name):
    midi_features = get_features(midi_path)

    model = tf.keras.models.load_model('utils\my_model.h5')
    comp_year = model.predict(midi_features)

    prompt = get_prompt(comp_year)

    image = get_pic(prompt)
    display_images(image)

def main():
    generate_picture(sys.argv[1], sys.argv[2])
