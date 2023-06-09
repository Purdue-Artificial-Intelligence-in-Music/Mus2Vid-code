from utils.midi import *
from utils.diffusion import *
from utils.prompting import *

def main():
    midi_path = 'blind comp render E-PIANO ONLY.mp3'
    midi_features = get_features(midi_path)
    model = tf.keras.models.load_model('utils\my_model.h5')
    comp_year = model.predict(midi_features)
    prompt = get_prompt(comp_year)
    image = get_pic(prompt)
    display_images(image)

if __name__ == "__main__":
    main()