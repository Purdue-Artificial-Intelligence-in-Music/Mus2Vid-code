import sys
from utils.midi import *
from utils.diffusion import *
from utils.prompting import *

def generate_picture(midi_path, image_name):
    midi_obj = predict(midi_path)
    midi_features = get_features(midi_obj)
    print(midi_features.shape)

    model = tf.keras.models.load_model('utils\_best_model.h5')
    subgenre_num = np.argmax(model.predict(midi_features))
    subgenre = get_genre(subgenre_num)
    print(subgenre)

    prompt = get_prompt(subgenre)
    print(prompt)

    image = get_pic(prompt)
    display_images(image)
    image.images[0].save(image_name)

def main():
    # generate_picture(sys.argv[1], sys.argv[2])
    generate_picture('test_mp3/moonlight_sonata.mp3', 'image.png')

if __name__ == "__main__":
    main()