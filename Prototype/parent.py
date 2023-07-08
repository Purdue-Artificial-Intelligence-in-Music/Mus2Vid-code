import sys
from utils.midi import *
from utils.diffusion import *
from utils.prompting import *
from utils.upscaling import *
model = tf.keras.models.load_model('utils\model.h5')

def generate_picture(midi_path, image_name):
    midi_obj = predict(midi_path)
    midi_features = get_features(midi_obj)

    subgenre_num = model.predict(midi_features)
    subgenre = get_subgenre(np.argmax(subgenre_num))
    print(subgenre)

    prompt = get_prompt(subgenre)
    print(prompt)

    image = get_pic(prompt)
    display_images(image)
    image.images[0].save(image_name)
    upscale_image(image_name,image_name)

def main():
    # generate_picture(sys.argv[1], sys.argv[2])
    generate_picture('test_mp3/blue_danube.mp3', 'image.png')
    

if __name__ == "__main__":
    main()