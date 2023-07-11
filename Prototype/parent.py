import sys
from utils.midi import *
from utils.diffusion import *
from utils.prompting import *
from utils.upscaling import *
model = tf.keras.models.load_model('utils\model.h5')

def generate_picture(audio, image_name):
    audio_features = get_features(audio)

    subgenre_num = model.predict(audio_features)
    subgenre = get_subgenre(np.argmax(subgenre_num))

    prompt = get_prompt(subgenre)

    image = get_pic(prompt)

    upscaled_img = upscale_image(image.images[0])
    display_images(upscaled_img)
    upscaled_img[0].save(image_name)

def main():
    # generate_picture(sys.argv[1], sys.argv[2])
    generate_picture('test_mp3/blue_danube.mp3', 'image.png')
    

if __name__ == "__main__":
    main()