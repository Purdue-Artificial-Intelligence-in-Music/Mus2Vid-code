import sys
from utils.midi import *
from utils.diffusion import *
from utils.prompting import *
model = tf.keras.models.load_model('utils\model.h5')

def generate_picture(midi_path, image_name):
    midi_obj = predict(midi_path)
    midi_features = get_features(midi_obj)
    print(midi_features.shape)

    subgenre_num = model.predict(midi_features)
    print(subgenre_num)
    subgenre = get_genre(np.argmax(subgenre_num))
    print(subgenre)

    prompt = get_prompt(subgenre)
    print (prompt)

    image = get_pic(prompt)
    display_images(image)
    image.images[0].save(image_name)

def main():
    # generate_picture(sys.argv[1], sys.argv[2])
    generate_picture('..\Genre_NN\maestro-v3.0.0-wav\maestro-v3.0.0\\2004\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav', 'image.png')
    

if __name__ == "__main__":
    main()