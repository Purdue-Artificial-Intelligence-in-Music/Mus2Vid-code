import numpy as np
import scipy
import librosa
import pytsmod as tsm
import os
import pydub

FILE_PATH = "C:\\Users\\TPNml\\Downloads\\test_audio_files\\"
INPUT_EXT = "input\\"
OUTPUT_EXT = "output\\"


class Aberrator():
    def __init__(self,
                 pitch_range=1,
                 time_range=0.2,
                 noise_gain_range=0.07,
                 min_noise=0.02,
                 num_variations=5,
                 length_of_output_audio=1,
                 max_sample_length=30,
                 file_format="mp3"
                 ):
        self.PITCH_RANGE = pitch_range
        self.TIME_RANGE = time_range
        self.NOISE_GAIN_RANGE = noise_gain_range
        self.MIN_NOISE = min_noise
        self.NUM_VARIATIONS = num_variations
        self.LENGTH_OF_OUTPUT_AUDIO = length_of_output_audio
        self.MAX_SAMPLE_LENGTH = max_sample_length
        self.FORMAT = file_format

    def aberrate(self, y, sr):
        pitch_factor = pow(2, (np.random.rand(1)[0] * self.PITCH_RANGE * 2 - self.PITCH_RANGE) / 12.0)
        time_factor = float(np.random.rand(1, )[0]) * self.TIME_RANGE * 2 + (1 - self.TIME_RANGE)
        noise_gain = float(np.random.rand(1, )[0]) * self.NOISE_GAIN_RANGE + self.MIN_NOISE

        y_out = tsm.wsola(y, pitch_factor / time_factor)

        noise = np.random.rand(len(y_out))
        noise = scipy.stats.norm.cdf(20 * noise - 10) * noise_gain

        y_out *= (1 - noise_gain)
        y_out += noise

        sr_out = sr * pitch_factor
        sr_out = int(sr_out)

        return y_out, sr_out, pitch_factor, time_factor, noise_gain

    def modify_files(self, file_path, input_ext, output_ext):
        for file in os.listdir(file_path + input_ext):
            file_parts = file.split(".")
            name = file[0:len(file) - len(file_parts[len(file_parts) - 1]) - 1]
            y, sr = librosa.load(file_path + input_ext + file, sr=44100)

            for j in range(self.NUM_VARIATIONS):
                print("Iteration", j)
                y_out, sr_out, pitch_factor, time_factor, noise_gain = self.aberrate(y, sr)
                y_out *= 2 ** 15
                y_out = y_out.astype(np.int16)

                print(np.shape(y_out))

                for i in range(0, np.floor_divide(np.shape(y_out)[0], sr_out * self.LENGTH_OF_OUTPUT_AUDIO * 2)):
                    if i * self.LENGTH_OF_OUTPUT_AUDIO > self.MAX_SAMPLE_LENGTH:
                        break
                    output = y_out[sr_out * i * self.LENGTH_OF_OUTPUT_AUDIO:
                                   sr_out * (i + 1) * self.LENGTH_OF_OUTPUT_AUDIO]
                    song = pydub.AudioSegment(output.tobytes(), frame_rate=sr_out, sample_width=2, channels=1)
                    song.export(file_path + output_ext + name + "###" + f"(%.3f,%.3f,%.3f),(%d,%d).mp3" %
                                (pitch_factor, time_factor, noise_gain, i * self.LENGTH_OF_OUTPUT_AUDIO,
                                 (i + 1) * self.LENGTH_OF_OUTPUT_AUDIO),
                                format=file_format)


def main():
    ab = Aberrator()
    ab.modify_files(FILE_PATH, INPUT_EXT, OUTPUT_EXT)


if __name__ == "main":
    main()
