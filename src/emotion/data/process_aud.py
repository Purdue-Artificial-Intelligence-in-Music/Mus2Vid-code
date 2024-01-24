import os
import subprocess
from utils.util import RAW_ANNOTATIONS_DIR, RAW_AUDIO_DIR, PROCESSED_ANNOTATIONS_DIR, PROCESSED_AUDIO_DIR, \
    ANNOTATIONS_FILE, TEMP_AUDIO_DIR
import sys
sys.path.insert(0, "C:\\Users\\TPNml\\Documents\\GitHub\\Mus2Vid-code\\data\\scripts")
from aberration import Aberrator


def process_audio(abb=False) -> None:
    """Process raw audio files for use with emotion module.

    Take .mp3 audio files, convert them to .wav files,
    and save them in a separate directory.

    Returns
    -------
    None
    """
    if not os.path.exists(PROCESSED_AUDIO_DIR):
        os.mkdir(PROCESSED_AUDIO_DIR)


    if abb and (not os.path.exists(TEMP_AUDIO_DIR)):
        os.mkdir(TEMP_AUDIO_DIR)

    for filename in os.listdir(RAW_AUDIO_DIR):
        print(filename)
        root, ext = os.path.splitext(filename)
        if abb:
            if ext == ".mp3" and not os.path.exists(f"{TEMP_AUDIO_DIR}/{root}.wav"):
                subprocess.run([
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    f"{RAW_AUDIO_DIR}/{root}.mp3",
                    "-ar",
                    "44100",
                    f"{TEMP_AUDIO_DIR}/{root}.wav",
                ])
        else:
            if ext == ".mp3" and not os.path.exists(f"{PROCESSED_AUDIO_DIR}/{root}.wav"):
                subprocess.run([
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    f"{RAW_AUDIO_DIR}/{root}.mp3",
                    "-ar",
                    "44100",
                    f"{PROCESSED_AUDIO_DIR}/{root}.wav",
                ])
    if abb:
        abb = Aberrator()
        abb.modify_files(TEMP_AUDIO_DIR + "/", PROCESSED_AUDIO_DIR + "2/", dumb_name=True)


if __name__ == "__main__":
    process_audio(abb=True)
