import os
import subprocess


RAW_ANNOTATIONS_PATH = "./data/raw/deam_dataset/DEAM_Annotations/annotations/annotations averaged per song/song_level"
RAW_AUDIO_PATH = "./data/raw/deam_dataset/DEAM_audio/MEMD_audio"
RAW_FEATURES_PATH = "./data/raw/deam_dataset/features/features"
RAW_METADATA_PATH = "./data/raw/deam_dataset/metadata/metadata"

PROCESSED_ANNOTATIONS_PATH = "./data/processed/annotations"
PROCESSED_AUDIO_PATH = "./data/processed/audio"

ANNOTATIONS_FILE = "static_annotations_averaged_songs_1_2000.csv"


def process_audio():
    if not os.path.exists(PROCESSED_AUDIO_PATH):
        os.mkdir(PROCESSED_AUDIO_PATH)

    for filename in os.listdir(RAW_AUDIO_PATH):
        root, ext = os.path.splitext(filename)
        if ext == ".mp3" and not os.path.exists(f"{PROCESSED_AUDIO_PATH}/{root}.wav"):
            subprocess.run([
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                f"{RAW_AUDIO_PATH}/{root}.mp3",
                f"{PROCESSED_AUDIO_PATH}/{root}.wav",
            ])

def process_annotations():
    if not os.path.exists(PROCESSED_ANNOTATIONS_PATH):
        os.mkdir(PROCESSED_ANNOTATIONS_PATH)

    with open(f"{RAW_ANNOTATIONS_PATH}/{ANNOTATIONS_FILE}", "r") as f_read:
        lines = f_read.read().replace(" ", "")
        with open(f"{PROCESSED_ANNOTATIONS_PATH}/{ANNOTATIONS_FILE}", "w") as f_write:
            f_write.write(lines)

def process_data():
    process_audio()
    process_annotations()


if __name__ == "__main__":
    process_data()
