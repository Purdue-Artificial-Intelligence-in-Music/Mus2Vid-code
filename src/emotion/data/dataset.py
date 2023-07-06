import os
import subprocess


RAW_ANNOTATIONS_DIR = "./data/raw/deam_dataset/DEAM_Annotations/annotations/annotations averaged per song/song_level"
RAW_AUDIO_DIR = "./data/raw/deam_dataset/DEAM_audio/MEMD_audio"
RAW_FEATURES_DIR = "./data/raw/deam_dataset/features/features"
RAW_METADATA_DIR = "./data/raw/deam_dataset/metadata/metadata"
PROCESSED_ANNOTATIONS_DIR = "./data/processed/annotations"
PROCESSED_AUDIO_DIR = "./data/processed/audio"
ANNOTATIONS_FILE = "static_annotations_averaged_songs_1_2000.csv"


def process_audio():
    if not os.path.exists(PROCESSED_AUDIO_DIR):
        os.mkdir(PROCESSED_AUDIO_DIR)

    for filename in os.listdir(RAW_AUDIO_DIR):
        root, ext = os.path.splitext(filename)
        if ext == ".mp3" and not os.path.exists(f"{PROCESSED_AUDIO_DIR}/{root}.wav"):
            subprocess.run([
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                f"{RAW_AUDIO_DIR}/{root}.mp3",
                f"{PROCESSED_AUDIO_DIR}/{root}.wav",
            ])


def process_annotations():
    if not os.path.exists(PROCESSED_ANNOTATIONS_DIR):
        os.mkdir(PROCESSED_ANNOTATIONS_DIR)

    with open(f"{RAW_ANNOTATIONS_DIR}/{ANNOTATIONS_FILE}", "r") as f_read:
        lines = f_read.read().replace(" ", "")
        with open(f"{PROCESSED_ANNOTATIONS_DIR}/{ANNOTATIONS_FILE}", "w") as f_write:
            f_write.write(lines)


def process_data():
    process_audio()
    process_annotations()


if __name__ == "__main__":
    process_data()
