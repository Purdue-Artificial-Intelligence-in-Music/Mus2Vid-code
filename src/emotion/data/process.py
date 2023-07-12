import os
import subprocess
import pandas as pd
from src.emotion.data.util import RAW_ANNOTATIONS_DIR, RAW_AUDIO_DIR, PROCESSED_ANNOTATIONS_DIR, PROCESSED_AUDIO_DIR, ANNOTATIONS_FILE


def process_audio() -> None:
    """Process raw audio files for use with emotion module.

    Take .mp3 audio files, convert them to .wav files,
    and save them in a separate directory.

    Returns
    -------
    None
    """
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


def process_annotations() -> None:
    """Process audio annotations for use with emotion module.

    Take raw annotation files and remove extra space from column headers.
    
    Returns
    -------
    None
    """
    if not os.path.exists(PROCESSED_ANNOTATIONS_DIR):
        os.mkdir(PROCESSED_ANNOTATIONS_DIR)

    annotations = pd.read_csv(f"{RAW_ANNOTATIONS_DIR}/{ANNOTATIONS_FILE}")
    annotations = annotations.rename(columns={
        " valence_mean": "valence_mean",
        " valence_std": "valence_std",
        " arousal_mean": "arousal_mean",
        " arousal_std": "arousal_std"
    })
    annotations.to_csv(f"{PROCESSED_ANNOTATIONS_DIR}/{ANNOTATIONS_FILE}", index=False)


if __name__ == "__main__":
    process_audio()
    process_annotations()
