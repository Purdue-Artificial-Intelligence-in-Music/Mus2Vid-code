import pandas as pd


ANNOTATIONS_PATH = "./data/processed/annotations/static_annotations_averaged_songs_1_2000.csv"
AUDIO_DIR = "./data/processed/audio"
FEATURES_DIR = "./data/interim/features"
FEATURES_EXT = "features"
SELECTOR_EXT = "selector"
CHUNK_SIZE = 10


def get_audio_filepaths() -> list[str]:
    """Return a list of audio filepaths relative to repository root."""
    song_id_list = pd.read_csv(ANNOTATIONS_PATH)["song_id"]

    audio_filepaths = []
    for song_id in song_id_list:
        audio_filepaths.append(f"{AUDIO_DIR}/{song_id}.mp3")

    return audio_filepaths


def get_valence_targets() -> pd.Series:
    """Return a pandas.Series of target values for valence."""
    return pd.read_csv(ANNOTATIONS_PATH)["valence_mean"]


def get_arousal_targets() -> pd.Series:
    """Return a pandas.Series of target values for arousal."""
    return pd.read_csv(ANNOTATIONS_PATH)["arousal_mean"]


if __name__ == "__main__":
    audio_filepaths, valence_targets, arousal_targets = get_audio_filepaths(), get_valence_targets(), get_arousal_targets()

    print(audio_filepaths)
    print(valence_targets)
    print(arousal_targets)
