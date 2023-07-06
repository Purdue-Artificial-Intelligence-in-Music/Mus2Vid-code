import pandas as pd


ANNOTATIONS_PATH = "./data/processed/annotations/static_annotations_averaged_songs_1_2000.csv"
AUDIO_DIR = "./data/processed/audio"
FEATURES_DIR = "./data/interim/features"
FEATURES_EXT = "features"
CHUNK_SIZE = 10


def get_song_id_list():
    return pd.read_csv(ANNOTATIONS_PATH)["song_id"]


def get_valence_targets():
    return pd.read_csv(ANNOTATIONS_PATH)["valence_mean"]


def get_arousal_targets():
    return pd.read_csv(ANNOTATIONS_PATH)["arousal_mean"]


if __name__ == "__main__":
    song_id_list, valence_targets, arousal_targets = get_song_id_list(), get_valence_targets(), get_arousal_targets()

    print(song_id_list)
    print(valence_targets)
    print(arousal_targets)
