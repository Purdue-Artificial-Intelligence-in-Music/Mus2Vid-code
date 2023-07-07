import pandas as pd


ROOT = "./../../.."
ANNOTATIONS_PATH = f"{ROOT}/data/processed/annotations/static_annotations_averaged_songs_1_2000.csv"
AUDIO_DIR = f"{ROOT}/data/processed/audio"
FEATURES_DIR = f"{ROOT}/data/interim/features"
FEATURES_EXT = "features"
SELECTOR_EXT = "selector"
CHUNK_SIZE = 10


def get_audio_filepaths():
    song_id_list = pd.read_csv(ANNOTATIONS_PATH)["song_id"]

    audio_filepaths = []
    for song_id in song_id_list:
        audio_filepaths.append(f"{AUDIO_DIR}/{song_id}.wav")

    return audio_filepaths


def get_valence_targets():
    return pd.read_csv(ANNOTATIONS_PATH)["valence_mean"]


def get_arousal_targets():
    return pd.read_csv(ANNOTATIONS_PATH)["arousal_mean"]


if __name__ == "__main__":
    audio_filepaths, valence_targets, arousal_targets = get_audio_filepaths(), get_valence_targets(), get_arousal_targets()

    print(audio_filepaths)
    print(valence_targets)
    print(arousal_targets)
