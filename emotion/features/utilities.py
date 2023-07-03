import pandas as pd

ANNOTATIONS_PATH = "./data/processed/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"

def get_song_id_list():
    df = pd.read_csv(ANNOTATIONS_PATH)
    song_id_list = df["song_id"]

    return song_id_list

def get_targets():
    df = pd.read_csv(ANNOTATIONS_PATH)
    targets = df[["valence_mean", "arousal_mean"]]

    return targets

if __name__ == "__main__":
    song_id_list, targets = get_song_id_list(), get_targets()

    print(song_id_list)
    print(targets)
