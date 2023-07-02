import pandas as pd

ANNOTATIONS_PATH = "./data/processed/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"

def get_data():
    dataframe = pd.read_csv(ANNOTATIONS_PATH)
    song_id_list = dataframe["song_id"]
    targets = dataframe[["valence_mean", "arousal_mean"]]

    return song_id_list, targets

if __name__ == "__main__":
    song_id_list, targets = get_data()
    print(song_id_list)
    print(targets)