import joblib
import pandas as pd

ANNOTATIONS_PATH = "./data/processed/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
FEATURES_PATH = "./data/processed/features/"
TARGETS_PATH = "./data/processed/targets/"
FEATURES_EXT = ".features"
TARGETS_EXT = ".targets"

def get_features(filename):
    return joblib.load(FEATURES_PATH + filename + FEATURES_EXT)

def get_valence_targets():
    df = pd.read_csv(ANNOTATIONS_PATH)
    targets = df["valence_mean"]

    return targets

def get_arousal_targets():
    df = pd.read_csv(ANNOTATIONS_PATH)
    targets = df["arousal_mean"]

    return targets

if __name__ == "__main__":
    pass