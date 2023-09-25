import joblib
import pandas as pd


ANNOTATIONS_PATH = "./data/processed/annotations/static_annotations_averaged_songs_1_2000.csv"
FEATURES_DIR = "./data/interim/features"
TARGETS_DIR = "./data/interim/targets"
MODEL_DIR = "./models/emotion"
FEATURES_EXT = "features"
SELECTOR_EXT = "selector"
TARGETS_EXT = "targets"
MODEL_EXT = "model"

def get_valence_selector():
    return joblib.load(f"{FEATURES_DIR}/fcnn_valence.{SELECTOR_EXT}")

def get_arousal_selector():
    return joblib.load(f"{FEATURES_DIR}/fcnn_arousal.{SELECTOR_EXT}")


def get_features(filename) -> pd.DataFrame:
    """Return a pandas.DataFrame of features loaded from a file."""
    return joblib.load(f"{FEATURES_DIR}/{filename}.{FEATURES_EXT}")


def get_valence_targets() -> pd.Series:
    """Return a pandas.Series of target values for valence."""
    return pd.Series.tolist(pd.read_csv(ANNOTATIONS_PATH)["valence_mean"])


def get_arousal_targets() -> pd.Series:
    """Return a pandas.Series of target values for arousal."""
    return pd.Series.tolist(pd.read_csv(ANNOTATIONS_PATH)["arousal_mean"])


if __name__ == "__main__":
    pass
