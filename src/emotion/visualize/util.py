import pandas as pd


ANNOTATIONS_PATH = "./data/processed/annotations/static_annotations_averaged_songs_1_2000.csv"


def get_dataset_va_values() -> tuple[pd.Series, pd.Series]:
    """Return the valence and arousal values for the dataset."""
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    return annotations["valence_mean"], annotations["arousal_mean"]


if __name__ == "__main__":
    print(get_dataset_va_values())
