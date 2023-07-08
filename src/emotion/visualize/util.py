import pandas as pd


ROOT = ".."
ANNOTATIONS_PATH = f"{ROOT}/data/processed/annotations/static_annotations_averaged_songs_1_2000.csv"


def get_dataset_va_values():
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    return annotations["valence_mean"], annotations["arousal_mean"]


if __name__ == "__main__":
    print(get_dataset_va_values())
