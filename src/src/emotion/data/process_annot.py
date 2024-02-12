import os
import pandas as pd
from utils.util import RAW_ANNOTATIONS_DIR, RAW_AUDIO_DIR, PROCESSED_ANNOTATIONS_DIR, PROCESSED_AUDIO_DIR, \
    ANNOTATIONS_FILE, ANNOTATIONS_FILE_1, ANNOTATIONS_FILE_2, TEMP_AUDIO_DIR


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


def process_annotations_new() -> None:
    """Process audio annotations for use with emotion module.

    Take raw annotation files and remove extra space from column headers.

    Returns
    -------
    None
    """
    if not os.path.exists(PROCESSED_ANNOTATIONS_DIR):
        os.mkdir(PROCESSED_ANNOTATIONS_DIR)

    annotations_1 = pd.read_csv(f"{RAW_ANNOTATIONS_DIR}/{ANNOTATIONS_FILE_1}")
    annotations_2 = pd.read_csv(f"{RAW_ANNOTATIONS_DIR}/{ANNOTATIONS_FILE_2}")
    annotations = pd.concat([annotations_1, annotations_2])
    annotations = annotations.rename(columns={
        " valence_mean": "valence_mean",
        " valence_std": "valence_std",
        " arousal_mean": "arousal_mean",
        " arousal_std": "arousal_std"
    })
    output = pd.DataFrame(columns=annotations.columns)
    row_list = []
    for filename in os.listdir(PROCESSED_AUDIO_DIR):
        id = filename.split("#")[0]
        root = filename.split(".")[0]
        if annotations.loc[annotations['song_id'] == int(id)].empty:
            print(id)
            print(root)
            raise Exception("song_id not found in annotations df")
        else:
            idx = len(output.index)
            row_list.append(annotations.loc[annotations['song_id'] == int(id)].copy())
            row_list[len(row_list) - 1]['valence_mean'] = round((row_list[len(row_list) - 1]['valence_mean'] - 1) / 8.0,
                                                                4)
            row_list[len(row_list) - 1]['arousal_mean'] = round((row_list[len(row_list) - 1]['arousal_mean'] - 1) / 8.0,
                                                                4)
            row_list[len(row_list) - 1]['valence_std'] = round(row_list[len(row_list) - 1]['valence_std'] / 8.0, 4)
            row_list[len(row_list) - 1]['arousal_std'] = round(row_list[len(row_list) - 1]['arousal_std'] / 8.0, 4)
    output = pd.concat(row_list)
    output.to_csv(f"{PROCESSED_ANNOTATIONS_DIR}/{ANNOTATIONS_FILE}", index=False)


if __name__ == "__main__":
    process_annotations_new()
