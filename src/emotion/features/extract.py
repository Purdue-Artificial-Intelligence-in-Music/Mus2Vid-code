import os
import pandas as pd
import opensmile
import joblib 
from utils.util import FEATURES_DIR, FEATURES_EXT, CHUNK_SIZE


def extract_opensmile_features(audio_filepaths: list[str]):
    """Return a pandas.DataFrame of openSMILE features 
    extracted from audio specified at the given filepaths.

    Parameters
    ----------
    audio_filepaths
        A list of audio filepaths relative to repository root.
    """
    # if os.path.exists(f"{FEATURES_DIR}/opensmile.{FEATURES_EXT}"):
    #     return joblib.load(f"{FEATURES_DIR}/opensmile.{FEATURES_EXT}")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    opensmile_features_list = []

    size = len(audio_filepaths)
    for i, audio_filepath in enumerate(audio_filepaths):
        if i % CHUNK_SIZE == 0:
            print(f"{i}/{size}")

        # Get smile features, convert from df to list, and convert from 2D list to 1D list
        opensmile_features_list.append(sum(smile.process_file(audio_filepath).values.tolist(), []))

    if not os.path.exists(FEATURES_DIR): os.mkdir(FEATURES_DIR)
    joblib.dump(opensmile_features_list, f"{FEATURES_DIR}/opensmile.{FEATURES_EXT}")

    return pd.DataFrame(data=opensmile_features_list)


if __name__ == "__main__":
    from utils.util import get_audio_filepaths

    audio_filepaths = get_audio_filepaths()
    print(extract_opensmile_features(audio_filepaths))
