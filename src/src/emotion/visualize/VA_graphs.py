import matplotlib.pyplot as plt   
import pandas as pd
import joblib
import opensmile
import os
import keras
DATA_DIR = "data/interim"
ANNOTATIONS_PATH = "./data/processed/annotations/static_annotations_averaged_songs_1_2000.csv"
AUDIO_DIR = "./data/processed/audio"
FEATURES_DIR = "./data/interim/features"
FEATURES_EXT = "features"
SELECTOR_EXT = "selector"
MODEL_DIR = "models/emotion"
MODEL_EXT = "keras"
CHUNK_SIZE = 10

def get_best_opensmile_features(opensmile_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the best pre-extracted openSMILE features.

    Parameters
    ----------
    opensmile_features
        Pre-extracted openSMILE features.

    Returns
    -------
    opensmile_valence_features: pandas.DataFrame
    opensmile_arousal_features: pandas.DataFrame
    """

    opensmile_valence_selector = joblib.load(f"{FEATURES_DIR}/fcnn_valence.{SELECTOR_EXT}")
    opensmile_arousal_selector = joblib.load(f"{FEATURES_DIR}/fcnn_arousal.{SELECTOR_EXT}")

    opensmile_valence_features = opensmile_valence_selector.transform(opensmile_features)
    opensmile_arousal_features = opensmile_arousal_selector.transform(opensmile_features)

    return opensmile_valence_features, opensmile_arousal_features

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

def get_va_values(audio_filepath: str) -> tuple[float, float]:
    """Process audio at given filepath and return valence and arousal values.

    Parameters
    ----------
    audio_filepath
        Filepath relative to repository root.

    Returns
    -------
    valence: float
        A float between 1 and 9.
    arousal: float
        A float between 1 and 9.
    """
    valence_regressor = keras.models.load_model(f"{MODEL_DIR}/valence.{MODEL_EXT}")
    arousal_regressor = keras.models.load_model(f"{MODEL_DIR}/arousal.{MODEL_EXT}")


    opensmile_features = extract_opensmile_features([audio_filepath])
    opensmile_valence_features, opensmile_arousal_features = get_best_opensmile_features(opensmile_features)
    
    valence = valence_regressor.predict(opensmile_valence_features)[0]
    arousal = arousal_regressor.predict(opensmile_arousal_features)[0]

    return valence, arousal

(bd_v, bd_a) = get_va_values(f"{DATA_DIR}/test_wavs/blue_danube.wav")
(ms_v, ms_a) = get_va_values(f"{DATA_DIR}/test_wavs/moonlight_sonata.wav")
(vw_v, vw_a) = get_va_values(f"{DATA_DIR}/test_wavs/vivaldi-winter.wav")
print(bd_v, bd_a, ms_v, ms_a, vw_v, vw_a)

VAs = pd.read_csv(f"{DATA_DIR}/vacsv.csv")
emotions = VAs["Emotion"]
valences = VAs["Valence"]
arousals = VAs["      Arousal"]
annotate = VAs["Annotate"]

fig, ax = plt.subplots()
plt.xlim([1,9])
plt.ylim([1,9])
plt.xlabel("Valence")
plt.ylabel("Arousal")
ax.scatter(valences, arousals)

for i, txt in enumerate(emotions):
    if(annotate[i]):
        ax.annotate(txt, (valences[i], arousals[i]))

ax.scatter(bd_v, bd_a, color = "m")
ax.scatter(ms_v, ms_a, color = "r")
ax.scatter(vw_v, vw_a, color = "k")

plt.show()