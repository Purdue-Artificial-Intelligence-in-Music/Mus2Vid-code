from src.emotion.features.extract import extract_opensmile_features
from src.emotion.features.best import get_best_opensmile_features
import keras

MODEL_DIR = "models/emotion"
MODEL_EXT = "keras"

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


def get_emotion(audio_filepath: str) -> str:
    """Process audio at given filepath and return an emotion.

    Parameters
    ----------
    audio_filepath
        Filepath of audio relative to repository root.

    Returns
    -------
    emotion: str
        An emotion word or phrase that describes the provided audio.
    """
    valence, arousal = get_va_values(audio_filepath)

    emotion = "happy"

    return emotion


if __name__ == "__main__":
    valence, arousal = get_va_values("../data/processed/audio/2.wav")
    print(f"(valence, arousal) = ({valence:.4f}, {arousal:.4f})")
