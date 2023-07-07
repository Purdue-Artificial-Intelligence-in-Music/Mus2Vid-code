import sys
sys.path.append("./src/emotion/data")
sys.path.append("./src/emotion/features")
sys.path.append("./src/emotion/model")
sys.path.append("./src/emotion/visualize")
print(sys.path)

from model.regressor import EmotionRegressor
from features.extract import extract_opensmile_features
from features.best import get_best_opensmile_features


def get_va_values(audio_filepath):
    valence_regressor, arousal_regressor = EmotionRegressor(), EmotionRegressor()
    valence_regressor.load("valence_regressor")
    arousal_regressor.load("arousal_regressor")

    opensmile_features = extract_opensmile_features([audio_filepath])
    opensmile_valence_features, opensmile_arousal_features = get_best_opensmile_features(opensmile_features)

    valence = valence_regressor.predict(opensmile_valence_features)[0]
    arousal = arousal_regressor.predict(opensmile_arousal_features)[0]

    return valence, arousal


def get_emotion(audio_filepath):
    valence, arousal = get_va_values(audio_filepath)


if __name__ == "__main__":
    valence, arousal = get_va_values("./data/processed/audio/2.wav")
    print(f"(valence, arousal) = ({valence:.4f}, {arousal:.4f})")
