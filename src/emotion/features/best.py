import opensmile
import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from utilities import get_valence_targets, get_arousal_targets, FEATURES_DIR, FEATURES_EXT


def select_librosa_features(librosa_features):
    valence_targets = get_valence_targets()
    arousal_targets = get_arousal_targets()

    selector = SelectKBest(f_regression, k=100)

    librosa_valence_features = selector.fit_transform(librosa_features, valence_targets)
    librosa_arousal_features = selector.fit_transform(librosa_features, arousal_targets)

    return librosa_valence_features, librosa_arousal_features

def select_opensmile_features(opensmile_features):
    valence_targets = get_valence_targets()
    arousal_targets = get_arousal_targets()

    selector = SelectKBest(score_func=f_regression, k=100) # Choose the 100 most effective features
    opensmile_valence_features = selector.fit_transform(opensmile_features, valence_targets)
    opensmile_arousal_features = selector.fit_transform(opensmile_features, arousal_targets)

    return opensmile_valence_features, opensmile_arousal_features


if __name__ == "__main__":
    opensmile_features = joblib.load(f"{FEATURES_DIR}/opensmile.{FEATURES_EXT}")

    opensmile_valence_features, opensmile_arousal_features = select_opensmile_features(opensmile_features)

    joblib.dump(opensmile_valence_features, f"{FEATURES_DIR}/opensmile_valence.{FEATURES_EXT}")
    joblib.dump(opensmile_arousal_features, f"{FEATURES_DIR}/opensmile_arousal.{FEATURES_EXT}")
