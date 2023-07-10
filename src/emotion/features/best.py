import joblib
import os
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from src.emotion.features.util import get_valence_targets, get_arousal_targets, FEATURES_DIR, FEATURES_EXT, SELECTOR_EXT


def _save_opensmile_feature_selectors():
    opensmile_features = joblib.load(f"{FEATURES_DIR}/opensmile.{FEATURES_EXT}")
    valence_targets = get_valence_targets()
    arousal_targets = get_arousal_targets()

    opensmile_valence_selector = SelectKBest(score_func=f_regression, k=100) # Choose the 100 most effective features
    opensmile_arousal_selector = SelectKBest(score_func=f_regression, k=100) # Choose the 100 most effective features

    opensmile_valence_selector.fit(opensmile_features, valence_targets)
    opensmile_arousal_selector.fit(opensmile_features, arousal_targets)

    if not os.path.exists(FEATURES_DIR): os.mkdir(FEATURES_DIR)
    joblib.dump(opensmile_valence_selector, f"{FEATURES_DIR}/opensmile_valence.{SELECTOR_EXT}")
    joblib.dump(opensmile_arousal_selector, f"{FEATURES_DIR}/opensmile_arousal.{SELECTOR_EXT}")


def get_best_opensmile_features(opensmile_features: pd.DataFrame) -> tuple[list, list]:
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
    if not (
        os.path.exists(f"{FEATURES_DIR}/opensmile_valence.{SELECTOR_EXT}") and
        os.path.exists(f"{FEATURES_DIR}/opensmile_arousal.{SELECTOR_EXT}")
    ):
        _save_opensmile_feature_selectors()

    opensmile_valence_selector = joblib.load(f"{FEATURES_DIR}/opensmile_valence.{SELECTOR_EXT}")
    opensmile_arousal_selector = joblib.load(f"{FEATURES_DIR}/opensmile_arousal.{SELECTOR_EXT}")

    opensmile_valence_features = opensmile_valence_selector.transform(opensmile_features)
    opensmile_arousal_features = opensmile_arousal_selector.transform(opensmile_features)

    return opensmile_valence_features, opensmile_arousal_features


if __name__ == "__main__":
    print(get_best_opensmile_features(joblib.load(f"{FEATURES_DIR}/opensmile.{FEATURES_EXT}")))
