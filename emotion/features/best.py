# FIXME
import opensmile
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from utilities import get_targets, FEATURES_PATH, FEATURES_EXT
import joblib
import pickle, lzma

def select_librosa_features(librosa_features):
    targets = get_targets()[:10]
    selector = SelectKBest(f_regression, k=100)
    best_librosa_features = selector.fit_transform(librosa_features, targets)

    return best_librosa_features

def select_opensmile_features(matched_smile_df):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    feature_names = smile.feature_names

    # Convert to numpy array of features
    smile_features = matched_smile_df['features']
    smile_features = np.stack(smile_features.values)
    # Convert to numpy array of labels
    valence_labels = matched_smile_df['valence']
    arousal_labels = matched_smile_df['arousal']

    selector = SelectKBest(score_func=f_regression, k=100) # Choose the 100 most effective features

    valence_fit = selector.fit(smile_features, valence_labels)
    valence_features = selector.transform(smile_features)
    print(selector.get_feature_names_out())

    arousal_fit = selector.fit(smile_features, arousal_labels)
    arousal_features = selector.transform(smile_features)
    print(selector.get_feature_names_out())

    return valence_features, arousal_features

if __name__ == "__main__":
    with lzma.open('emotion/data/matched_smile.xz', 'rb') as f:
        matched_smile_df = pickle.load(f)
    valence_feats, arousal_feats = select_opensmile_features(matched_smile_df)
    joblib.dump(valence_feats, FEATURES_PATH + "valence_features" + FEATURES_EXT)
    joblib.dump(arousal_feats, FEATURES_PATH + "arousal_features" + FEATURES_EXT)
