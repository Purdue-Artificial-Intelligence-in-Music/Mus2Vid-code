# FIXME
# import opensmile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from utilities import get_targets, FEATURES_PATH, FEATURES_EXT
import joblib

def select_librosa_features(librosa_features):
    targets = get_targets()[:10]
    selector = SelectKBest(f_regression, k=100)
    best_librosa_features = selector.fit_transform(librosa_features, targets)

    return best_librosa_features

def select_opensmile_features():
    pass

if __name__ == "__main__":
    librosa_features = joblib.load(FEATURES_PATH + "librosa_features" + FEATURES_EXT)
    print(librosa_features["mfcc"][0][0])
    print(len(librosa_features["mfcc"][0][0]))
    best_features = select_librosa_features(librosa_features)
    print(best_features)
    print(best_features.shape)
