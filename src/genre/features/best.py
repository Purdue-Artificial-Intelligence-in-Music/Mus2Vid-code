from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import joblib
import os
import numpy as np
from src.genre.features.util import INTERIM_DATA_DIR, SELECTOR_EXT
from src.genre.features.extract import get_features

def select_features():
    if(not(os.path.exists(f"{INTERIM_DATA_DIR}labeled_features.xz"))):
        get_features()
    
    with lzma.open(f"{INTERIM_DATA_DIR}labeled_features.xz", "rb") as f:
        labeled_features = pickle.load(f)

    labeled_selected_features = select_features(labeled_features)

    with lzma.open(f"{INTERIM_DATA_DIR}labeled_selected_features.xz", "rb") as f:
        pickle.dump(labeled_selected_features, f)

def select_features(labeled_features) -> np.array():
    num_cols = labeled_features.shape[1] - 1
    print(num_cols)

    X = labeled_features[:, :num_cols] # All columns except labels
    Y = labeled_features[:, num_cols] # Only labels

    selector = SelectKBest(score_func=f_classif, k=256) # Selects the 256 features with strongest relationships to classification

    fit = selector.fit(X, Y)

    if (not(os.path.exists(f"{INTERIM_DATA_DIR}genre_features{SELECTOR_EXT}"))): # save selector to util folder
        joblib.dump(selector, f"{INTERIM_DATA_DIR}genre_features{SELECTOR_EXT}")

    selected_features = selector.transform(X) # Narrows feature matrix

    labeled_selected_features = np.column_stack([features, Y]) # Add labels back to feature matrix

    return labeled_selected_features