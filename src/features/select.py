import opensmile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from src.data import get as data_get 
from src.features import extract as features_extract

def select_librosa_features():
    pass

def select_opensmile_features():
    song_id_list, targets = data_get.get_data()

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    feature_names = smile.feature_names

if __name__ == "__main__":
    pass
