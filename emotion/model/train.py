from regressor import EmotionRegressor
from utilities import get_features, get_valence_targets, get_arousal_targets

def train():
    features = get_features()
    valence_targets = get_valence_targets()
    arousal_targets = get_arousal_targets()

    valence_regressor, arousal_regressor = EmotionRegressor(), EmotionRegressor()

    valence_regressor.fit(features, valence_targets)
    arousal_regressor.fit(features, arousal_targets)
    
    valence_regressor.save("valence_regressor")
    arousal_regressor.save("arousal_regressor")

if __name__ == "__main__":
    pass
