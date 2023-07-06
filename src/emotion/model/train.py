from regressor import EmotionRegressor
from utilities import get_features, get_valence_targets, get_arousal_targets


def train():
    valence_features = get_features("opensmile_valence")
    arousal_features = get_features("opensmile_arousal")
    valence_targets = get_valence_targets()
    arousal_targets = get_arousal_targets()

    valence_regressor, arousal_regressor = EmotionRegressor(), EmotionRegressor()

    valence_regressor.fit(valence_features, valence_targets)
    arousal_regressor.fit(arousal_features, arousal_targets)
    
    valence_regressor.save("valence_regressor")
    arousal_regressor.save("arousal_regressor")


if __name__ == "__main__":
    train()
