from regressor import EmotionRegressor

def test(test_input):
    valence_regressor, arousal_regressor = EmotionRegressor(), EmotionRegressor()

    valence_regressor.load("valence_regressor")
    arousal_regressor.load("arousal_regressor")

    valence_regressor.predict(test_input)

if __name__ == "__main__":
    pass
