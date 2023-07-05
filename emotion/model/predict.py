from regressor import EmotionRegressor

def predict(input):
    valence_regressor, arousal_regressor = EmotionRegressor(), EmotionRegressor()

    valence_regressor.load("valence_regressor")
    arousal_regressor.load("arousal_regressor")

    valence_output = valence_regressor.predict(input)
    arousal_output = arousal_regressor.predict(input)

    return valence_output, arousal_output

if __name__ == "__main__":
    input = []
    valence_output, arousal_output = predict(input)
