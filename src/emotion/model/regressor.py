from sklearn.svm import SVR
import joblib


MODEL_PATH = "./models/emotion/"
MODEL_EXT = ".model"


class EmotionRegressor():

    def __init__(self):
        self.svr = SVR(epsilon=0.2)

    def fit(self, features, targets):
        self.svr.fit(features, targets)

    def predict(self, input):
        return self.svr.predict(input)

    def save(self, filename):
        joblib.dump(self.svr, MODEL_PATH + filename + MODEL_EXT)

    def load(self, filename):
        self.svr = joblib.load(MODEL_PATH + filename + MODEL_EXT) 


if __name__ == "__main__":
    pass
