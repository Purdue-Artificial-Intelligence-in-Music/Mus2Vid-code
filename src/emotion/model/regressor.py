from sklearn.svm import SVR
import os
import joblib


MODEL_DIR = "./models/emotion"
MODEL_EXT = "model"


class EmotionRegressor():

    def __init__(self):
        self.svr = SVR(epsilon=0.2)

    def fit(self, features, targets):
        self.svr.fit(features, targets)

    def predict(self, input):
        return self.svr.predict(input)

    def save(self, filename):
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        joblib.dump(self.svr, f"{MODEL_DIR}/{filename}.{MODEL_EXT}")

    def load(self, filename):
        self.svr = joblib.load(f"{MODEL_DIR}/{filename}.{MODEL_EXT}") 


if __name__ == "__main__":
    pass
