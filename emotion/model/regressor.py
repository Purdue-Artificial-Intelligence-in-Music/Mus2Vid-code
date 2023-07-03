from sklearn.svm import SVR
import joblib

FILE_EXTENSION = ".model"
MODEL_PATH = "./models/emotion/"

class EmotionRegressor():

    def __init__(self):
        self.svr = SVR(epsilon=0.2)

    def fit(self, features, targets):
        self.svr.fit(features, targets)

    def predict(self, input):
        self.svr.predict(input)

    def save(self, filename):
        joblib.dump(self.svr, MODEL_PATH + filename + FILE_EXTENSION)

    def load(self, filename):
        self.svr = joblib.load(MODEL_PATH + filename + FILE_EXTENSION) 

if __name__ == "__main__":
    pass
