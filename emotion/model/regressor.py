from sklearn.svm import SVR

class EmotionRegressor():

    def __init__(self):
        self.svr = SVR(epsilon=0.2)

    def fit(self, features, targets):
        self.svr.fit(features, targets)

    def predict(self, input):
        self.svr.predict(input)

    def save(self):
        pass

if __name__ == "__main__":
    pass