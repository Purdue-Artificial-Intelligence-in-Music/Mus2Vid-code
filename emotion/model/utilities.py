import joblib

FEATURES_PATH = "./data/processed/features/"
TARGETS_PATH = "./data/processed/targets/"
FEATURES_EXT = ".features"
TARGETS_EXT = ".targets"

def get_features(filename):
    return joblib.load(FEATURES_PATH + filename + FEATURES_EXT)

def get_valence_targets(filename):
    return joblib.load(TARGETS_PATH + filename + TARGETS_EXT)

def get_arousal_targets(filename):
    return joblib.load(TARGETS_PATH + filename + TARGETS_EXT)

if __name__ == "__main__":
    pass
