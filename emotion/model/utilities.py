import pickle 

ROOT = "./data/processed/"

def get_features():
    with open(ROOT + "features", "rb") as f:
        return pickle.load(f)

def get_valence_targets():
    with open(ROOT + "valence-targets", "rb") as f:
        return pickle.load(f)

def get_arousal_targets():
    with open(ROOT + "arousal-targets", "rb") as f:
        return pickle.load(f)