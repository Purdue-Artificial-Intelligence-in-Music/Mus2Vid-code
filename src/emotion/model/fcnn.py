from utils.util import *
import os
import pandas as pd
import numpy as np
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt

def get_matrix(type):
    features = get_features("opensmile")
    if (type == "valence"):
        targets = get_valence_targets()
        selector = get_valence_selector()
        features = selector.transform(features).tolist()
    elif (type == "arousal"):
        targets = get_arousal_targets()
        selector = get_arousal_selector()
        features = selector.transform(features).tolist()
    else:
        print("invalid type")
        return

    for i in range(len(features)):
        features[i].append(targets[i])

    return features

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

def split(labeled_features):
    labeled_features = np.random.permutation(labeled_features)

    # Partition the Dataset into 3 Sets: Training, Validation, and Test
    num = len(labeled_features)
    # Calculate the number of samples for training data (60% of the dataset)
    num_training = int(num * 0.6)
    # Calculate the number of samples for validation data (20% of the dataset)
    num_validation = int(num * 0.8)

    # Extract the training data (60% of the labeled features)
    training_data = (labeled_features[:num_training])
    # Extract the validation data (20% of the labeled features)
    validation_data = (labeled_features[num_training:num_validation])
    # Extract the test data (remaining 20% of the labeled features)
    test_data = (labeled_features[num_validation:])

    # Separate the features from the labels
    num_cols = training_data.shape[1] - 1
    # Extract features from the training data
    training_features = training_data[:, :num_cols]
    # Extract features from the validation data
    validation_features = validation_data[:, :num_cols]
    # Extract features from the test data
    test_features = test_data[:, :num_cols]

    # Extract years from the training data
    training_labels = training_data[:, num_cols]
    # Extract years from the validation data
    validation_labels = validation_data[:, num_cols]
    # Extract years from the test data
    test_labels = test_data[:, num_cols]

    return training_features, test_features, validation_features, training_labels, test_labels, validation_labels

def test_load(test_feats, test_labels, path):
    model = keras.models.load_model(path)


def build_model(type):
    features = get_matrix(type)
    train_features, test_features, validation_features, train_labels, test_labels, validation_labels = split(features)

    num_feats = train_features.shape[1]
    print(num_feats)

    normalizer = keras.layers.Normalization(input_dim=256)
    normalizer.adapt(train_features)

    model = keras.Sequential([
        normalizer,
        keras.layers.Dense(num_feats, activation='relu', kernel_regularizer='l2'),
        keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')

    history = model.fit(
        x = train_features,
        y = train_labels,
        validation_data = (validation_features, validation_labels),
        epochs=100,
        verbose=1,
        batch_size=8,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    test_preds = model.predict(test_features).flatten()
    error = test_preds - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction error (" + str(type) + ")")
    plt.ylabel("Count")
    plt.show()

    # plot_loss(history)

    if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)
    model.save(f"{MODEL_DIR}/{type}.keras")

    test_load(test_features, test_labels, f"{MODEL_DIR}/{type}.keras")

    return

if __name__ == "__main__":
    build_model("valence")
    build_model("arousal")
