import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import resample
import os
from tensorflow import keras
from keras import metrics
import matplotlib.pyplot as plt
import pickle, lzma
from src.genre.model.util import *
from src.genre.features.best import select_features

def model():
    if(not(os.path.exists(f"{INTERIM_DATA_DIR}labeled_selected_features.xz"))):
        select_features()

    with lzma.open(f"{INTERIM_DATA_DIR}labeled_selected_features.xz", "rb") as f:
        labeled_selected_features = pickle.load(f)

    (training_features, training_labels, validation_features, validation_labels, test_features, test_labels) = test_train_split(labeled_selected_features)
    
    model = classification_network(training_features, training_labels, validation_features, validation_labels)
    model.save(f"{INTERIM_DATA_DIR}genre_model{MODEL_EXT}")


def balance_data(data_array, classes = genre_list):
    """Resamples underrespresented classes in dataset so each class has the same amount of audio clips.
    Avoids the model prediciting the most common class all the time.

    Parameters
    ----------
    data_array: np.array
        Feature matrix to be balanced (with labels)
    genres : list, optional
        list of potential classes, by default genre_list

    Returns
    -------
    balanced_array: np.array
        resampled feature matrix with equal samples in each class
    """

    # Create a list of column titles to seperate features and genres
    name_list = ['data'] * (data_array.shape[1] - 1)
    name_list.append('genre')
    # We need a dataframe to use the sklearn.resample function
    data_df = pd.DataFrame(data_array)
    data_df.columns = name_list
    
    # Count the number of samples in each class
    class_balance = np.zeros(len(classes))
    for index, row in data_df.iterrows():
        class_balance[int(row['genre'])] += 1
    max_samples = int(max(class_balance))

    # resample other genres to have the same number of samples as the most common genre
    df_list = [] # list of dataframes to be resampleds
    for class_num in range(len(classes)):
        df_list.append(data_df[data_df['genre'] == class_num])

    balanced_list = [] # list of resampled dataframes
    for df in df_list:
        if (len(df) != max_samples):
            df = resample(df, random_state=42, n_samples=max_samples, replace=True)
        balanced_list.append(df)

    balanced_df = pd.concat(balanced_list) # Add dataframes vertically
    balanced_array = balanced_df.to_numpy() # Convert back to a np.array
    
    return balanced_array


def test_train_split(labeled_features):
        # Shuffle the features
    labeled_features = np.random.permutation(labeled_features)

    # Partition the Dataset into 3 Sets: Training, Validation, and Test
    num = len(labeled_features)
    # Calculate the number of samples for training data (60% of the dataset)
    num_training = int(num * 0.6)
    # Calculate the number of samples for validation data (20% of the dataset)
    num_validation = int(num * 0.8)

    # Extract the training data (60% of the labeled features)
    training_data = balance_data(labeled_features[:num_training])
    # Extract the validation data (20% of the labeled features)
    validation_data = balance_data(labeled_features[num_training:num_validation])
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

    # Format the features for this multi-class classification problem
    num_classes = len(genre_list)
    # Extract years from the training data
    training_labels = training_data[:, num_cols].astype(int)
    # Extract years from the validation data
    validation_labels = validation_data[:, num_cols].astype(int)
    # Extract years from the test data
    test_labels = test_data[:, num_cols].astype(int)

    return training_features, training_labels, validation_features, validation_labels, test_features, test_labels


def classification_network(training_features, training_labels, validation_features, validation_labels):
    num_features = training_features.shape[1]

    # Define the model architecture
    normalizer = keras.layers.BatchNormalization()
    model = keras.Sequential([
        # Input layer with 256 neurons (one for every feature) and ReLU activation
        keras.layers.Dense(num_features, input_shape=(training_features.shape[1],), activation='relu'),
        
        # Hidden layer with 2/3 the neurons (https://shorturl.at/sBLWZ) and ReLU activation
        keras.layers.Dense((num_features * 2/3), activation='relu'),
        
        # Output layer with num_classes neurons and softmax activation for multi-class classification
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    """
    optimizer="adam": The optimizer algorithm to use during training. 
    Adam optimizer is chosen, which is a popular optimization algorithm known for its efficiency.

    loss='categorical_crossentropy': The loss function used to measure the discrepancy between the 
    predicted output and the true output labels. Categorical cross-entropy is suitable for
    multi-class classification tasks.

    metrics=['accuracy']: The metric(s) to be evaluated during training and testing. 
    Accuracy is a commonly used metric to assess the model's performance.
    """

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    """
    training_features, train_labels: Input features and corresponding labels for model training.

    validation_features, val_labels: Validation set used to monitor the model's performance 
                                            during training.

    batch_size=32: Number of samples per gradient update. Training data is divided into batches, 
                and the model's weights are updated after each batch.

    epochs=50: Number of times the model will iterate over the entire training dataset.

    callbacks: EarlyStopping to stop training if the validation loss does not improve for a certain 
            number of epochs, and ModelCheckpoint to save the best model based on validation loss.
    """

    # Encode the training and validation labels using one-hot encoding
    train_labels_encoded = to_categorical(training_labels)
    val_labels_encoded = to_categorical(validation_labels)

    history = model.fit(x=training_features, y=train_labels_encoded, 
                        validation_data=(validation_features, val_labels_encoded),
                        batch_size=10, epochs=50, verbose=2,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                                keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)])

    return model