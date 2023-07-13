import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import resample
import os
from tensorflow import keras
from keras import metrics
import matplotlib.pyplot as plt

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

    def test_train_split