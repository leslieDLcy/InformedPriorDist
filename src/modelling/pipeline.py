import pandas as pd
import numpy as np
import tensorflow as tf
from utils import sin_transformer, cos_transformer

def import_data(path='data/raw/weekly_media_sample.csv'):
    """ import raw data from csv file into a dataframe """

    dataset = pd.read_csv(path, header=0,)
    dataset.drop(columns='X', inplace=True)
    time_axis = dataset.pop('DATE')

    return dataset, time_axis


def base_process_data(dataset, split_index=200, normalize=False):
    """ processing operations on the raw data for base models 
    
    Parameters
    ----------
    normalize: bool
        if True, normalize the feature vector in situ; otherwise, use the `normalizer` layer

    Note
    ----
        base models are models that do not use time information,
    """ 

    # transform `revenue` in millions 
    dataset['revenue'] = dataset['revenue'] / 1000000

    train_val_split = split_index
    train_dataset = dataset[:train_val_split]
    test_dataset = dataset[train_val_split:]

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('revenue')
    test_labels = test_features.pop('revenue')

    # change to numpy arrays
    train_features = train_features.values
    test_features = test_features.values

    train_labels = train_labels.values
    test_labels = test_labels.values

    if normalize == True:
        train_mean = train_features.mean()
        train_std = train_features.std()

        train_features = (train_features - train_mean) / train_std
        test_features = (test_features - train_mean) / train_std

    return train_features, test_features, train_labels, test_labels







def temporal_process_data(dataset, split_index=200):
    """ procedures for temporal models """

    # transform `revenue` in millions 
    dataset['revenue'] = dataset['revenue'] / 1000000

    # an index axis as `index_time`
    index_time = np.arange(1, len(dataset)+1)
    
    # # a bit feature engineering
    # dataset['month_sin'] = sin_transformer(period=4, x=index_time)
    # dataset['month_cos'] = cos_transformer(period=4, x=index_time)

    dataset['year_sin']  = sin_transformer(period=4*12, x=index_time)
    # dataset['year_cos']  = cos_transformer(period=4*12, x=index_time)

    print('Processed dataset shape: ', dataset.shape)

    processed_dataset = dataset.copy()

    # split train and test
    train_val_split = split_index
    train_df = dataset[:train_val_split]
    val_df = dataset[train_val_split:]

    # normalize also the target itself due to AR process
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std

    return train_df, val_df


def feature_normalization(train_features):
    """ normalize the feature vector 
    
    Style
    -----
        - use of `Normalization` layer into the model
    """

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    print("The mean of feature vector:", normalizer.mean.numpy())

    print('Check the normalized first example:')
    first = np.array(train_features[:1])

    with np.printoptions(precision=2, suppress=True):
        print('First example:', first)
        print()
        print('Normalized:', normalizer(first).numpy())

    return normalizer



def ensemble_predict(model, test_data, ensemble_size):
    """ ensemble prediction and concatenate """

    ensem_preds = [np.squeeze(model.predict(test_data)) for _ in range(ensemble_size)]
    return np.stack(ensem_preds, axis=0)


