from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import time
import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd
from sklearn.metrics import log_loss

from keras.layers import Input, Dense
from keras.models import Model

def main():
    # load data
    df_train = pd.read_csv('data/train_data.csv')
    df_valid = pd.read_csv('data/valid_data.csv')
    df_test = pd.read_csv('data/test_data.csv')

    feature_cols = [f for f in list(df_train) if "feature" in f]
    target_col = df_train.columns[-1]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_valid = df_valid[feature_cols].values
    y_valid = df_valid[target_col].values

    X_test = df_test[feature_cols].values

    # autoencoding to reduce some of the noise
    input_vector = Input(shape=(21,))
    encoded = Dense(14, activation='relu')(input_vector)
    encoded = Dense(7, activation='relu')(encoded)

    decoded = Dense(7, activation='relu')(encoded)
    decoded = Dense(14, activation='relu')(decoded)
    decoded = Dense(21, activation='sigmoid')(decoded)

    autoencoder = Model(input_vector, decoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)

    print("Fitting...")
    autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(X_valid, X_valid))

    print("Predicting...")
    X_denoised_train = autoencoder.predict(X_train)
    X_denoised_valid = autoencoder.predict(X_valid)
    X_denoised_test = autoencoder.predict(X_test)

    save_path = 'data/autoencoder_denoised.npz'

    np.savez(save_path, \
        train=X_denoised_train, \
        valid=X_denoised_valid, \
        test=X_denoised_test)
    print('Saved: {}'.format(save_path))

if __name__ == '__main__':
    main()
