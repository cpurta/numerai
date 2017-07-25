from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression

reduction_dim = 2

def main():
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

    estimator = LogisticRegression(C=10.0)
    rfe = feature_selection.RFE(estimator=estimator, n_features_to_select=reduction_dim, verbose=1)
    print('Fitting Recursive Feature Elimination on data...')
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_valid_rfe = rfe.fit_transform(X_valid, y_valid)
    X_test_rfe  = rfe.transform(X_test)

    print('Saving...')

    save_path = 'data/rfe_selection_data_{}d.npz'.format(reduction_dim)
    np.savez(save_path, \
        train=X_train_rfe, \
        valid=X_valid_rfe, \
        test=X_test_rfe)

if __name__ == '__main__':
    main()
