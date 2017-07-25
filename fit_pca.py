from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd
from sklearn import decomposition

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

    X_all = np.concatenate([X_train, X_valid, X_test], axis=0)

    pca = decomposition.PCA(n_components=7, svd_solver='full')
    print('Fitting PCA on data...')
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.fit_transform(X_valid)
    X_test_pca  = pca.fit_transform(X_test)

    print(X_train_pca[:10])
    print(X_valid_pca[:10])
    print(X_test_pca[:10])

    print('Saving...')
    np.savez('data/pca_data.npz', \
        train=X_train_pca, \
        valid=X_valid_pca, \
        test=X_test_pca)

if __name__ == '__main__':
    main()
