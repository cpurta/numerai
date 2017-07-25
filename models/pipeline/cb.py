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
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

from keras.layers import Input, Dense

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

    tsne_3d_50p = np.load('data/tsne_3d_50p.npz')

    tsne_3d_50p_train = tsne_3d_50p['train']
    tsne_3d_50p_valid = tsne_3d_50p['valid']
    tsne_3d_50p_test  = tsne_3d_50p['test']

    X_train_concat = np.concatenate((X_train, tsne_3d_50p_train), axis=1)
    X_valid_concat = np.concatenate((X_valid, tsne_3d_50p_valid), axis=1)
    X_test_concat  = np.concatenate((X_test, tsne_3d_50p_test), axis=1)

    parameter_grid = [
        {'learning_rate':[0.005, 0.008, 0.01, 0.03, 0.05], 'iterations':[500, 1000, 1500, 2000], 'depth':[6, 7, 8, 9, 10]},
        {'learning_rate':[0.01,0.02,0.03], 'border_count':range(32,40), 'l2_leaf_reg':range(3,5)},
    ]

    X_search = np.concatenate([
            np.concatenate([X_train, tsne_3d_50p_train], axis=1),
                np.concatenate([X_valid, tsne_3d_50p_valid], axis=1),
                ], axis=0)
    y_search = np.concatenate([y_train, y_valid], axis=0)

    classifier = CatBoostClassifier(n_jobs=1)

    search = GridSearchCV(classifier, parameter_grid, verbose=1)

    print('Tuning hyperparameters...')
    search.fit(X_search, y_search)

    print('Found best parameters:')
    print(search.best_score_)
    print(search.best_params_)

    classifier = CatBoostClassifier( \
        learning_rate=search.best_params_['learning_rate'], \
        iterations=search.best_params_['iterations'], \
        depth=search.best_params_['depth'])

    print('Fitting...')
    start_time = time.time()
    classifier.fit(X_train_concat, y_train)
    print('Fit: {}s'.format(time.time() - start_time))

    p_valid = classifier.predict_proba(X_valid_concat)
    loss = log_loss(y_valid, p_valid)
    print('Loss: {}'.format(loss))

    p_test = classifier.predict_proba(X_test_concat)
    df_pred = pd.DataFrame({
        'id': df_test['id'],
        'probability': p_test[:,1]
    })
    csv_path = 'predictions/predictions_{}_{}.csv'.format(int(time.time()), loss)
    df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))


if __name__ == '__main__':
    main()
