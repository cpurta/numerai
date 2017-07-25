from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

import os

def main():
    df_train = pd.read_csv('data/train_data.csv')
    df_valid = pd.read_csv('data/valid_data.csv')
    df_test  = pd.read_csv('data/test_data.csv')

    feature_cols = [f for f in list(df_train) if "feature" in f]
    target_col = df_train.columns[-1]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_valid = df_valid[feature_cols]
    y_valid = df_valid[target_col]

    X_test = df_test[feature_cols]

    clf1 = LogisticRegression(C=1e-2, penalty='l2', n_jobs=-1)
    clf2 = RandomForestClassifier(n_jobs=-1, warm_start=True)
    clf3 = CatBoostClassifier(learning_rate=1e-2)

    ensemble = VotingClassifier( \
        estimators=[('lr', clf1), ('rf', clf2), ('cb', clf3)], \
        voting='soft', \
        n_jobs=-1)

    print('Fitting...')
    start_time = time.time()
    ensemble.fit(X_train, y_train)
    print('Fit: {}s'.format(time.time() - start_time))

    p_valid = ensemble.predict_proba(X_valid)
    loss = log_loss(y_valid, p_valid)
    print('Loss: {}'.format(loss))

    p_test = ensemble.predict_proba(X_test)
    df_pred = pd.DataFrame({
        'id': df_test['id'],
        'probability': p_test[:,1]
    })
    csv_path = 'predictions/predictions_{}_{}.csv'.format(int(time.time()), loss)
    df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
