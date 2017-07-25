from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import numpy as np
import pandas as pd

import os

preds = os.listdir('predictions')

def main():
    ids = []
    probs = []
    for p in preds:
        df = pd.read_csv(f)
        ids = df['id'].values
        probs.append(df['probability'].values)

    probability = np.power(np.prod(probs, axis=0), 1.0 / len(preds))
    assert(len(probability) == len(ids))

    df_pred = pd.DataFrame({
        'id': ids,
        'probability': probability,
    })
    csv_path = 'predictions_ensemble_{}.csv'.format(int(time.time()))
    df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
