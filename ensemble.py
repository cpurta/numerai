from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import numpy as np
import pandas as pd

paths = [
    'predictions/predictions_1500400687_0.693150778975.csv',
    'predictions/predictions_1500400931_0.692735149191.csv',
    'predictions/predictions_1500401397_0.692893209636.csv',
    'predictions/predictions_1500405562_0.693071552363.csv'
]

def main():
    ids = []
    probs = []
    for path in paths:
        df = pd.read_csv(path)
        ids = df['id'].values
        probs.append(df['probability'].values)

    probability = np.power(np.prod(probs, axis=0), 1.0 / len(paths))
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
