import os

import numpy as np
import pandas as pd


def generate_csv():
    files = os.listdir('inputs\\')
    train = []
    test = []

    size = int(len(files)*0.8)
    # idx = np.random.randint(low=0, high=len(files), size=size, dtype=int).tolist()
    idx = np.arange(len(files), dtype=int)
    idx = np.random.choice(idx, size, replace=False)

    for i in range(len(files)):
        if i in idx:
            train.append(files[i])
        else:
            test.append(files[i])
            
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    train_df['file'] = train
    test_df['file'] = test

    print(train_df.shape)
    print(test_df.shape)

    train_df.to_csv('train.csv')
    test_df.to_csv('test.csv')

def encode(array, number_of_labels):
    result = np.zeros((array.shape[0], number_of_labels), dtype=int)
    for i in range(array.shape[0]):
        result[i, array[i]] = 1
    return result

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
