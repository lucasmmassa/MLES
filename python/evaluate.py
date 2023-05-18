import os

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from scipy.io import loadmat
from tensorflow.keras.models import load_model
from tqdm import tqdm


def prepare_data(data_path, csv_path):
    df = pd.read_csv(csv_path)
    values = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        file = row['file']
        label = row['label']
        mat = loadmat(os.path.join(data_path, f"{file}.mat"))
        x = mat['val']
        x = np.squeeze(x)[:1212][1::16]
        values.append(x)
        if label == "N":
            labels.append(1)
        else:
            labels.append(0)
    return np.array(values), np.array(labels)

X_train, y_train = prepare_data('../training2017', 'train.csv')
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# print(X_train.shape, y_train.shape)

# X_test, y_test = prepare_data('../sample2017/validation', 'test.csv')
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
# print(X_test.shape, y_test.shape)

model = load_model('model.h5')
pred = model.predict(X_train)
# pred = model.predict(X_test)
pred[pred<0.5] = 0
pred[pred>0] = 1
print(np.mean(pred==y_train))
# print(np.mean(pred==y_test))