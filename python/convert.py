import os

import numpy as np
import pandas as pd
import tensorflow as tf
from everywhereml.code_generators.tensorflow import tf_porter
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
        x = np.squeeze(x)[:1212][1::4]
        values.append(x)
        if label == "N":
            labels.append(1)
        else:
            labels.append(0)
    return np.array(values), np.array(labels)

X_test, y_test = prepare_data('../sample2017/validation', 'test.csv')

model = load_model('model.h5')

porter = tf_porter(model, X_test, y_test)
cpp_code = porter.to_cpp(instance_name='ECGModel', arena_size=4096)

text_file = open("ECGModel.h", "w")
n = text_file.write(cpp_code)
text_file.close()