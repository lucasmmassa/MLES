import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.io import loadmat
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import Callback, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tqdm import tqdm


def create_model():
    # encoding
    input_signal = layers.Input(shape=(76))
    x = layers.Dense(64, activation='sigmoid')(input_signal)
    # x = layers.Dropout(.2)(x)
    # x = layers.Dense(32, activation='sigmoid')(x)
    x = layers.Dense(64, activation='sigmoid')(x)
    # x = layers.Dropout(.2)(x)
    pred = layers.Dense(1, activation='sigmoid')(x)    
    model = Model(input_signal, pred)
    return model


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
    values = np.array(values)
    # values = (values - np.min(values))/(np.max(values) - np.min(values))
    return values.astype(np.float32), np.array(labels)      


X_train, y_train = prepare_data('../training2017', 'train.csv')
# X_train = (X_train - np.min(X_train))/(np.max(X_train) - np.min(X_train))
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape, y_train.shape)

X_test, y_test = prepare_data('../sample2017/validation', 'test.csv')
# X_test = (X_test - np.min(X_test))/(np.max(X_test) - np.min(X_test))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape, y_test.shape)

autoencoder = create_model()
# autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), 
#                     loss=[losses.MeanSquaredError()])
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    loss='mse')


mc = tf.keras.callbacks.ModelCheckpoint(
    filepath="model.h5",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

filename='history.csv'
csv_callback=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

autoencoder.fit(x=X_train, y=y_train, batch_size=256, epochs=1000,
                validation_data=(X_test, y_test), shuffle=True,
                callbacks=[mc, csv_callback])

