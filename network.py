import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import wfdb
from scipy import rand
from sklearn.model_selection import KFold
from tensorflow.keras import Sequential
from tensorflow.keras import backend as kerasBackend
from tensorflow.keras import models
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

import utils

SIZE = 200

def create_model():
    model = Sequential()
    model.add(TimeDistributed(Conv1D(16, 3, padding="same"), input_shape=(1, SIZE, 1)))
    model.add(TimeDistributed(Conv1D(32, 3, padding="same")))
    # model.add(TimeDistributed(Conv1D(128, 3, padding="same")))
    model.add(Reshape((SIZE, 32)))
    # model.add(TimeDistributed(Permute((2, 1))))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Dense(4, activation='softmax')))
    
    return model

class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, X, y,
                 batch_size,
                 input_size=(1, 1, SIZE, 1)):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.input_size = input_size
        self.n = y.shape[0]

    def shuffle_data(self): 
        print('shuffling data...')
        idx = np.random.randint(low=0, high=self.n, size=self.n)
        self.X = self.X[idx]
        self.y = self.y[idx]
        print('finished.\n')

    def __getitem__(self, index):        
        X = self.X[index * self.batch_size:(index + 1) * self.batch_size].reshape((self.batch_size, 1, SIZE, 1))
        y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

class ReshuffleDataCallback(Callback):
    def __init__(self, train_generator, validation_generator):
        super().__init__()
        self.train_generator = train_generator
        self.validation_generator = validation_generator

    def on_epoch_begin(self, epoch, logs=None):
        self.train_generator.shuffle_data()
        self.validation_generator.shuffle_data()

def read_files_from_csv(path):
    print('Reading csv files...')
    df = pd.read_csv(path)
    df = df.iloc[:int(df.shape[0]/20)]
    inputs = []
    outputs = []
    for index, row in tqdm(df.iterrows()):
        inputs.append(np.load(os.path.join('inputs', row['file']))[:,0])
        y = np.load(os.path.join('outputs', row['file']))
        y = utils.encode(y, number_of_labels=4)
        outputs.append(y)
    inputs = np.vstack(inputs)
    outputs = np.stack(outputs, axis=0)
    print('Finished')
    return (inputs, outputs)

def train_model():
    if not os.path.isdir("checkpoints"): os.mkdir("checkpoints")
    
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    
    batch_size = 16
    
    x_train, y_train = read_files_from_csv('train.csv')
    train_generator = CustomDataGen(x_train, y_train, batch_size)
    
    x_test, y_test = read_files_from_csv('test.csv')
    validation_generator = CustomDataGen(x_test, y_test, batch_size)
    
    mc = ModelCheckpoint(filepath='checkpoints/model.h5', monitor='val_loss',
                         verbose=0, save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')
    
    reshuffle_data_callback = ReshuffleDataCallback(train_generator, validation_generator)
    
    history = model.fit_generator(train_generator,
                validation_data=validation_generator,
                epochs=12, callbacks=[mc, reshuffle_data_callback],verbose=1, workers=1, use_multiprocessing=False)

    pd.DataFrame.from_dict(history.history).to_excel('history/qrs_history.xlsx', index=False)


def cross_validation():
    
    batch_size = 16
    folds = 5
    
    x, y = read_files_from_csv('train.csv')
    kf = KFold(n_splits=folds, shuffle=True) 
    
    for index, (train_indices, val_indices) in enumerate(kf.split(x, y)):
        print('*'*30)
        print(f"Training on fold {index+1}")
        print('*'*30)
        
        xtrain, xval = x[train_indices], x[val_indices]
        ytrain, yval = y[train_indices], y[val_indices]
        
        train_generator = CustomDataGen(xtrain, ytrain, batch_size)
        validation_generator = CustomDataGen(xval, yval, batch_size)
        
        model = create_model()
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])
        
        reshuffle_data_callback = ReshuffleDataCallback(train_generator, validation_generator)
        
        history = model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=15, callbacks=[reshuffle_data_callback],verbose=1, workers=1, use_multiprocessing=False)

        pd.DataFrame.from_dict(history.history).to_excel(f'fold_results/fold{index+1}.xlsx', index=False)

train_model()
# cross_validation()

