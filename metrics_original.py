import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils import encode

# Metrics for original model

def read_files_from_csv(path):
    print('Reading csv files...')
    df = pd.read_csv(path)
    df = df.iloc[int(df.shape[0]/20):int(df.shape[0]/10)]
    inputs = []
    outputs = []
    for _, row in tqdm(df.iterrows()):
        inputs.append(np.load(os.path.join('inputs', row['file']))[:,0])
        y = np.load(os.path.join('outputs', row['file']))
        y = encode(y, number_of_labels=4)
        outputs.append(y)
    inputs = np.vstack(inputs)
    outputs = np.stack(outputs, axis=0)
    print('Finished')
    return (inputs, outputs)


x_test, y_test = read_files_from_csv('test.csv')

model = tf.keras.models.load_model('checkpoints/model.h5')

x_test = x_test.reshape(x_test.shape[0], 1, 200, 1)

output = model.predict(x_test)

print('output= ', output)
print('output= ', output)


for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        output[i,j,:] = np.argmax(output[i,j,:])
        y_test[i,j,:] = np.argmax(y_test[i,j,:])
        
output = output[:,:,0].reshape(output.shape[0], output.shape[1])
y_test = y_test[:,:,0].reshape(y_test.shape[0], y_test.shape[1])

print('metrics= ', output)
print('metrics= ', output.shape)

results = np.equal(output, y_test)

results = np.mean(results)

print(results)
