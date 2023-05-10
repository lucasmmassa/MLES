import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils import encode

# Metrics for TFLite Model

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


interpreter = tf.lite.Interpreter('converted_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)

outputs_lite = []

for i in tqdm(range(x_test.shape[0])):
    input_data = tf.constant(x_test[i].reshape(1,1,200,1), dtype=tf.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
        
    outputs_lite.append(output_data)
    
outputs_lite = np.vstack(outputs_lite)
        
for i in range(outputs_lite.shape[0]):
    for j in range(outputs_lite.shape[1]):
        outputs_lite[i,j,:] = np.argmax(outputs_lite[i,j,:])
        y_test[i,j,:] = np.argmax(y_test[i,j,:])
        
outputs_lite = outputs_lite[:,:,0].reshape(outputs_lite.shape[0], outputs_lite.shape[1])
y_test = y_test[:,:,0].reshape(y_test.shape[0], y_test.shape[1])

results = np.equal(outputs_lite, y_test)

results = np.mean(results)

print(results)
