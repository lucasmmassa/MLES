import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.lines import Line2D
from tqdm import tqdm

from utils import encode, find_runs

DATA_PATH = 'test.csv'
VALUES_PATH = 'base_pred.npy'

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

def plot_sample(x, y, pvc=False):
    
    colors = [
        (0.7, 0.7, 0, 0.3),
        (1, 0, 0, 0.3),
        (0, 0.7, 0, 0.3),
        (0, 0, 1, 0.3)
    ]
    
    legends = [
        'No wave',
        'P',
        'QRS',
        'T'
    ]
    
    values, idx, leng = find_runs(y)

    plt.plot(x, color='black')
    
    if pvc:
        plt.axvline(x = 500, color='red')
    
    for i in range(len(values)):
        plt.axvspan(idx[i], idx[i]+leng[i], facecolor=colors[values[i]], alpha=0.3, label=legends[values[i]])
        
        
    custom_lines = [Line2D([0], [0], color=colors[0], lw=10),
                    Line2D([0], [0], color=colors[1], lw=10),
                    Line2D([0], [0], color=colors[2], lw=10),
                    Line2D([0], [0], color=colors[3], lw=10)]
    plt.legend(custom_lines, legends)
    # leg = plt.legend()
    # for lh in leg.legendHandles: 
    #     lh._legmarker.set_alpha(1)

    plt.show()
    
x_test, y_test = read_files_from_csv('test.csv')
model = tf.keras.models.load_model('checkpoints/model.h5')
x_test = x_test.reshape(x_test.shape[0], 1, 200, 1)
output = model.predict(x_test)

for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        output[i,j,:] = np.argmax(output[i,j,:])
        y_test[i,j,:] = np.argmax(y_test[i,j,:])
        
output = output[:,:,0].reshape(output.shape[0], output.shape[1])
y_test = y_test[:,:,0].reshape(y_test.shape[0], y_test.shape[1])
x_test = x_test.reshape(x_test.shape[0], 200)

interpreter = tf.lite.Interpreter('converted_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
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
        # y_test[i,j,:] = np.argmax(y_test[i,j,:])
        
outputs_lite = outputs_lite[:,:,0].reshape(outputs_lite.shape[0], outputs_lite.shape[1])
# y_test = y_test[:,:,0].reshape(y_test.shape[0], y_test.shape[1])

for i in range(5):
    plot_sample(x_test[i], output[i].astype(int))
    plot_sample(x_test[i], outputs_lite[i].astype(int))