import os

import numpy as np
import tqdm
import wfdb


def generate_output(labels, samples, size):
    output = np.zeros(size, dtype=int)
    current_start = 0
    current_label = ''
    current_end = 0
    
    for i in range(len(labels)):
        aux = labels[i]
        
        if aux == '(':
            current_start = samples[i]
        elif aux == ')':
            current_end = samples[i]
            if current_label == 'p':
                output[current_start:current_end+1] = 1
            elif current_label == 'N':
                output[current_start:current_end+1] = 2
            else:
                output[current_start:current_end+1] = 3            
        else:
            current_label = aux
            
    return output
    
SIZE = 200

directory = "qt-database-1.0.0"
unique_names = [name.split('.')[0] for name in os.listdir(directory) if 'sel' in name]
unique_names = np.unique(unique_names)

if not os.path.isdir("inputs"): os.mkdir("inputs")
if not os.path.isdir("outputs"): os.mkdir("outputs")

for name in tqdm.tqdm(unique_names):
    path = os.path.join(directory, name)
    
    record = wfdb.io.rdrecord(path)
    signal = record.p_signal
        
    annotation = wfdb.io.rdann(path, extension='pu0')
    labels = annotation.symbol    
    
    output = generate_output(annotation.symbol, annotation.sample, signal.shape[0])
    output = generate_output(annotation.symbol, annotation.sample, signal.shape[0])
    
    X = np.split(signal, np.arange(SIZE, signal.shape[0], SIZE))
    Y = np.split(output, np.arange(SIZE, output.shape[0], SIZE))
    
    for i in range(len(X)-1):
        np.save(os.path.join('inputs', '{}_{}.npy'.format(name, i)), X[i])
        np.save(os.path.join('outputs', '{}_{}.npy'.format(name, i)), Y[i])