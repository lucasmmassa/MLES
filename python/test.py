import os
import struct
import time

import numpy as np
import pandas as pd
import serial
from scipy.io import loadmat

ser = serial.Serial()
ser.baudrate = 115200
ser.port = 'COM3'
ser.setDTR(False)
ser.setRTS(False)
ser.open()

data_path = '../training2017'
csv_path = 'train.csv'

df = pd.read_csv(csv_path)
for _, row in (df.iterrows()):
    file = row['file']
    label = row['label']
    mat = loadmat(os.path.join(data_path, f"{file}.mat"))
    x = mat['val']
    x = np.squeeze(x)[:1212][1::16]
    break

x = x.astype(np.float32)
size = x.shape[0]
# print(size)
idx = 0

while True:
    string = ','.join(str(el) for el in x) + ','
    print("SENDING: ", string, '\n')
    ser.write(string.encode())
    data = ser.read(4)
    if data:
        try:
            data = float(data)
            label = 'normal' if data >= 0.5 else "arrhythmia"
            print(f'RESPONSE: {data} --> {label}')
        except: pass
        break
     
ser.close()