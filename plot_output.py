import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from utils import find_runs

DATA_PATH = 'test.csv'
VALUES_PATH = 'base_pred.npy'

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
    

for file in os.listdir('inputs'):
    y = np.load(os.path.join('outputs', file))
    x = np.load(os.path.join('inputs', file))[:, 0]
    plot_sample(x, y, False)

# name = '100_1'
# x = np.load(os.path.join('bih_inputs', '{}.npy'.format(name)))[:, 0]

# directory = "bih-database-1.0.0"
# name = '100'

# symbols = np.load(os.path.join('bih_symbols', '{}.npy'.format(name)))
# samples = np.load(os.path.join('bih_samples', '{}.npy'.format(name)))

# plt.plot(x)

# i = 0
# while True:
#     idx = samples[i]
#     if idx >=1000:
#         break
    
#     symbol = symbols[i]
#     if symbol == 'N':
#         plt.axvline(x=idx, color='red')
        
#     i+=1

# plt.show()
