import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt

# colors = ('#1f77b4,#ff7f0e,#3ca02c,#d62728,''#9467bd,#8c564b,#e377c2,''#7f7f7f,#bcbd22,#17becf')
markers = 's^oxv<>'

def generate_random_data(size=60): 
    x = np.linspace(-2, 0.5, size//2)
    y = np.linspace(-2, 0.5, size//2)
    x_offset = np.random.normal(1, 1, size=x.shape) 
    y_offset = np.random.normal(1, 1, size=y.shape) 
    x1 = np.vstack([x+ x_offset, y + y_offset]).T
    y1 = np.ones(len(x1))

    x = np.linspace(-2, 6, size//2)
    y = -x + 5
    x_offset = np.random.normal(1, 1, size=x.shape) 
    y_offset = np.random.normal(1, 1, size=y.shape)
    x2 = np.vstack([x+ x_offset, y + y_offset]).T
    y2 = np.zeros(len(x2))

    X = np.vstack([x1, x2])
    y = np.hstack([y1, y2])
    
    shuffled_indx = np.random.permutation(len(X)) 
    
    df_dict = {
        "feature1" : X[shuffled_indx, 0],
        "feature2" : X[shuffled_indx, 1],
        "label": y[shuffled_indx]
    }
    return pd.DataFrame(df_dict)

def set_default_plotting():
    # change matplotlib default behaviour
    plt.rcParams['figure.figsize'] = [8.0, 6.0]
    plt.rcParams['lines.markersize'] = np.sqrt(100)
    plt.rcParams['lines.markeredgecolor'] = 'black'
    plt.rcParams['font.size'] = 16
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['axes.labelsize'] = 'medium'
    plt.rcParams['axes.titlesize'] = 'large'
    # plt.rcParams['axes.grid'] = False
    
