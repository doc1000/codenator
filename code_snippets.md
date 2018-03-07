code snippents

import matplotlib.pyplot as plt
import seaborn
import numpy as np
%matplotlib inline

from scipy import stats

### How about some legible text in our graph?
import matplotlib as mpl
mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
})

def bootstrap():
  x = [2,4,3,5,7,2]
  n = len(x)
  bootstraps = 20
  mean_x = np.array([np.mean([x[np.random.randint(n)] for i in np.arange(n)]) for i in np.arange(bootstraps)])
  np.mean(mean_x)
