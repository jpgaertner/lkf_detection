import numpy as np
import sys
sys.path.insert(1, '/home/a/a270230/LKF/lkf_tools/lkf_tools/')
from dataset import *
import warnings
warnings.filterwarnings('ignore')

path = '/work/bk1377/a270230/'
path_ds = path + 'datasets/'

year = 2013
res = '4km'

lkf_data = np.load(path_ds + f'ds_{year}_{res}.npy', allow_pickle=True)[0]

lkf_data.track_lkfs(indexes=np.arange(365))
