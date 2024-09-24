import numpy as np
import sys
sys.path.insert(1, '../../lkf_tools/lkf_tools/')
from dataset import *
import warnings
warnings.filterwarnings('ignore')


res = '4km'

path = '/work/bk1377/a270230/'
path_ds = path + 'datasets/'

year = 2013

lkf_data = np.load(path_ds + f'ds_{year}_{res}.npy', allow_pickle=True)[0]

lkf_data.track_lkfs(indexes=np.arange(365))

np.save(path_ds + f'ds_{year}_{res}', [lkf_data])
