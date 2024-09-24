import numpy as np
import sys
sys.path.insert(1, '../../lkf_tools/lkf_tools/')
from dataset import *
import warnings
warnings.filterwarnings('ignore')

res = '1km'
path_ds = f'/work/bk1377/a270230/datasets/{res}/'

year = 2097
lower = 360
step = 10

lkf_data = np.load(path_ds + f'ds_{year}_{res}.npy', allow_pickle=True)[0]

lkf_data.track_lkfs(indexes=np.arange(lower,lower+step+1))