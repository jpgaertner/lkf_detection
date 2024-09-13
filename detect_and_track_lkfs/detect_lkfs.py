import numpy as np
import sys
sys.path.insert(1, '/home/a/a270230/LKF/lkf_tools/lkf_tools/')
from dataset import *
import warnings
warnings.filterwarnings('ignore')

path = '/work/bk1377/a270230/'
path_nc   = path + 'interpolated_fesom_output/1km/'
path_lkfs = path + 'lkfs/'
path_ds   = path + 'datasets/'

year = 2013

lkf_data = process_dataset(netcdf_file = path_nc + f'{year}_1km.nc',
                           output_path = path_lkfs,
                           dog_thres = 0.01,
                           t_red = 1)

lkf_data.detect_lkfs(indexes=np.arange(29,100,1), use_eps=True)

np.save(path_ds + f'ds_{year}_1km', [lkf_data])