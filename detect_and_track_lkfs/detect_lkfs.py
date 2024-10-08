import numpy as np
import sys
sys.path.insert(1, '../../lkf_tools/lkf_tools/')
from dataset import *
import warnings
warnings.filterwarnings('ignore')


res = '4km'

path = '/work/bk1377/a270230/'
path_nc   = path + f'interpolated_fesom_output/{res}/'
path_lkfs = path + 'lkfs/'
path_ds   = path + 'datasets/'

year = 2013

lkf_data = process_dataset(netcdf_file = path_nc + f'{year}_{res}.nc',
                           output_path = path_lkfs,
                           dog_thres = 0.01,
                           t_red = 1)

lkf_data.detect_lkfs(indexes=np.arange(365))

np.save(path_ds + f'ds_{year}_{res}', [lkf_data])
