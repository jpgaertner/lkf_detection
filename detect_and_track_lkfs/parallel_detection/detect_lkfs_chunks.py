import numpy as np
import sys
sys.path.insert(1, '/home/a/a270230/LKF/lkf_tools/lkf_tools/')
from dataset import *
import warnings
warnings.filterwarnings('ignore')

res = '1km'
path = '/work/bk1377/a270230/'
path_nc   = path + f'interpolated_fesom_output/{res}/'
path_lkfs = path + f'lkfs/{res}/'

year = 2016
lower = 360
step = 20

lkf_data = process_dataset(netcdf_file = path_nc + f'{year}_{res}.nc',
                           output_path = path_lkfs,
                           dog_thres = 0.01,
                           t_red = 1)

lkf_data.detect_lkfs(indexes=np.arange(lower,lower+step))
