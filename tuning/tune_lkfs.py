import numpy as np
import sys
sys.path.insert(1, '/p/project/chhb19/gaertner2/lkf_tools/lkf_tools/')
from dataset import *
import warnings
warnings.filterwarnings('ignore')

path = '/p/project/chhb19/gaertner2/data/awicm_cvmix/'
path_ds = path + 'datasets_tuning/'
path_lkfs = path + 'lkfs_tuning/'

path_scratch = '/p/scratch/chhb19/gaertner2/'
path_nc = path_scratch + 'interpolated_fesom_output/1km/'


year = 2095

dog = 20.0
thresh = 0.7

### use_eps = False for dogs like 20, use_eps = True for dogs like 0.05 ###

lkf_data = process_dataset(netcdf_file = path_nc + f'1km_{year}.nc',
                           output_path = path_lkfs + f'dog{dog}_new{thresh}',
                           dog_thres = dog, aice_thresh = thresh,
                           t_red = 1)

lkf_data.detect_lkfs(indexes=np.arange(0,365,15), use_eps=False)

np.save(path_ds + f'ds_F{dog}_{thresh}_{year}_new', [lkf_data])