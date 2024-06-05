import numpy as np
import sys
sys.path.insert(1, '/p/project/chhb19/gaertner2/lkf_tools/lkf_tools/')
from dataset import *
import warnings
warnings.filterwarnings('ignore')

path = '/p/project/chhb19/gaertner2/data/awicm_cvmix/'
path_ds = path + 'datasets/'

path_scratch = '/p/scratch/chhb19/gaertner2/'
path_nc = path_scratch + 'interpolated_fesom_output/1km/'
path_lkfs = path_scratch + 'lkfs/'


year = 2018

lkf_data = process_dataset(netcdf_file = path_nc + f'1km_{year}.nc',
                           output_path = path_lkfs,
                           dog_thres = 0.01,
                           t_red = 1)

lkf_data.detect_lkfs(indexes=np.arange(365), use_eps=True)

np.save(path_ds + f'ds_{year}', [lkf_data])