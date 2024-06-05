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


years = [2094, 2095, 2096]

for year in years:
    lkf_data = process_dataset(netcdf_file = path_nc + f'1km_{year}.nc',
                               output_path = path + 'lkfs/',
                               dog_thres = 0.01,
                               t_red = 1)

    lkf_data.detect_lkfs(indexes=[0])
    lkf_data.indexes = np.arange(365)

    lkf_data.track_lkfs(indexes=np.arange(365))


    np.save(path_ds + f'ds_{year}.npy', [lkf_data], allow_pickle=True)