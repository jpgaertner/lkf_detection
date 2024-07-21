import numpy as np
import sys
sys.path.insert(1, '/home/a/a270230/LKF/lkf_tools/lkf_tools/')
from dataset import *
import warnings
warnings.filterwarnings('ignore')
import os

res = '1km'
year = 2013
lower = 360
step = 10

path = '/work/bk1377/a270230/'
path_nc   = path + f'interpolated_fesom_output/{res}/'
path_lkfs = path + f'lkfs/{res}/'
path_detected_lkfs = path_lkfs + f'{year}_{res}'


### find the days that still need to be detected ###

# if the lkf folder does not exists, no lkfs have been detected
if not os.path.isdir(path_detected_lkfs):
    days_to_detect = np.arange(1,366)

# if the lkf folder does exists, check for the days that are missing
if os.path.isdir(path_detected_lkfs):
    # load the files of the detected lkfs
    lkf_files = os.listdir(path_detected_lkfs)
    lkf_files.sort()
    try: lkf_files.remove('.ipynb_checkpoints')
    except: pass

    # read which days have been detected
    detected_days = []
    for lkf_file in lkf_files:
        detected_days += int(lkf_file.split('.')[0].split('_')[-1]),

    # create array of the lkfs that still need to be detected
    # (the -1 converts from day to index in days_all)
    days_all = np.arange(1,366)
    days_to_detect = np.delete(days_all, np.array(detected_days) - 1) # the -1


### detect lkfs ###

lkf_data = process_dataset(netcdf_file = path_nc + f'{year}_{res}.nc',
                           output_path = path_lkfs,
                           dog_thres = 0.01,
                           t_red = 1)

lkf_data.detect_lkfs(indexes = days_to_detect[lower:lower+step])
