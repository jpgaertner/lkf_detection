import numpy as np
import sys
sys.path.insert(1, '/home/a/a270230/LKF/lkf_tools/lkf_tools/')
from dataset import *
import warnings
warnings.filterwarnings('ignore')

res = '1km'
year = 2016
startday = 360
step = 10

path = '/work/bk1377/a270230/'
path_nc   = path + f'interpolated_fesom_output/{res}/'
path_lkfs = path + f'lkfs/{res}/'

days_to_detect = np.array([170,177,187,196,197,204,207,208,209,210,214,219,224,225,226,235,236,237,238,239,244,249,250,255,256,257,265,268,269,276,286,289,290,297,307,316,317,318,330,340,360])

startday_index = np.where(days_to_detect==startday)[0][0]

indizes_to_detect = days_to_detect - 1

lkf_data = process_dataset(netcdf_file = path_nc + f'{year}_{res}.nc',
                           output_path = path_lkfs,
                           dog_thres = 0.01,
                           t_red = 1)

lkf_data.detect_lkfs(indexes = indizes_to_detect[startday_index:startday_index+step])
