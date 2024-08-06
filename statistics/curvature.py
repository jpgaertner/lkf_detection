import sys
sys.path.insert(1, '../functions/')
from statistics_functions import *
import numpy as np
import warnings
warnings.filterwarnings('ignore')


res = '4km'

path_stat = '/work/bk1377/a270230/statistics/'

# load the lkf data for all analyzed years
data = np.load(path_stat + f'lkfs_paths_{res}.npz', allow_pickle=True)
years, lkfs, paths, paths_all = [data[key] for key in data.files]

LKFs = np.load(path_stat + f'LKFs_{res}.npy', allow_pickle=True)[0]

# use already calculated resolutions (can be calculated either from the nc files like in
# plot/area_thickness.ipynb, or from the lkf_data objects like in statistics_main.ipynb)
if res == '4km': res_km = 4.337849218906646
if res == '1km': res_km = 1.083648783567869

# calculate the curvature of the lkfs according to
# curvature = 1 - distance_between_start_and_end_of_lkf / length_of_lkf
# a curvature of 0 thus corresponds to a straigth line
for y, year in enumerate(years):

    for d in range(365):

        curv = []
        if (len(lkfs[y][d])==0):
            # if no lkfs are detected for the current day, set it to nan
            LKFs[f'{year} daily'][f'{d+1}']['curvature'] = np.nan
        else:
            # calculate the curvature for the individual lkfs
            for i, lkf in enumerate(lkfs[y][d]):
                start = np.array([lkf[0,0], lkf[0,1]], dtype='int')
                end = np.array([lkf[-1,0], lkf[-1,1]], dtype='int')

                distance = np.sqrt( (end-start)[0]**2 + (end-start)[1]**2 ) * res_km

                LKFs[f'{year} daily'][f'{d+1}'].loc[i,'curvature'] = 1 - distance / LKFs[f'{year} daily'][f'{d+1}'].loc[i, 'length']

# calculate the mean curvature for each day
for year in years:
    for d in range(365):
        LKFs[f'{year}'].loc[d+1,'mean curvature'] = np.nanmean(LKFs[f'{year} daily'][f'{d+1}']['curvature'])

if True:
    # collect all mean curvatures in a single array, then calculate the interannual
    # mean and standart deviation for the two time periods and write them into the dictionary
    mean_curvature = []
    for year in years:
        mean_curvature += LKFs[f'{year}']['mean curvature'],

    for ystart, yend in zip([2013, 2093], [2020, 2100]):
        LKFs[f'{ystart} - {yend}']['mean curvature av'], LKFs[f'{ystart} - {yend}']['mean curvature sd'] = (
            av_sd(mean_curvature, ystart, yend, years)
        )
    
np.save(path_stat + f'LKFs_{res}.npy', [LKFs])
