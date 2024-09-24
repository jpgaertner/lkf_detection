import sys
sys.path.insert(1, '../functions/')
from statistics_functions import *
import numpy as np
import warnings
warnings.filterwarnings('ignore')


res = '4km'

path_stat = '/work/bk1377/a270230/statistics/'

# load the lkf data for all analyzed years
data = np.load(path_stat + f'lkfs_paths_{res}_all.npz', allow_pickle=True)
years, lkfs, paths, paths_all = [data[key] for key in data.files]

LKFs = np.load(path_stat + f'LKFs_{res}.npy', allow_pickle=True)[0]
res_km = LKFs['res_km']

# calculate the curvature of the lkfs according to
# curvature = 1 - distance_between_start_and_end_of_lkf / length_of_lkf,
# a curvature of 0 thus corresponds to a straigth line
for y, year in enumerate(years):

    for d in range(365):
        # save the dataframe of the current day to avoid repeated dictionary lookups
        daily_lkfs = LKFs[f'{year} daily'][f'{d+1}']
        
        if len(lkfs[y][d])==0:
            # if no lkfs are detected for the current day, set it to nan
            daily_lkfs['curvature'] = np.nan
            continue

        for i_lkf, lkf in enumerate(lkfs[y][d]):
            # get the pixel coordinates of the start and end point of the lkf
            start = np.array([lkf[0,0], lkf[0,1]])
            end = np.array([lkf[-1,0], lkf[-1,1]])

            # euklidian norm scaled with the spatial resolution
            distance = np.linalg.norm(end - start) * res_km

            # add curvature to the dataframe
            daily_lkfs.loc[i_lkf,'curvature'] = 1 - distance / daily_lkfs.loc[i_lkf, 'length']

# calculate the mean curvature and standard deviation for each day
for year in years:
    for d in range(365):
        LKFs[f'{year}'].loc[d+1,'mean curvature'] = np.nanmean(LKFs[f'{year} daily'][f'{d+1}']['curvature'])
        LKFs[f'{year}'].loc[d+1,'mean curvature sd'] = np.nanstd(LKFs[f'{year} daily'][f'{d+1}']['curvature'])

if False:
    # collect all mean curvatures in a single array, then calculate the weighted interannual
    # mean for the two time periods and write them into the dictionary
    mean_curvature, mean_curvature_sd = [], []
    for year in years:
        mean_curvature += LKFs[f'{year}']['mean curvature'],
        mean_curvature_sd += LKFs[f'{year}']['mean curvature sd'],


    for ystart, yend in zip([2013, 2093], [2020, 2100]):
        LKFs[f'{ystart} - {yend}']['mean curvature av'], LKFs[f'{ystart} - {yend}']['mean curvature sd'] = (
            interannual_mean_weighted(mean_curvature, mean_curvature_sd, ystart, yend, years)
        )

np.save(path_stat + f'LKFs_{res}.npy', [LKFs])
