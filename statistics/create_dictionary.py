import numpy as np
import sys
sys.path.append('../functions/')
from statistics_functions import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


res = '4km'

path = '/work/bk1377/a270230/'
path_stat = path + 'statistics/'

# load mean ice concentration, total ice covered area, mean ice thickness,
# and total ice volume for all analyzed years 
data = np.load(path_stat + f'lkfs_paths_{res}_all.npz', allow_pickle=True)
years, lkfs, paths, paths_all = [data[key] for key in data.files]

# load mean ice concentration, total ice covered area, mean ice thickness,
# and total ice volume for all years of the model run (1986 - 2100 for 4km,
# 2013 - 2020 & 2093 - 2100 for 1km)
a_mean, area_total, h_mean, ice_vol_total, years_all = np.load(
    path_stat + f'ice_area_thickness_{res}.npy', allow_pickle=True)

# make an array of the right np.shape out of area_total
arr = np.zeros((len(area_total),365))
for year, area_total_year in enumerate(area_total):
    arr[year,:] = area_total_year
area_total = arr

# only use the analyzed years
inds = [np.where(years_all==year)[0][0] for year in years]
area_total = area_total[inds]

# use already calculated resolutions (can be calculated either from the nc files like in
# plot/area_thickness.ipynb, or from the lkf_data objects like in functions/statistics_functions.ipynb)
if res == '4km': res_km = 4.337849218906646
if res == '1km': res_km = 1.083648783567869

# calculate metrics
n_lkfs = get_n_lkfs(lkfs)
n_lkfs_per_area = n_lkfs / area_total * 10000 # in 1 / 10000 km2
length, mean_length, mean_length_sd, total_length = get_lkf_length(lkfs, res_km) # in km
total_length_per_area = total_length / area_total * 10000 # in km / 10000 km2
lifetimes, mean_lifetime, mean_lifetime_sd = get_lkf_lifetimes(paths) # in days
# lifetimes_all includes the lifetimes of LKFs that are
# already counted in previous timesteps
lifetimes_all, _, _ = get_lkf_lifetimes(paths_all) # in days

# create lkf dictionary
LKFs = dict()

# set this to false for the 1986 to 2100 run
if False:
    # calculate mean and standart deviation of each lkf variable
    # for the two time periods where the 1 km data is available
    for ystart, yend in zip([2013, 2093], [2020, 2100]):

        # if the variable has no uncertainty in a dataset/ year (eg number of lkfs at a specfic day),
        # the interannual mean is the arithmetic mean.
        # if the variable has an uncertainty in a dataset/ year (eg mean length of lkfs at a specfic day),
        # the interannual mean is calculated via inverse variance weighting
        df = pd.DataFrame()
        df['number av'], df['number sd'] = interannual_mean(n_lkfs, ystart, yend, years)
        df['number per area av'], df['number per area sd'] = interannual_mean(n_lkfs_per_area, ystart, yend, years)
        df['mean length av'], df['mean length sd'] = interannual_mean_weighted(mean_length, mean_length_sd, ystart, yend, years)
        df['total length av'], df['total length sd'] = interannual_mean(total_length, ystart, yend, years)
        df['total length per area av'], df['total length per area sd'] = interannual_mean(total_length_per_area, ystart, yend, years)
        df['mean lifetime av'], df['mean lifetime sd'] = interannual_mean_weighted(mean_lifetime, mean_lifetime_sd, ystart, yend, years)

        LKFs[f'{ystart} - {yend}'] = df

# store each lkf variable for each year
for y, year in enumerate(years):
    df_y = pd.DataFrame()
    df_y['number']           = n_lkfs[y]
    df_y['number per area']  = n_lkfs_per_area[y]
    df_y['mean length']      = mean_length[y]
    df_y['mean length sd']   = mean_length_sd[y]
    df_y['total length']     = total_length[y]
    df_y['total length per area'] = total_length_per_area[y]
    df_y['mean lifetime']    = mean_lifetime[y]
    df_y['mean lifetime sd'] = mean_lifetime_sd[y]
    
    LKFs[f'{year}'] = df_y
    
    # shift indices so they go from 1 to 365 instead of from 0 to 364
    LKFs[f'{year}'].index = LKFs[f'{year}'].index + 1
    
    # store variables of individual lkfs
    LKFs[f'{year} daily'] = dict()
    for d in range(365):
        df_d = pd.DataFrame(length[y][d], columns=['length'])
        df_d['lifetime'] = lifetimes_all[y][d]
        
        LKFs[f'{year} daily'][f'{d+1}'] = df_d

LKFs['res_km'] = res_km

np.save(path_stat + f'LKFs_{res}.npy', [LKFs])
