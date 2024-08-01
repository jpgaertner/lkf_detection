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

# select the years you want to analyze
years = [i for i in range(2013,2021)]
years += [i for i in range(2093,2101)]

# load mean ice concentration, total ice covered area, mean ice thickness,
# and total ice volume for all years of the model run (1986 - 2100 for 4km,
# 2013 - 2020 & 2093 - 2100 for 1km)
data = np.load(path_stat + f'lkfs_paths_{res}.npz', allow_pickle=True)
years, lkfs, paths, paths_all = [data[key] for key in data.files]

# load mean ice concentration, total ice covered area, mean ice thickness,
# and total ice volume for all years
a_mean, area_total, h_mean, ice_vol_total, years_all = np.load(
    path_stat + f'a_mean_tot_h_mean_tot_{res}.npy', allow_pickle=True)

# make an array of the right np.shape out of area_total
arr = np.zeros((len(area_total),365))
for year, area_total_year in enumerate(area_total):
    arr[year,:] = area_total_year
area_total = arr

# only use the selected years
inds = [np.where(years_all==year)[0][0] for year in years]
area_total = area_total[inds]

# use already calculated resolutions (can be calculated either from the nc files like in
# plot/area_thickness.ipynb, or from the lkf_data objects like in statistics_main.ipynb)
if res == '4km': res_km = 4.337849218906646
if res == '1km': res_km = 1.083648783567869

# calculate metrics
n_lkfs = get_n_lkfs(lkfs)
rho_lkfs = n_lkfs / area_total * 10000
length, mean_length, total_length = get_lkf_length(lkfs, res_km)
lifetimes, mean_lifetime = get_lkf_lifetimes(paths)
# lifetimes_all includes the lifetimes of LKFs that are
# already counted in previous timesteps
lifetimes_all, _ = get_lkf_lifetimes(paths_all)

# create lkf dictionary only if it does not already exist
try: LKFs = np.load(path_stat + f'LKFs_{res}.npy', allow_pickle=True)[0]
except: LKFs = dict()

if True:
    # calculate decadal mean and standart deviation of each lkf variable
    # and store it in the dictionary
    for ystart, yend in zip([2013, 2093], [2020, 2100]):

        df = pd.DataFrame()
        df['number av'], df['number sd']               = av_sd(n_lkfs, ystart, yend, years)
        df['density av'], df['density sd']             = av_sd(rho_lkfs, ystart, yend, years)
        df['mean length av'], df['mean length sd']     = av_sd(mean_length, ystart, yend, years)
        df['total length av'], df['total length sd']   = av_sd(total_length, ystart, yend, years)
        df['mean lifetime av'], df['mean lifetime sd'] = av_sd(mean_lifetime, ystart, yend, years)

        decade = f'{ystart} - {yend}'
        LKFs[decade] = df

# store each lkf variable for each year
for y, year in enumerate(years):
    df_y = pd.DataFrame(n_lkfs[y], columns=['number'])
    df_y['density']       = rho_lkfs[y]
    df_y['mean length']   = mean_length[y]
    df_y['total length']  = total_length[y]
    df_y['mean lifetime'] = mean_lifetime[y]
    
    LKFs[f'{year}'] = df_y
    
    # shift indices so they go from 1 to 365 instead of from 0 to 364
    LKFs[f'{year}'].index = LKFs[f'{year}'].index + 1
    
    # store variables of individual lkfs
    LKFs[f'{year} daily'] = dict()
    for d in range(365):
        df_d = pd.DataFrame(length[y][d], columns=['length'])
        df_d['lifetime'] = lifetimes_all[y][d]
        
        LKFs[f'{year} daily'][f'{d+1}'] = df_d

        
np.save(path_stat + f'LKFs_{res}.npy', [LKFs])
