import sys
sys.path.append('../functions/')
from statistics_functions import *
sys.path.append('../../lkf_tools/lkf_tools/')
from dataset import *
import numpy as np


res = '4km'

path = '/work/bk1377/a270230/'
path_stat = path + 'statistics/'
path_ds = path + f'datasets/{res}/'

# select the years you want to analyze
years = [i for i in range(2013,2021)]
years += [i for i in range(2093,2101)]

# load mean ice concentration, total ice covered area, mean ice thickness,
# and total ice volume for all years of the model run (1986 - 2100 for 4km,
# 2013 - 2020 & 2093 - 2100 for 1km)
a_mean, area_total, h_mean, ice_vol_total, years_all = np.load(
    path_stat + f'a_mean_tot_h_mean_tot_{res}.npy', allow_pickle=True)

# make an array of the right np.shape out of area_total
arr = np.zeros((len(area_total),365))
for year, area_total_year in enumerate(area_total):
    arr[year,:] = area_total_year
area_total = arr

inds = [np.where(years_all==year)[0][0] for year in years]
area_total = area_total[inds]

# load the lkfs from the lkf_data files
files = [path_ds + f'ds_{year}_{res}.npy' for year in years]
datasets, lkfs = get_lkfs(files)

# load tracks and paths (paths_all also includes the paths
# that are going through each timestep)
tracks = get_tracks(datasets)
paths, paths_all = get_paths(lkfs, tracks)

np.savez_compressed(path_stat + f'lkfs_paths_{res}.npz',
                    years=years, lkfs=lkfs, paths=paths, paths_all=paths_all)
