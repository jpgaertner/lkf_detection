import sys
sys.path.append('../functions/')
from statistics_functions import *
import numpy as np


res = '4km'

path = '/work/bk1377/a270230/'
path_stat = path + 'statistics/'

# load the lkf data
data = np.load(path_stat + f'lkfs_paths_{res}_all.npz', allow_pickle=True)
years, lkfs, paths, paths_all = [data[key] for key in data.files]

LKFs = np.load(path_stat + f'LKFs_{res}.npy', allow_pickle=True)[0]

# calculate the mean deformation, divergence, shear,
# and vorticity of every individual lkf
for i_year, year in enumerate(years):
    for i_day in range(365):

        # initialize lists that contain the mean values for each lkf
        eps_d, div_d, shr_d, vor_d = [], [], [], []
        for lkf in lkfs[i_year][i_day]:
            # loop over lkfs and take the mean over all lkf pixels
            div_d += np.mean(lkf[:,4]),
            shr_d += np.mean(lkf[:,5]),
            vor_d += np.mean(lkf[:,6]),

        # calculate total deformation
        eps_d = np.sqrt(np.array(div_d)**2+np.array(shr_d)**2)

        LKFs[f'{year} daily'][f'{i_day+1}']['deformation'] = eps_d
        LKFs[f'{year} daily'][f'{i_day+1}']['divergence']  = div_d
        LKFs[f'{year} daily'][f'{i_day+1}']['shear']       = shr_d
        LKFs[f'{year} daily'][f'{i_day+1}']['vorticity']   = vor_d

# determine whether an lkf is a lead (1), a ridge (2) or not quantifiable (=longer lived) (3).
# "divergence>0 => lead" only works for the first day of the lkf. on the second day, a lead
# can have divergence<0 but is still a region of open water
for i_year, year in enumerate(years):

    # on the first day, all lkfs are first day lkfs (duh)
    LKFs[f'{year} daily']['1']['lead or ridge'] = np.where(LKFs[f'{year} daily']['1']['divergence']>0, 1, 2)

    for day in range(2,366):
        # first do the lead/ ridge classification for all lkfs
        LKFs[f'{year} daily'][f'{day}']['lead or ridge'] = np.where(LKFs[f'{year} daily'][f'{day}']['divergence']>0, 1, 2)

        # then go through all the paths from the day before
        # (-1 to go to the day before, and another -1 to go from day to index)
        for path_individual_lkf in paths_all[i_year][day-2]:

            # this is true for all tracked lkfs
            if np.size(path_individual_lkf)>1:

                # the index of the lkf in the second day
                index_of_tracked_lkf = path_individual_lkf[1]

                # set it to 3 (in the daily lkf panda series, index and lkf label is the same)
                LKFs[f'{year} daily'][f'{day}'].loc[index_of_tracked_lkf, ('lead or ridge')] = int(3)

np.save(path_stat + f'LKFs_{res}.npy', [LKFs])
