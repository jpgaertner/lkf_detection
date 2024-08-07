import sys
sys.path.append('../functions/')
from statistics_functions import *
import numpy as np


res = '1km'

path = '/work/bk1377/a270230/'
path_stat = path + 'statistics/'

# load the lkf data
data = np.load(path_stat + f'lkfs_paths_{res}.npz', allow_pickle=True)
years, lkfs, paths, paths_all = [data[key] for key in data.files]

LKFs = np.load(path_stat + f'LKFs_{res}.npy', allow_pickle=True)[0]

# calculate the mean of deformation, divergence, shear, and vorticity 
# of each lkf and add it to the dictionary
for year in years:
    
    y = np.where(years==year)[0][0]
    for d in range(365):

        eps_d, div_d, shr_d, vor_d = [], [], [], []
        for lkf in lkfs[y][d]:

            div_d += np.mean(lkf[:,4]),
            shr_d += np.mean(lkf[:,5]),
            vor_d += np.mean(lkf[:,6]),

        eps_d = np.sqrt(np.array(div_d)**2+np.array(shr_d)**2)

        LKFs[f'{year} daily'][f'{d+1}']['deformation'] = eps_d
        LKFs[f'{year} daily'][f'{d+1}']['divergence']  = div_d
        LKFs[f'{year} daily'][f'{d+1}']['shear']       = shr_d
        LKFs[f'{year} daily'][f'{d+1}']['vorticity']   = vor_d

# determine whether an lkf is a lead (1), a ridge (2) or not quantifiable (=longer lived) (3).
# "divergence>0 => lead" only works for the first day of the lkf. on the second day, a lead
# can have divergence<0 but is still a region of open water
for year in years:
    LKFs[f'{year} daily']['1']['lead or ridge'] = np.where(LKFs[f'{year} daily']['1']['divergence']>0, 1, 2)
    
    y = np.where(years==year)[0][0]
    for d in range(1,365):
        
        LKFs[f'{year} daily'][f'{d+1}']['lead or ridge'] = np.where(LKFs[f'{year} daily'][f'{d+1}']['divergence']>0, 1, 2)
        for lkf in range(len(lkfs[y][d])):
            
            for lkf_path in paths_all[0][0]:
                if np.size(lkf_path)>1:
                    if lkf_path[1] == lkf:
                        LKFs[f'{year} daily'][f'{d+1}'].loc[lkf, 'lead or ridge'] = 3
                        break

np.save(path_stat + f'LKFs_{res}.npy', [LKFs])
