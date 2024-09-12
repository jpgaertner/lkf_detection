import numpy as np
import sys
sys.path.insert(1, '../functions/')
from statistics_functions import *
import warnings
warnings.filterwarnings('ignore')


res = '4km'

path = '/work/bk1377/a270230/'
path_ds   = path + f'datasets/{res}/'
path_stat = path + 'statistics/'

# if this is set to np.nan, no coarse graining is applied
coarse_grid_box_len_km = np.nan

LKFs = np.load(path_stat + f'LKFs_{res}_all.npy', allow_pickle=True)[0]
years = [key for key in LKFs.keys() if len(key)==4]

if res == '4km':
    res_km = 4.337849218906646
    lon = lon_4km
elif res == '1km':
    res_km = 1.083648783567869
    z = 70*4
    lon = lon_1km

# start and end day of the months
startdays = np.append(0,xticks)[:-1]+1
enddays = np.append(xticks[:-1], 365)+1 # this takes into account the exclusive indexing of python in [start:end]

months = [str(i) for i in range(1,13)]

def nan_to_0(field):
    return np.where(np.isnan(field),0,field)

# get the coarse grained maps of the lkf frequency and
# of the monthly averages of lifetime, length, and curvature

maps = dict()

for year in years:

    maps[year] = dict()
    lkf_data = np.load(path_ds + f'ds_{year}_{res}.npy', allow_pickle=True)[0] # used for the path to the lkf files

    # loop over all months, save monthly frequencies and monthly averages in the dictionary
    for startday, endday, month in zip(startdays, enddays, months):

        maps[year][month] = dict()

        # these contain all data of the current month
        [coarse_lkf_map, coarse_lead_map, coarse_ridge_map, coarse_nq_map,
        coarse_lifetime_map, coarse_length_map, coarse_curv_map] = [[] for _ in range(7)]

        days = np.arange(startday, endday)
        for day in days:
            # these contain the maps of the current day
            [lkf_map, lead_map, ridge_map, nq_map, lifetime_map, length_map, curv_map
            ] = [np.full_like(lon, np.nan) for _ in range(7)]

            # save lkf dataframe and load lkf data of the current day
            daily_lkfs = LKFs[f'{year} daily'][f'{day}']
            lkfs = np.load(lkf_data.lkfpath.joinpath('lkf_%s_%03i.npy' %(lkf_data.netcdf_file.split('/')[-1].split('.')[0],(day))),allow_pickle=True)

            for i_lkf, lkf in enumerate(lkfs):
                # get lkf pixel coordinates
                lkf_y = lkf[:,0].astype(int)
                lkf_x = lkf[:,1].astype(int)

                # mark all the lkf pixels
                lkf_map[lkf_y, lkf_x] = 1

                # assign the lkf type
                if daily_lkfs['lead or ridge'][i_lkf]==1: # the lkf is a lead
                    lead_map[lkf_y, lkf_x] = 1
                elif daily_lkfs['lead or ridge'][i_lkf]==2: # the lkf is a ridge
                    ridge_map[lkf_y, lkf_x] = 1
                elif daily_lkfs['lead or ridge'][i_lkf]==3: # the lkf is not quantifiable
                    nq_map[lkf_y, lkf_x] = 1

                # assign the rest of the variables to the lkf pixels
                lifetime_map[lkf_y, lkf_x] = daily_lkfs['lifetime'][i_lkf]
                length_map[lkf_y, lkf_x] = daily_lkfs['length'][i_lkf]
                curv_map[lkf_y, lkf_x] = daily_lkfs['curvature'][i_lkf]

            # apply coarse graining and add the daily maps to the month-lists
            coarse_lkf_map      += coarse_graining(lkf_map, res_km, coarse_grid_box_len_km),
            coarse_lead_map     += coarse_graining(lead_map, res_km, coarse_grid_box_len_km),
            coarse_ridge_map    += coarse_graining(ridge_map, res_km, coarse_grid_box_len_km),
            coarse_nq_map       += coarse_graining(nq_map, res_km, coarse_grid_box_len_km),
            coarse_lifetime_map += coarse_graining(lifetime_map, res_km, coarse_grid_box_len_km),
            coarse_length_map   += coarse_graining(length_map, res_km, coarse_grid_box_len_km),
            coarse_curv_map     += coarse_graining(curv_map, res_km, coarse_grid_box_len_km),

        # calculate the monthly values/ monthly means
        maps[year][month]['lkf frequency'] = np.nansum(coarse_lkf_map, axis=0) / len(days)
        maps[year][month]['lead frequency'] = np.nansum(coarse_lead_map, axis=0) / len(days)
        maps[year][month]['ridge frequency'] = np.nansum(coarse_ridge_map, axis=0) / len(days)
        maps[year][month]['nq frequency'] = np.nansum(coarse_nq_map, axis=0) / len(days)
        maps[year][month]['lifetime'] = np.nanmean(coarse_lifetime_map, axis=0)
        maps[year][month]['length'] = np.nanmean(coarse_length_map, axis=0)
        maps[year][month]['curvature'] = np.nanmean(coarse_curv_map, axis=0)

        # replace np.nan with 0
        maps[year][month]['lifetime'] = nan_to_0(maps[year][month]['lifetime'])
        maps[year][month]['length'] = nan_to_0(maps[year][month]['length'])
        maps[year][month]['curvature'] = nan_to_0(maps[year][month]['curvature'])

np.savez_compressed(path_stat + f'Maps_{res}_to_{coarse_grid_box_len_km}km.npz', maps=[maps])
