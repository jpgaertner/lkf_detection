import sys
sys.path.append('../functions/')
from statistics_functions import *
sys.path.append('../../lkf_tools/lkf_tools/')
from dataset import *
import numpy as np
import warnings
warnings.filterwarnings('ignore')


res = '4km'

path = '/work/bk1377/a270230/'
path_nc   = path + f'interpolated_fesom_output/{res}/'
path_stat = path + 'statistics/'

# select the years you want to analyze
years = [i for i in range(2013,2021)]
years += [i for i in range(2093,2101)]

# length of the grid box used in the coarse graining filter
coarse_grid_box_len_km = 25

# use already calculated resolutions (can be calculated either from the nc files like in
# plot/area_thickness.ipynb, or from the lkf_data objects like in statistics_main.ipynb)
if res == '4km': res_km = 4.337849218906646
if res == '1km': res_km = 1.083648783567869

lkf_map = []
for year in years:
    lkf_map_y = []
    for d in range(365):
        # first retrieve all pixels marked as lkf pixels at each time step with the function get_lkf_pixels,
        # then apply the coarse graining filter. the lkf map is one in the grid boxes containing an lkf at that
        # time step and zero everywhere ekse
        lkf_map_y += coarse_graining(get_lkf_pixels(path_nc + f'{year}_{res}.nc', i=d, dog_thres=0.01, plot=False),
                                    coarse_grid_box_len_km, res_km),
    lkf_map += lkf_map_y,
    
days = np.append(0, xticks)

# calculate the monthly lkf frequency. it is one in a grid
# box for an lkf being present at every day of the month
lkf_map_monthly = []
for lkf_map_y in lkf_map:
    
    lkf_map_monthly_y = []
    # collect the data for each month
    for startday, endday in zip(days, np.roll(days,-1)):
        
        tmp = []
        for i in range(startday, endday):
            
            # sum up the lkf maps for each day in the current month
            # and divide by the number of days
            tmp += lkf_map_y[i],
        lkf_map_monthly_y += np.sum(tmp,axis=0)/(endday-startday),

    # this is to delete the day=365 to day=0 element
    del lkf_map_monthly_y[-1]
    lkf_map_monthly += lkf_map_monthly_y,
    
np.save(path_stat + f'lkf_map_monthly_{res}', lkf_map_monthly)
