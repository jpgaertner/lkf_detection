import numpy as np
import xarray as xr
import os
import sys
sys.path.append('/home/a/a270230/LKF/lkf_tools/lkf_tools/')
from dataset import DoG_leads
from rgps import mSSMI
import matplotlib.pyplot as plt
import cartopy
import skimage.morphology

xticks = np.array([31,59,90,120,151,181,212,243,273,304,334,365])
xticks_minor = np.array([16,45,75,105,136,166,197,228,258,289,319,350])
xticks_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_strings = [
    'January', 'February', 'March', 'April', 'May', 'June', 
    'July', 'August', 'September', 'October', 'November', 'December'
]

# arcic basin 1km
lkf_data = np.load('/work/bk1377/a270230/datasets/1km/ds_2013_1km.npy', allow_pickle=True)[0]
a = max([0,lkf_data.index_y[0][0]-1])
b = lkf_data.index_y[0][-1]+2
c = max([0,lkf_data.index_x[0][0]-1])
d = lkf_data.index_x[0][-1]+2 - 70*4
e = lkf_data.red_fac
lon_1km = lkf_data.lon[a:b:e,c:d:e]
lat_1km = lkf_data.lat[a:b:e,c:d:e]

# arcic basin 4km
lkf_data = np.load('/work/bk1377/a270230/datasets/4km/ds_2013_4km.npy', allow_pickle=True)[0]
a = max([0,lkf_data.index_y[0][0]-1])
b = lkf_data.index_y[0][-1]+2
c = max([0,lkf_data.index_x[0][0]-1])
d = lkf_data.index_x[0][-1]+2 - 70
e = lkf_data.red_fac
lon_4km = lkf_data.lon[a:b:e,c:d:e]
lat_4km = lkf_data.lat[a:b:e,c:d:e]

def get_lkfs(files):
    '''files is a list containing the paths to the lkf_data files
    (see in detect_lkfs: np.save(path_ds + f'ds_{year}', [lkf_data])).
    
    datasets is a list containing the loaded lkf_data objects.
    lkfs is a list containing the lkf data for each dataset and day.
    len(lkfs) = len(datasets), len(lkfs[i]) = ntimesteps.
    
    the lkfs at a certain day are stored as (p_len,7) shaped array with p_len
    being their pixel length. it has 7 attributes:
    pixel coordinate x, pixel coordinate y, longitude, latitude,
    divergence, shear, and vorticity
    '''
    
    datasets, lkfs = [], []
    
    for file in files:
        lkf_data = np.load(file, allow_pickle=True)[0]
        datasets += lkf_data,

        lkfs_y = np.zeros_like(lkf_data.indexes[:365], dtype='object')
        for i, it in enumerate(lkf_data.indexes[:365]):
            lkfs_path = lkf_data.lkfpath.joinpath('lkf_%s_%03i.npy'
                                                  % (str(lkf_data.lkfpath).split('/')[-1],
                                                     (it+1))
                                                 )
            lkfs_y[i] = np.load(lkfs_path, allow_pickle=True)
        
        lkfs += lkfs_y,

    return datasets, lkfs

def get_tracks(datasets):
    '''returns a list with the tracks for each dataset and day.
    len(tracks) = len(datasets), len(tracks[i]) = ntimesteps.
    the tracks at day n are matching pairs, i.e. the number of an LKF in day n
    and the number of the associated LKF in day n+1.
    '''
    
    tracks = []
    for lkf_data in datasets:
    
        tracks_y = []
        tracks_list = os.listdir(lkf_data.track_output_path)
        tracks_list.sort()
        for it in range(len(lkf_data.indexes)-1):
            tracks_path = lkf_data.track_output_path.joinpath('lkf_tracked_pairs_%s_to_%s.npy'
                                                              % (lkf_data.lkf_filelist[it][4:-4],
                                                                 lkf_data.lkf_filelist[it+1][4:-4])
                                                             )
            if str(tracks_path).split('/')[-1] in tracks_list:
                tracks_y += np.load(tracks_path, allow_pickle=True),
                if np.size(tracks_y[-1]) == 0:
                    tracks_y[-1] = np.array([[np.nan, np.nan]])
            else:
                tracks_y += np.array([[np.nan, np.nan]]),
        
        tracks_y += np.array([[np.nan, np.nan]]), # to make it of length 365
        tracks += tracks_y,
    
    return tracks

def get_paths(lkfs, tracks):
    '''returns a list with the paths for each datasets and day.
    tracks[i][n] are the lkf paths of datasets[i] that start at day n. it contains
    the index of the lkfs at every day they are tracked to.
    paths_all[i][n] also contains the paths that go through day n.

    e.g.
    paths[i][0][0] = [0]           -> lkf 0 at day 1 is not tracked to day 2
    paths[i][0][4] = [4, 2, 5]     -> lkf 4 is tracked until day 3. in the
                                        second record, it has the index 2, in
                                        the third record the index 5
    
    todo: in this configuration, only the first path is saved if the lkf has
    two associated feature in the following record.
    '''
    
    paths, paths_all = [], []
    
    for lkfs_y, tracks_y in zip(lkfs, tracks):
    
        paths_y = np.zeros_like(lkfs_y,dtype='object')
        # number all the lkfs on day one
        for startday in range(len(lkfs_y)-1):
            paths_y[startday] = np.arange(np.shape(lkfs_y[startday])[0],
                                             dtype='object')

            # get the lkf number on day two (if the lkf can be tracked)
            # and add it to the lkf
            for ind, item in enumerate(paths_y[startday]):
                in_next_record = (item == tracks_y[startday][:,0])

                if np.any(in_next_record):
                    pos_in_next_record = np.where(in_next_record)
                    lkf_number_in_next_record = tracks_y[startday][pos_in_next_record,1]

                    paths_y[startday][ind] = np.append(item, lkf_number_in_next_record[0,0])

            # loop over the following days
            for i in range(1,len(tracks_y)-startday):
                for ind, item in enumerate(paths_y[startday]):
                    if np.array(item).size == i+1:
                        in_next_record = (item[-1] == tracks_y[i+startday][:,0])

                        if np.any(in_next_record):
                            pos_in_next_record = np.where(in_next_record)

                            lkf_number_in_next_record = tracks_y[i+startday][pos_in_next_record,1]

                            paths_y[startday][ind] = np.append(item, lkf_number_in_next_record[0,0])

        paths_all_y = paths_y.copy()

        # remove already tracked paths, i.e. delete a path from day n if
        # it starts at day n-1, so only the paths that start at day n remain
        already_tracked = np.zeros(len(paths_y)-1,dtype='object')
        for it in range(len(paths_y)-1):
            for ind, item in enumerate(paths_y[it]):
                if np.array(item).size > 1:
                    already_tracked[it] = [np.array(item).flat[1]
                                           if already_tracked[it]==0
                                           else np.append(already_tracked[it],np.array(item).flat[1])
                                          ]

        for i in range(1,len(tracks_y)):
            try:
                paths_y[i] = np.delete(paths_y[i], already_tracked[i-1])
            except:
                pass
            
            
        paths += paths_y,
        paths_all += paths_all_y,

    return paths, paths_all

def get_n_lkfs(lkfs):
    '''returns a list with the number of detected lkfs at every
    time step (day) in each dataset (year)
    '''
    
    n_lkfs = []
    for lkfs_year in lkfs:

        n_lkfs_year = [len(lkfs_day) for lkfs_day in lkfs_year]
        n_lkfs += n_lkfs_year,
        
    return n_lkfs

def get_res_km(lkf_data):
    '''returns the spatial resolution of a dataset in km
    '''

    return 0.0005 * (np.mean(lkf_data.dxu) + np.mean(lkf_data.dyu))

def get_lkf_length(lkfs, res_km):
    '''returns the length of every lkf as well as the mean and total lkf length,
    and the standart deviation at every time step in each dataset
    '''
    
    # initialize lists containing the data of all datasets
    length, av_length, sd_length, total_length = [], [], [], []

    # loop over datasets/ years
    for lkfs_year in lkfs:

        # initialize lists containing the data of the current dataset/ year
        length_year, av_length_year, sd_length_year, total_length_year = [], [], [], []

        # loop over time steps/ days
        for lkfs_day in lkfs_year:

            # calculate the step size between lkf pixels:
            # lkf[:,0] and lkf[:,1] contain the pixel coordinates in x and y direction of the current lkf.
            # the distance between two consecutive lkf pixels is calculated from the difference in x and y
            # pixel coordinates combined with the size of the pixels in km (spatial resolution, res_km)
            steps = [
                np.sqrt(np.array(np.diff(lkf[:,0])**2 + np.diff(lkf[:,1])**2, dtype='int')) * res_km
                for lkf in lkfs_day # loop over all lkfs in current time step
            ]

            # total lengths of the lkfs in current time step
            length_day = [np.sum(steps_lkf) for steps_lkf in steps]

            # append the data of one day as single element to the list of the whole year
            length_year += length_day,
            av_length_year += np.mean(length_day),
            sd_length_year += np.std(length_day),
            total_length_year += np.sum(length_day),

        # append the data of one year as single elements to the final list
        length += length_year,
        av_length += av_length_year,
        sd_length += sd_length_year,
        total_length += total_length_year,
    
    return length, av_length, sd_length, total_length

def get_lkf_lifetimes(paths):
    '''returns the lifetime of every lkf as well as the mean lifetime and
    the standart deviation at every time step in each dataset
    '''
    
    # initialize lists containing the data of all datasets
    lifetimes, av_lifetime, sd_lifetime = [], [], []

    # loop over datasets/ years
    for paths_year in paths:

        # initialize lists containing the data of the current dataset/ year
        lifetimes_year, av_lifetime_year, sd_lifetime_year = [], [], []

        # loop over all time steps/ days except the last one
        # (nothing is tracked there, the array just contains a 0)
        for paths_day in paths_year[:-1]:

            # loop over all lkfs in current time step and get their lifetime
            # (= length of their path)
            lifetimes_day = [np.size(path_lkfs) for path_lkfs in paths_day]

            # append the data of one day as single element to the list of the whole year
            lifetimes_year += lifetimes_day,
            av_lifetime_year += np.mean(lifetimes_day),
            sd_lifetime_year += np.std(lifetimes_day),

        # append nan day to make it of length of one year
        lifetimes_year += np.nan,
        av_lifetime_year += np.nan,
        sd_lifetime_year += np.nan,

        # append the data of one year as single elements to the final list
        lifetimes += lifetimes_year,
        av_lifetime += av_lifetime_year,
        sd_lifetime += sd_lifetime_year,

    return lifetimes, av_lifetime, sd_lifetime

def interannual_mean(data, startyear, endyear, years):
    '''get the interannual mean with its standart deviation for
    the time period from startyear til endyear
    '''
    # indices of startyear and endyear
    istart = np.where(np.array(years) == startyear)[0][0]
    iend = np.where(np.array(years) == endyear)[0][0]
    
    av = np.mean(data[istart:iend], axis=0)
    sd = np.std(data[istart:iend], axis=0)

    return av, sd

def interannual_mean_weighted(data, std, startyear, endyear, years):
    '''get the weighted interannual mean with its standart deviation
    for the time period from startyear til endyear for data that has
    uncertainty std
    '''
    # indices of startyear and endyear
    istart = np.where(np.array(years) == startyear)[0][0]
    iend = np.where(np.array(years) == endyear)[0][0]
    
    # the weights are the inverse variances
    values = np.array(data[istart:iend])
    weights = 1 / np.array(std[istart:iend])**2
    
    weighted_mean = np.sum(values * weights, axis=0) / np.sum(weights, axis=0)
    sd = np.sqrt(1 / np.sum(weights, axis=0))

    return weighted_mean, sd

def coarse_graining(field, res_km, coarse_grid_box_len_km):
    ''' Apply a coarse-graining filter to a 2D field. The function fills the coarse grid
    box with the mean value of the contained pixels. `res_km` is the spatial resolution 
    of the input field, `coarse_grid_box_len_km` is the length of one side of the coarse grid cell.
    '''
    if np.isnan(coarse_grid_box_len_km):
        # don't do coarse graining
        return field
    else:
        # coarse graining factor, how many original grid points make up on coarse resolution grid cell
        coarse_fac = int(coarse_grid_box_len_km / res_km)

        # shape of the coarse grained field 
        new_shape = (field.shape[0]//coarse_fac, field.shape[1]//coarse_fac)

        # cut off rows/ columns so that the field is divisible by the coarse graining factor
        field = field[:new_shape[0]*coarse_fac, :new_shape[1]*coarse_fac]

        # reshape field into 4D:
        # number of coarse rows, size of each block along rows, number of coarse columns, size of each block along columns
        # this splits the field into blocks of shape (coarse_fac, coarse_fac)
        field = field.reshape(new_shape[0], coarse_fac, new_shape[1], coarse_fac)

        # compute the average within each coarse_fac x coarse_fac block
        coarse_field = np.nanmean(field, axis=(1, 3))

        return coarse_field

def get_lkf_pixels(path_to_ncfile, i, dog_thres=0.01, aice_thresh=0, min_kernel=1, max_kernel=5, red_fac=1, skeleton_kernel=0, use_eps=True, plot=True, vmax=[0.4,0.5]):
    '''
    parameters to adjust (i is the timestep):
    dog_thres   : threshold in the DoG filtered image for a feature to be marked as LKF
                (default = 0.01 units of deformation)
    aice_thresh : threshold for the ice concentration below which the pixel is not marked
                as lkf pixel are not used (default = 0)
    min_kernel  : smallest scale of features to be detected (default = 1 pixel)
    max_kernel  : largest scale of features to be detected (default = 5 pixel)
                (with this, the background deformation is calculated:
                DoG filter = blurred image using min_kernel - blurred image using max_kernel)
    red_fac     : spatial filtering of the deformation rate field. only every red_fac pixel
                is usd for the lkf detection (default = 1)
    use_eps     : flag for using the total deformation (if True, default)
                or its natural logarithm and a histogram equalization (if False)
                (the latter highlights local differences across scales and thus enhances
                contrast in regions of low deformation)

    lkf_thin (return value): a 2D map being 1 at lkf pixels and 0 everywhere else
    '''

    # read file and store variables
    data = xr.open_dataset(path_to_ncfile)
    time = data.time
    lon = data.ULON
    lat = data.ULAT
    lon = lon.where(lon<=1e30); lat = lat.where(lat<=1e30);
    lon = lon.where(lon<180,other=lon-360)
    uice = np.array(data.U[i,:,:])
    vice = np.array(data.V[i,:,:])
    aice = np.array(data.A[i,:,:])

    # calculate grid cell sizes
    m = mSSMI()
    x,y = m(lon,lat)
    dxu = np.sqrt((x[:,1:]-x[:,:-1])**2 + (y[:,1:]-y[:,:-1])**2)
    dxu = np.concatenate([dxu,dxu[:,-1].reshape((dxu.shape[0],1))],axis=1)
    dyu = np.sqrt((x[1:,:]-x[:-1,:])**2 + (y[1:,:]-y[:-1,:])**2)
    dyu = np.concatenate([dyu,dyu[-1,:].reshape((1,dyu.shape[1]))],axis=0)
    
    # calculate strain rates, given in 1/day
    dudx = ((uice[2:,:]-uice[:-2,:])/(dxu[:-2,:]+dxu[1:-1,:]))[:,1:-1]
    dvdx = ((vice[2:,:]-vice[:-2,:])/(dxu[:-2,:]+dxu[1:-1,:]))[:,1:-1]
    dudy = ((uice[:,2:]-uice[:,:-2])/(dyu[:,:-2]+dyu[:,1:-1]))[1:-1,:]
    dvdy = ((vice[:,2:]-vice[:,:-2])/(dyu[:,:-2]+dyu[:,1:-1]))[1:-1,:]

    # calculate divergence, shear, vorticity, and total deformation
    div = (dudx + dvdy) * 3600. *24.
    shr = np.sqrt((dudx-dvdy)**2 + (dudy + dvdx)**2) * 3600. *24.
    vor = 0.5*(dudy-dvdx) * 3600. *24.
    eps_tot = np.sqrt(div**2+shr**2)
    eps_tot = np.where((aice[1:-1,1:-1]>0) & (aice[1:-1,1:-1]<=1), eps_tot, np.nan)
    
    # generate arctic basin mask
    mask = ((((lon > -120) & (lon < 100)) & (lat >= 80)) |
            ((lon <= -120) & (lat >= 70)) |
            ((lon >= 100) & (lat >= 70)))
    index_x = np.where(np.sum(mask[1:-1,1:-1],axis=0)>0)
    index_y = np.where(np.sum(mask[1:-1,1:-1],axis=1)>0)
    
    # apply mask and shrink arrays
    eps_tot = np.where(mask[1:-1,1:-1], eps_tot, np.nan)
    eps_tot = eps_tot[max([0,index_y[0][0]-1]):index_y[0][-1]+2,
                      max([0,index_x[0][0]-1]):index_x[0][-1]+2]
    eps_tot[0,:] = np.nan; eps_tot[-1,:] = np.nan
    eps_tot[:,0] = np.nan; eps_tot[:,-1] = np.nan
    eps_tot[1,:] = np.nan; eps_tot[-2,:] = np.nan
    eps_tot[:,1] = np.nan; eps_tot[:,-2] = np.nan
    aice = aice[1:-1,1:-1]
    aice = aice[max([0,index_y[0][0]-1]):index_y[0][-1]+2,
                max([0,index_x[0][0]-1]):index_x[0][-1]+2]

    # calculate spatial scaling correction factor and scale the detection parameters
    corfac = 12.5e3/np.mean([np.nanmean(dxu),np.nanmean(dyu)])/float(red_fac)
    max_kernel = max_kernel*(1+corfac)*0.5
    min_kernel = min_kernel*(1+corfac)*0.5
    
    # use_eps: use the deformation rate field for lkf detection
    # not use_eps: apply logarithm and histogram equalization to the
    # deformation rate before the lkf detection
    if use_eps:
        proc_eps = eps_tot
    else:
        proc_eps = np.log(eps_tot)
    proc_eps[~np.isfinite(proc_eps)] = np.NaN
    if not use_eps:
        proc_eps = hist_eq(proc_eps)

    # apply dog filter
    lkf_detect = DoG_leads(proc_eps,max_kernel,min_kernel)

    if plot:
        fig = plt.figure(figsize=(10,8))
        axs = [fig.add_subplot(2, 2, n, projection=cartopy.crs.Orthographic(0, 90)) for n in range(1,5)]

        for ax, data, title, vmax_ in zip(
            axs[:3], [eps_tot, lkf_detect], ['total deformation', 'difference of gaussian filter (DoG)'], vmax
        ):
            ax.pcolormesh(data,vmin=0, vmax=vmax_)
            ax.set_title(title, fontsize=16)

    # apply threshold, i.e. mark as lkf pixels if value after dog filter > dog_thres
    lkf_detect = (lkf_detect > dog_thres).astype('float')
    lkf_detect[~np.isfinite(proc_eps)] = np.NaN
    lkf_detect = (lkf_detect > 0)

    # calculate average total deformation
    eps_tot = np.nanmean(np.stack(eps_tot),axis=0)

    # apply morphological thinning
    if skeleton_kernel==0:
        lkf_thin =  skimage.morphology.skeletonize(lkf_detect).astype('float')
    else:
        lkf_thin = skeleton_along_max(eps_tot,lkf_detect,kernelsize=skeleton_kernel).astype('float')
        lkf_thin[:2,:] = 0.; lkf_thin[-2:,:] = 0.
        lkf_thin[:,:2] = 0.; lkf_thin[:,-2:] = 0.

    # this averages the ice concentration over 2 grid cells in every direction
    # (originally implemented for np.where(aice_mean>aice_thresh, ...)
    # it is not used because it does not change the result and
    # slows down the calculation way too much)
    #aice_mean = aice.copy()
    #aice_mean[2:-2,2:-2] = [[np.nanmean(aice[i-2:i+3,j-2:j+3])
    #                        for j in range(2,np.shape(aice)[1]-2)]
    #                       for i in range(2,np.shape(aice)[0]-2)
    #                      ]

    lkf_thin = np.where(aice>aice_thresh,lkf_thin,0)    

    if plot:
        for ax, data, title in zip(
            axs[2:], [lkf_detect, lkf_thin], ['threshold applied to DoG', 'morphological thinning']
        ):
            ax.pcolormesh(data,vmin=0, vmax=1, cmap='Greys')
            ax.set_title(title, fontsize=16)

        fig.tight_layout()
        plt.show()

    return lkf_thin