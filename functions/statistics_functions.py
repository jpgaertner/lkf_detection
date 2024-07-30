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
        print(lkf_data.netcdf_file)
    
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
    time step in each dataset
    '''
    
    n_lkfs = []
    for lkfs_y in lkfs:
        
        n_lkfs_y = []
        for lkfs_d in lkfs_y:
            n_lkfs_y += len(lkfs_d),
            
        n_lkfs += n_lkfs_y,
        
    return n_lkfs

def get_res_km(lkf_data):
    '''returns the spatial resolution of a dataset in km
    '''

    return 0.0005 * (np.mean(lkf_data.dxu) + np.mean(lkf_data.dyu))

def get_lkf_length(lkfs, res_km):
    '''returns the length of every LKF as well as the mean and
    total LKF length at every time step in each dataset
    '''
    
    length, av_length, total_length = [[] for _ in range(3)]
    for yr in range(len(lkfs)):

        length_y, av_length_y, total_length_y = [[] for _ in range(3)]
        for day in range(len(lkfs[0])):

            length_d = []
            for lkf in lkfs[yr][day]:

                steps = np.sqrt(
                      np.diff(np.array(lkf[:,0], dtype='int'))**2 \
                    + np.diff(np.array(lkf[:,1], dtype='int'))**2) * res_km # step sizes between LKF pixels
                length_d += np.sum(steps),

            length_y       += length_d,
            av_length_y    += np.mean(length_d),
            total_length_y += np.sum(length_d),

        length       += length_y,       # lenth of every LKF
        av_length    += av_length_y,    # mean length of LKFs
        total_length += total_length_y, # total length of LKFs
    
    return length, av_length, total_length

def get_lkf_lifetimes(paths):
    '''returns the lifetimes of each tracked path that starts at the
    respective day, as well as the total mean lifetime of all tracked paths.
    '''
    
    lifetimes, mean_lifetime = [], []
    
    for paths_y in paths:
    
        # lifetime_y[i] contains the lifetimes of the paths that start at day i of year y
        lifetimes_y = np.empty_like(paths_y, dtype='object')
        lifetimes_y[-1] = np.empty_like(paths_y[-1], dtype='object')
        lifetimes_y[:] = np.nan

        for i in range(len(paths_y)-1):
            lifetimes_y[i] = np.zeros_like(paths_y[i], dtype='object')

            for ind, item in enumerate(paths_y[i]):
                lifetimes_y[i][ind] = np.array(item).size

        # mean lifetime of all tracked paths
        mean_lifetime_y = []
        for lifetimes_timestep in lifetimes_y[:-1]:
            mean_lifetime_y += lifetimes_timestep.mean(),

        # make it the same length as lifetime
        mean_lifetime_y += np.nan,
    
        lifetimes += lifetimes_y,
        mean_lifetime += mean_lifetime_y,

    return lifetimes, mean_lifetime

def av_sd(data, startyear, endyear, years):
    '''get the interannual mean with its standart deviation for
    the time from startyear til endyear
    '''
    istart = np.where(np.array(years) == startyear)[0][0]
    iend = np.where(np.array(years) == endyear)[0][0]
    
    av = np.mean(data[istart:iend], axis=0)
    sd = np.sqrt(np.var(data[istart:iend], axis=0))

    return av, sd

def coarse_graining(field, coarse_grid_box_size_km, res_km):
    '''apply a coarse graining filter to field. res_km is the spatial
    resolution of the input field, coarse_grid_box_size_km is the length
    of one side of the coarse grid cell.
    '''

    n_rows = round(np.shape(field)[0] * res_km / coarse_grid_box_size_km)
    n_cols = round(np.shape(field)[1] * res_km / coarse_grid_box_size_km)

    rows = np.array_split(field, n_rows, axis=0)
    columns = []
    for row in rows:
        columns.append(np.array_split(row, n_cols, axis=1))
    
    for i in range(n_rows):
        for j in range(n_cols):
            if np.any(columns[i][j]==1):
                columns[i][j] = np.ones_like(columns[i][j])
                
    assemble = []
    for row in columns:
        assemble.append(np.concatenate(row, axis=1))
        
    field = np.concatenate(assemble, axis=0)
    
    return field

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