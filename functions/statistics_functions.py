import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

def get_ice_coverage(years, path_ice):
    '''calculate the total ice area (in km2) at every time step
    '''

    ice_coverage = [[] for _ in range(len(years))]
    ice_coverage_weekly = [[] for _ in range(len(years))]
    for ind, year in enumerate(years):
        ice_file = path_ice + f'ice_{year}.nc'
        data = xr.open_dataset(ice_file)

        # use len(data.time) if you want to include leap years;
        # i want the years to have the same length
        for day in range(365): 
            ice_coverage[ind].append(np.nansum(data.A.isel(time=day)) * 4.45**2) 

        ice_coverage_weekly[ind] = ice_coverage[ind][::7]
        
    return ice_coverage, ice_coverage_weekly

def get_lkfs_all(lkf_data):
    '''returns a list with the lkf data for each day (len(lkfs_all) = ntimesteps).
    the lkfs at one day are stored as (p_len,7) shaped array with p_len
    being their pixel length. it has 7 attributes like longitude and latitude,
    stored in the [:,2] and [:,3] position, respectively.
    '''

    lkfs_all = []
    for it in lkf_data.indexes:
        lkfs_path = lkf_data.lkfpath.joinpath('lkf_%s_%03i.npy'
                                              % (lkf_data.netcdf_file.split('/')[-1].split('.')[0],
                                                 (it+1))
                                             )
        lkfs_all.append(np.load(lkfs_path, allow_pickle=True))

    return lkfs_all

def get_n_lkfs(lkfs_all):
    '''returns a list with the number of detected lkfs at each time step
    (len(get_n_lkfs) = ntimesteps).
    '''

    n_lkfs = []
    for lkfs_all_timestep in lkfs_all:
        n_lkfs.append(len(lkfs_all_timestep))

    return n_lkfs

def get_lkf_len(lkfs_all):
    '''get the length of every LKF as well as the average
    LKF length in pixels at every timestep (len(get_n_lkfs) = ntimesteps).
    '''
    
    lkf_len = [[] for _ in range(len(lkfs_all))]
    av_lkf_len, total_lkf_len = [], []
    for timestep, lkfs_all_timestep in enumerate(lkfs_all):
        for lkf in lkfs_all_timestep:
            lkf_len[timestep].append(int(len(lkf)*4.44))
        
        av_lkf_len.append(np.nanmean(lkf_len[timestep]))
        total_lkf_len.append(np.nansum(lkf_len[timestep]))
    
    return lkf_len, av_lkf_len, total_lkf_len
    

def get_tracks_all(lkf_data):
    '''returns a list with the tracks for each day (len(tracks_all) = ntimesteps-1).
    the tracks at day i are matching pairs, i.e. the number of an LKF in record i
    and the number of an associated LKF in record i+1.
    '''
    tracks_all = []
    
    tracks_list = os.listdir(lkf_data.track_output_path)
    tracks_list.sort()
    
    for it in range(len(lkf_data.indexes)-1):
        tracks_path = lkf_data.track_output_path.joinpath('lkf_tracked_pairs_%s_to_%s.npy'
                                                          % (lkf_data.lkf_filelist[it][4:-4],
                                                             lkf_data.lkf_filelist[it+1][4:-4])
                                                         )
        if str(tracks_path).split('/')[-1] in tracks_list:
            tracks_all.append(np.load(tracks_path, allow_pickle=True))
            if np.size(tracks_all[-1]) == 0:
                tracks_all[-1] = np.array([[np.nan, np.nan]])
        else:
            tracks_all.append(np.array([[np.nan, np.nan]]))
    
    return tracks_all

def get_lkf_paths(lkfs_all, tracks_all):
    '''returns the array lkf_tracks:
    lkf_tracks[i] are the lkf paths that start at day i. it contains
    the index of the lkfs at every day they are tracked to.

    e.g.
    lkf_tracks[0][0] = [0]           -> lkf 0 at day 1 is not tracked to day 2
    lkf_tracks[0][4] = [4, 2, 5]     -> lkf 4 is tracked until day 3. in the
                                        second record, it has the index 2, in
                                        the third record the index 5

    todo: in this configuration, only the first path is saved if the lkf has
    two associated feature in the following record.
    '''
    
    lkf_paths = np.zeros(len(lkfs_all),dtype='object')

    # number all the lkfs on day one
    for startday in range(len(lkfs_all)-1):
        lkf_paths[startday] = np.arange(np.shape(lkfs_all[startday])[0],
                                         dtype='object')

        # get the lkf number on day two (if the lkf can be tracked)
        # and add it to the lkf
        for ind, item in enumerate(lkf_paths[startday]):
            in_next_record = (item == tracks_all[startday][:,0])

            if np.any(in_next_record):
                pos_in_next_record = np.where(in_next_record)
                lkf_number_in_next_record = tracks_all[startday][pos_in_next_record,1]

                lkf_paths[startday][ind] = np.append(item, lkf_number_in_next_record[0,0])

        # loop over the following days
        for i in range(1,len(tracks_all)-startday):
            for ind, item in enumerate(lkf_paths[startday]):
                if np.array(item).size == i+1:
                    in_next_record = (item[-1] == tracks_all[i+startday][:,0])

                    if np.any(in_next_record):
                        pos_in_next_record = np.where(in_next_record)

                        lkf_number_in_next_record = tracks_all[i+startday][pos_in_next_record,1]

                        lkf_paths[startday][ind] = np.append(item, lkf_number_in_next_record[0,0])

    # remove already tracked paths, i.e. delete a path from day n if
    # it starts at day n-1, so only the paths start start at day n remain
    already_tracked = np.zeros(len(lkf_paths)-1,dtype='object')
    for it in range(len(lkf_paths)-1):
        for ind, item in enumerate(lkf_paths[it]):
            if np.array(item).size > 1:
                already_tracked[it] = [np.array(item).flat[1]
                                       if already_tracked[it]==0
                                       else np.append(already_tracked[it],np.array(item).flat[1])
                                      ]

    for i in range(1,len(tracks_all)):
        try:
            lkf_paths[i] = np.delete(lkf_paths[i], already_tracked[i-1])
        except:
            pass

    return lkf_paths

def get_lifetimes(lkf_paths):
    '''returns the lifetimes of each tracked path that starts at the
    respective day, as well as the total mean lifetime of all tracked paths
    and the total lifetime of all tracked paths that are longer than one day.
    '''
    
    # lifetimes[i] contains the lifetimes of the paths that start at day i
    lifetimes = np.zeros_like(lkf_paths, dtype='object')
    
    for i in range(len(lkf_paths)-1):
        lifetimes[i] = np.zeros_like(lkf_paths[i], dtype='object')
        
        for ind, item in enumerate(lkf_paths[i]):
            lifetimes[i][ind] = np.array(item).size
    
    # mean lifetime of all tracked paths
    mean_lifetime = []
    for lifetimes_timestep in lifetimes[:-1]:
        mean_lifetime.append(lifetimes_timestep.mean())
    

    # mean lifetime of all tracked paths longer than one day
    lifetimes_tracked_lkfs = [lifetimes[:-1][i][np.where(lifetimes[:-1][i]!=1)]
                              for i in range(len(lifetimes[:-1]))
                             ]

    mean_lifetime_tracked_lkfs = []
    for lifetimes_tracked_lkfs_timestep in lifetimes_tracked_lkfs:
        mean_lifetime_tracked_lkfs.append(lifetimes_tracked_lkfs_timestep.mean())

    return lifetimes, mean_lifetime, mean_lifetime_tracked_lkfs
    
def coarse_graining(field, coarse_grid_box_size_km):

    res_km = 4.44 # calculated in assemble_dataset
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