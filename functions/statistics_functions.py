import numpy as np
import os

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