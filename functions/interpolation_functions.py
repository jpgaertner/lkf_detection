import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import cartopy.crs as ccrs
import logging
from scipy.spatial import cKDTree


def proj_selection(projection):
    if projection == "pc":
        projection_ccrs = ccrs.PlateCarree()
    elif projection == "mer":
        projection_ccrs = ccrs.Mercator()
    elif projection == "np":
        projection_ccrs = ccrs.NorthPolarStereo()
    elif projection == "sp":
        projection_ccrs = ccrs.SouthPolarStereo()
    return projection_ccrs

def region_cartopy(box, res, projection="pc"):
    """ Computes coordinates for the region 
    Parameters
    ----------
    box : list
        List of left, right, bottom, top boundaries of the region in -180 180 degree format
    res: list
        List of two variables, defining number of points along x and y
    projection : str
        Options are:
            "pc" : cartopy PlateCarree
            "mer": cartopy Mercator
            "np" : cartopy NorthPolarStereo
            "sp" : cartopy SouthPolarStereo
    Returns
    -------
    x : numpy.array
        1 d array of coordinate values along x
    y : numpy.array
        1 d array of coordinate values along y
    lon : numpy.array
        2 d array of longitudes
    lat : numpy array
        2 d array of latitudes
    """
    projection_ccrs = proj_selection(projection)

    if not res is None:
        lonNumber, latNumber = res
    else:
        lonNumber, latNumber = 500, 500
    left, right, down, up = box
    logging.info('Box %s, %s, %s, %s', left, right, down, up)
    fig, ax = plt.subplots(
        1,
        1,
        subplot_kw=dict(projection=projection_ccrs),
        constrained_layout=True,
        figsize=(10, 10),
    )
    ax.set_extent([left, right, down, up], crs=ccrs.PlateCarree())
    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()

    # res = scl_fac * 300. # last number is the grid resolution in meters (NEEDS TO BE CHANGED)
    # nx = int((xmax-xmin)/res)+1; ny = int((ymax-ymin)/res)+1
    x = np.linspace(xmin, xmax, lonNumber)
    y = np.linspace(ymin, ymax, latNumber)
    x2d, y2d = np.meshgrid(x, y)

    npstere = ccrs.PlateCarree()
    transformed2 = npstere.transform_points(projection_ccrs, x2d, y2d)
    lon = transformed2[:, :, 0]  # .ravel()
    lat = transformed2[:, :, 1]  # .ravel()
    fig.clear()
    plt.close(fig)
   
    return x, y, lon, lat

def create_indexes_and_distances(model_lon, model_lat, lons, lats, k=1, workers=2):
    """
    Creates KDTree object and query it for indexes of points in FESOM mesh that are close to the
    points of the target grid. Also return distances of the original points to target points.
    Parameters
    ----------
    mesh : fesom_mesh object
        pyfesom mesh representation
    lons/lats : array
        2d arrays with target grid values.
    k : int
        k-th nearest neighbors to return.
    n_jobs : int, optional
        Number of jobs to schedule for parallel processing. If -1 is given
        all processors are used. Default: 1.
    Returns
    -------
    distances : array of floats
        The distances to the nearest neighbors.
    inds : ndarray of ints
        The locations of the neighbors in data.
    """
    xs, ys, zs = lon_lat_to_cartesian(model_lon, model_lat)
    xt, yt, zt = lon_lat_to_cartesian(lons.flatten(), lats.flatten())

    tree = cKDTree(list(zip(xs, ys, zs)))
    distances, inds = tree.query(list(zip(xt, yt, zt)), k=k, workers=workers)

    return distances, inds

def lon_lat_to_cartesian(lon, lat, R=6371000):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R. Taken from http://earthpy.org/interpolation_between_grids_with_ckdtree.html
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z

def interpolate(a_ice, u_ice, v_ice, inds, distances, ntimesteps, shape, r=5000):

    radius_of_influence = r
    
    a_ice_int = np.zeros((ntimesteps,*shape))
    u_ice_int = np.zeros_like(a_ice_int)
    v_ice_int = np.zeros_like(a_ice_int)

    for i in range(ntimesteps):
        a_ = a_ice[i].values[inds]
        u_ = u_ice[i].values[inds]
        v_ = v_ice[i].values[inds]

        a_[distances >= radius_of_influence] = np.nan
        u_[distances >= radius_of_influence] = np.nan
        v_[distances >= radius_of_influence] = np.nan

        a_.shape = shape
        u_.shape = shape
        v_.shape = shape
        
        a_ice_int[i] = a_
        u_ice_int[i] = u_
        v_ice_int[i] = v_

    return a_ice_int, u_ice_int, v_ice_int

def create_nc_file(a_ice_int, h_ice_int, u_ice_int, v_ice_int, int_lons, int_lats, ntimesteps, name):
    ds = nc.Dataset(name, 'w', format='NETCDF4')

    x = ds.createDimension('x', np.shape(a_ice_int)[2])
    y = ds.createDimension('y', np.shape(a_ice_int)[1])
    time = ds.createDimension('time', ntimesteps)

    x = ds.createVariable('x', 'f4', ('x',))
    y = ds.createVariable('y', 'f4', ('y',))
    time = ds.createVariable('time', 'f4', ('time'))

    a = ds.createVariable('A', 'f4', ('time','y','x'))
    h = ds.createVariable('H', 'f4', ('time','y','x'))
    u = ds.createVariable('U', 'f4', ('time','y','x'))
    v = ds.createVariable('V', 'f4', ('time','y','x'))
    lon = ds.createVariable('ULON', 'f4', ('y','x'))
    lat = ds.createVariable('ULAT', 'f4', ('y','x'))

    x[:] = np.arange(np.shape(a_ice_int)[2],dtype='int')
    y[:] = np.arange(np.shape(a_ice_int)[1],dtype='int')
    time[:] = np.arange(ntimesteps)
    a[:,:,:] = a_ice_int
    h[:,:,:] = h_ice_int
    u[:,:,:] = u_ice_int
    v[:,:,:] = v_ice_int
    lon[:,:] = int_lons
    lat[:,:] = int_lats

    ds.close()