flag_run = 1
# ================================================================
# Yu-Chiao @ WHOI Nov 18, 2018
# time series calculate
# ================================================================

# ================================================================
# Import functions
# ================================================================
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from math import isnan, radians
#from mpl_toolkits.basemap import Basemap
from IPython import get_ipython
import sys, os
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from pylab import setp, genfromtxt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from datetime import datetime
from scipy import stats
#import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.img_tiles as cimgt
from cartopy.io.img_tiles import StamenTerrain
import matplotlib.path as mpath
import xarray as xr

sys.path.append('/home/yliang/lib/python_tools/python_functions/whoi_projects/')
import whoi_data_process_f
sys.path.append('/home/yliang/lib/python_tools/python_functions/data_process/')
import data_process_f
import ERA_interim_data_process_f
sys.path.append('/home/yliang/lib/python_tools/python_functions/statistics')
import statistical_f, MCA_f, EOF_f
sys.path.append('/home/yliang/lib/python_tools/python_functions')
import plot_functions

def plot_box(ax,lon1,lon2,lat1,lat2, color_txt):

    ax.plot(np.linspace(lon1,lon1,100), np.linspace(lat1, lat2, 100), transform=ccrs.PlateCarree(), color=color_txt, linewidth=0.6)
    ax.plot(np.linspace(lon2,lon2,100), np.linspace(lat1, lat2, 100), transform=ccrs.PlateCarree(), color=color_txt, linewidth=0.6)
    ax.plot(np.linspace(lon1,lon2,100), np.linspace(lat1, lat1, 100), transform=ccrs.PlateCarree(), color=color_txt, linewidth=0.6)
    ax.plot(np.linspace(lon1,lon2,100), np.linspace(lat2, lat2, 100), transform=ccrs.PlateCarree(), color=color_txt, linewidth=0.6)

def perform_ttest_here(exp1_var,exp2_var,ny,nx,sig_level):
    ttest_map = np.zeros((ny,nx))*np.nan
    pvalue_map = np.zeros((ny,nx))*np.nan
    for JJ in range(ny):
        for II in range(nx):
            [xxx, pvalue] = stats.ttest_ind(exp1_var[:,JJ,II],exp2_var[:,JJ,II])
            if pvalue < sig_level:
               ttest_map[JJ,II] = 1.
               pvalue_map[JJ,II] = pvalue

    return ttest_map, pvalue_map

plt.close('all')

if flag_run == 1:
# ================================================================
# Read simulation results
# ================================================================
# read greid and mask information
   dirname = '/vortex/jetstream/Arctic_midlatitude_waccm_whoi_ncar/PAMIP/'
   filename = 'waccm001_future_bw01_panp04_monthly0_atm.nc'
   f = Dataset(dirname + filename, 'r')
   lat = f.variables['lat'][:].data
   lon = f.variables['lon'][:].data
   time = f.variables['time'][:].data
   f.close()

   dirname = '/vortex/jetstream/Arctic_midlatitude_waccm_whoi_ncar/PAMIP/raw_output/future/waccm001_future_b09_panp90/atm/hist/'
   filename = 'waccm001_future_b09_panp90.cam.h1.2000-04-01-00000.nc'
   f = Dataset(dirname + filename, 'r')
   lev = f.variables['lev'][:].data
   P0 = f.variables['P0'][:].data*0.01
   hyam = f.variables['hyam'][:].data
   hybm = f.variables['hybm'][:].data
   f.close()

   nz = len(lev)
   ny = len(lat)
   nx = len(lon)
   nt = len(time)

   case_name = 'preindustry'
   var_name = 'U'

# Read future case
   arr = os.listdir('/vortex/jetstream/Arctic_midlatitude_waccm_whoi_ncar/PAMIP/' + case_name + '/')
   n_case = len(arr)

   var_out = np.zeros((n_case,nt,nz,ny,nx))
   ps_out = np.zeros((n_case,nt,ny,nx))
   var_out1 = np.zeros((int(n_case/2),nt,nz,ny,nx)) 
   var_out2 = np.zeros((int(n_case/2),nt,nz,ny,nx))
   ps_out1 = np.zeros((int(n_case/2),nt,ny,nx))
   ps_out2 = np.zeros((int(n_case/2),nt,ny,nx))

   MMM = 0
   KKK = 0
   for NNN in range(n_case):
       print(NNN,arr[NNN],var_name)
       dirname = '/vortex/jetstream/Arctic_midlatitude_waccm_whoi_ncar/PAMIP/' + case_name + '/' + arr[NNN] + '/'
       filename = arr[NNN] + '_monthly0_atm.nc'
       if arr[NNN][-10:-7] == 'w01' or arr[NNN][-10:-7] == 'b03':
#          print(NNN,arr[NNN],var_name)
          ds = xr.open_dataset(dirname + filename, decode_times=True)
          var_out1[MMM,:,:,:,:] = ds[var_name][:,:,:,:].data.copy()
          ps_out1[MMM,:,:,:] = ds["PS"][:,:,:].data.copy()
          ds.close()
          MMM = MMM + 1
       else:
          ds = xr.open_dataset(dirname + filename, decode_times=True)
          var_out2[KKK,:,:,:,:] = ds[var_name][:,:,:,:].data.copy()
          ps_out2[KKK,:,:,:] = ds["PS"][:,:,:].data.copy()
          ds.close()
          KKK = KKK + 1

   var_out[0:100,:,:,:,:] = var_out1.copy()
   var_out[100:,:,:,:,:] = var_out2.copy()
   ps_out[0:100,:,:,:] = ps_out1.copy()
   ps_out[100:,:,:,:] = ps_out2.copy()

# Perform vertical interpolation
   var_out_regrid = np.zeros((n_case,nt,19,ny,nx))

   for NNN in range(n_case):
       for NT in range(nt):
           print(NNN, NT)

# Output temporary nc-file
           filename_tmp1 = 'temp_var.nc'
           files = os.listdir(os.getcwd())
           for file in files:
               if file == filename_tmp1:
                  print('Delete ' + filename_tmp1)
                  os.system("rm -rf " + file)
           f = Dataset(filename_tmp1, 'w',format='NETCDF4')
           f.createDimension('dz', nz)
           f.createDimension('dy', ny)
           f.createDimension('dx', nx)
           f.createDimension('dnull', 1)
           hyam_out = f.createVariable('hyam','f4',('dz'))
           hybm_out = f.createVariable('hybm','f4',('dz'))
           P0_out = f.createVariable('P0','f4',('dnull'))
           var_tmp1_out = f.createVariable('var_tmp1','f4',('dz','dy','dx'))
           ps_tmp_out = f.createVariable('ps','f4',('dy','dx'))
           var_tmp1_out[:,:,:] = var_out[NNN,NT,:,:,:].copy()
           ps_tmp_out[:,:] = ps_out[NNN,NT,:,:].copy()
           hyam_out[:] = hyam[:].copy()
           hybm_out[:] = hybm[:].copy()
           P0_out[:] = P0.copy()
           f.close()

# Execute ncl script
           os.system("ncl var_tzyx_vertical_regrid_cal.ncl")

# Read interpolated field
           f = Dataset("output_from_ncl.nc", 'r')
           var_regrid = f.variables['var_regrid'][:,:,:].data
           lev_regrid = f.variables['lev'][:].data
           f.close()
           nz_regrid = len(lev_regrid)
           var_regrid[abs(var_regrid)>1.e30] = np.nan
#           var_mask = (var_regrid/var_regrid).copy()
           lat_regrid = lat[:].copy()
           ny_regrid = len(lat_regrid)
           var_out_regrid[NNN,NT,:,:,:] = var_regrid.copy()

   dirname = '/vortex/jetstream/Arctic_midlatitude_waccm_whoi_ncar/PAMIP/' + case_name + '/' + arr[NNN] + '/'
   filename = arr[NNN] + '_monthly0_atm.nc'
   ds = xr.open_dataset(dirname + filename, decode_times=True)

# Output file
   filename_out = var_name + '_' + case_name + '_coupled_whoi_ncar.nc'
   files = os.listdir(os.getcwd())
   for file in files:
       if file == filename_out:
          print('Delete ' + filename_out)
          os.system("rm -rf " + file)
   f = Dataset(filename_out, 'w',format='NETCDF4')
   f.description = '1-100: AMV+/IPV-; 101-200: AMV-/IPV+'
   f.createDimension('lev', nz_regrid)
   f.createDimension('lat', ny_regrid)
   f.createDimension('lon', nx)
   f.createDimension('time', nt)
   f.createDimension('ens', n_case)
   lev_out = f.createVariable('lev','f4',('lev'))
   lat_out = f.createVariable('lat','f4',('lat'))
   lon_out = f.createVariable('lon','f4',('lon'))
   time_out = f.createVariable('time','f4',('time'))
   var_out_out = f.createVariable(var_name,'f4',('ens','time','lev','lat','lon'))

   lev_out[:] = lev_regrid[:].copy()
   lev_out.standard_name = 'vertical pressure level'
   lev_out.long_name = 'pressure level'
   lev_out.units = 'hPa'

   lat_out[:] = lat_regrid[:].copy()
   lat_out.standard_name = ds['lat'].attrs['standard_name']
   lat_out.long_name = ds['lat'].attrs['long_name']
   lat_out.units = ds['lat'].attrs['units']
   lat_out.axis = ds['lat'].attrs['axis']

   lon_out[:] = lon[:].copy()
   lon_out.standard_name = ds['lon'].attrs['standard_name']
   lon_out.long_name = ds['lon'].attrs['long_name']
   lon_out.units = ds['lon'].attrs['units']
   lon_out.axis = ds['lon'].attrs['axis']

   time_out[:] = time[:].copy() 
   time_out.standard_name = ds['time'].attrs['standard_name'] 
   time_out.long_name = ds['time'].attrs['long_name']
   time_out.bounds = ds['time'].attrs['bounds']
   time_out.axis = ds['time'].attrs['axis']

   var_out_out[:,:,:,:,:] = var_out_regrid[:,:,:,:,:].copy()
   var_out_out.long_name = ds[var_name].attrs['long_name']
   var_out_out.units = ds[var_name].attrs['units']
   var_out_out.cell_methods = ds[var_name].attrs['cell_methods']

   f.close()

   ds.close()
