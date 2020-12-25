## plotting drought duration metrics for AILIE

import sys
import os
import xarray as xr
import cartopy.crs as ccrs  # This a library for making 2D spatial plots in python
# import matplotlib
# matplotlib.use("TKAgg")
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt  # Also for plotting in python
plt.switch_backend('agg')
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.util import add_cyclic_point
import numpy as np
#from scipy.stats import kurtosis, skew
#from sklearn.utils import resample
import cmocean as cm  #special library for making beautiful colormaps
import cmocean.cm as cmo
from math import sqrt
import seaborn as sns
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy as cart
import cartopy
import metpy.calc as mpcalc 


##************************ **********************
## Importing data

# duration_spi1 = xr.open_dataset('/g/data/w35/ma9839/OBS_DROUGHT1A/SPI_3moderate_scale_3/A').duration
# duration_spi1 = xr.open_dataset("/g/data/w35/ma9839/OBS_DROUGHT1A/SPI_3moderate_scale_3/A").duration
# duration_spi2 = xr.open_dataset("/g/data/w35/ma9839/FOR_AILIE/years_all_actual.nc")
duration_spi2 = xr.open_dataset("/g/data/w35/ma9839/FOR_AILIE/years_all_actual.nc").years
print(np.min(duration_spi2))
 
duration_spi1 = xr.open_dataset('/g/data/w35/ma9839/OBS_DROUGHT1A/SPI_3moderate_scale_3/A').duration

# max_dur = np.max(duration_spi1, 0)
duration_spi1.mean(dim='time').plot()
plt.savefig('/g/data/w35/ma9839/FOR_AILIE/Results/test')
lat = duration_spi1.lat
lon = duration_spi1.lon

# ## create new data with year indices
# dur_mean = duration_spi1.mean(dim='time')

#**************************************************************************************
#************ this code is used to find the indices of where the max precipition occurs

# years_all = np.zeros(dur_mean.shape) * np.nan

# for i in range(len(lat)):

#     for j in range(len(lon)):

#         print(i,j)

#         ts = np.array(duration_spi1[:,i,j])
#         # print(ts)
    

#         max_ind = np.where(ts== np.nanmax(ts))

#         int_ts = np.array(int_spi1[:, i,j])


#         # if len(max_ind)>1:
#         #     years_all[i, j] = -888
        
        
#         # else:

#         year = int_ts[max_ind]

#         if len(year)>1:

#             years_all[i, j] = -888

#         else:

#             years_all[i, j] = year

#********************************************
#********************************************








#*** compute the actual section here**********

# awap_obs = xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/Awap_only/AWAP_25.nc').precip

# dur_mean = duration_spi1.mean(dim='time')
# years_all = np.zeros(dur_mean.shape) * np.nan

# years = xr.open_dataset('/g/data/w35/ma9839/FOR_AILIE/years_all.nc').__xarray_dataarray_variable__ 

# for i in range(len(lat)):

#     for j in range(len(lon)):

#         if np.mean(awap_obs[:,i,j]) !=0:

#             print(i,j)


#             ts = (awap_obs[:,i,j])

#             year_ind = np.array(years[i,j]).astype(int)

#             if year_ind == - 888:
#                 years_all[i,j] ==-888
#                 # print(year_ind)

#             else:


#                 y = ts[year_ind]

#                 years_all[i, j] = np.array(y['time'].dt.year)

#                 # print('y', np.array(y['time'].dt.year))


# years_all = xr.DataArray(years_all, coords = dict(lat=lat, lon=lon), dims=['lat', 'lon'],name='years')

# years_all.to_netcdf('/g/data/w35/ma9839/FOR_AILIE/years_all_actual.nc')
        

    


        




# duration_spi2 = xr.open_dataset('/g/data/w35/ma9839/OBS_DROUGHT1A/SPI_3moderate_scale_3/A').duration 





def get_area_mean(tas, lat_bottom, lat_top, lon_left, lon_right):
    """The array of mean temperatures in a region at all time points."""
    # print(tas.loc[lat_bottom:lat_top, lon_left:lon_right])
    return tas.loc[lat_bottom:lat_top, lon_left:lon_right].mean()


#-- SE_AUSTRALIA
en_lat_bottom = -37
en_lat_top = -28
en_lon_left = 103
en_lon_right = 122
duration_spi1 = duration_spi1.max(dim='time')

sw_aus_mean = np.nanmean((duration_spi1.sel(lat=slice(-36,-28), lon=slice(103,122))).values)
print('sw_aus_mean', sw_aus_mean)

sw_aus_median = np.nanpercentile((duration_spi1.sel(lat=slice(-36,-28), lon=slice(103,122))).values, q=50)
print('sw_aus_median', sw_aus_median)


sw_aus_mean = np.nanmean((duration_spi1.sel(lat=slice(-45,-24), lon=slice(139,151))).values)
print('se_aus_mean', sw_aus_mean)

sw_aus_median = np.nanpercentile((duration_spi1.sel(lat=slice(-45,-24), lon=slice(139,159))).values, q=50)
print('se_aus_median', sw_aus_median)


sw_aus_mean = np.nanmean((duration_spi1.sel(lon=slice(139,151))).values)
print('e_aus_mean', sw_aus_mean)

sw_aus_median = np.nanpercentile((duration_spi1.sel(lon=slice(139,159))).values, q=50)
print('e_aus_median', sw_aus_median)


sw_aus_mean = np.nanmean(duration_spi1)
print('aus_mean', sw_aus_mean)

sw_aus_median = np.nanpercentile(duration_spi1, q=50)
print('aus_median', sw_aus_median)








# print(f'SE_MEAN_OM: {np.array(get_area_mean(duration_spi2,en_lat_bottom, en_lat_top, en_lon_left, en_lon_right))}')

## computing means and medians for continent and regions::::::::



# print(duration_spi1)


##*************** compute maximum duration at each grid point **************
max_duration_spi1 = np.max(duration_spi1, 0)
# max_duration_spi2 = 
lon = max_duration_spi1.lon
lat = max_duration_spi1.lat


#********************************************
#**********plottiing
levels = [0,0.5,1.5,2,2.5,3,4.5,5,5.5, 6,6.5, 7,7.5, 8,8.5,9.5,10,10.5,11,12,14,17,20,22,30, 50, 70,80, 90,100,110]
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(1, 2, figsize=(12,7), sharex='all', sharey='all', gridspec_kw={'wspace': 0.2, 'hspace': -0.5},
                          subplot_kw=dict(projection=proj))
axes = ax.flatten()

ax3 = axes[0]
cmap ='Purples'

h = max_duration_spi1.plot.contourf(ax=ax3,cmap=cmap,levels =levels,add_labels=False,add_colorbar=False,)
ax3.coastlines()
# ax3.set_extent([110, 155,-48, -9.5, ], ccrs.PlateCarree())
ax3.set_title('Maximum Drought Duration SPI1', fontsize=12,weight='bold')
ax3.set_yticks([-43, -35, -25, -15])
ax3.set_xticks([110,120,130,140,150])
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax3.xaxis.set_major_formatter(lon_formatter, )
ax3.yaxis.set_major_formatter(lat_formatter)
ax3.tick_params(axis='both', which='major', labelsize=10)
ax3.grid(linestyle='--',linewidth=1, color='k',alpha=0.5)
ax3.add_feature(cart.feature.OCEAN, zorder=1,facecolor=cartopy.feature.COLORS['land_alt1'])



ax4 = axes[1]
levels = np.arange(1911, 2010, 2)
h1 = duration_spi2.plot.contourf(ax=ax4,cmap=cmap,levels=levels,add_labels=False,add_colorbar=False,)
# cs = ax4.contour(lon, lat, mpcalc.smooth_n_point(max_duration_spi2, 9, 2), levels, colors='w', linewidths=1)

ax4.coastlines(resolution='110m', zorder=3)
ax4.set_title('Maximum Drought Duration SPI3', fontsize=12,weight='bold')
ax4.set_yticks([-43, -35, -25, -15])
ax4.set_xticks([110,120,130,140,150])
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax4.xaxis.set_major_formatter(lon_formatter, )
ax4.yaxis.set_major_formatter(lat_formatter)
ax4.tick_params(axis='both', which='major', labelsize=10)
ax4.grid(linestyle='--', linewidth=1, color='k',alpha=0.5)
ax4.add_feature(cart.feature.OCEAN, zorder=1,facecolor=cartopy.feature.COLORS['land_alt1'])



cbar = fig.colorbar(h1, ax=axes,orientation='vertical', shrink=0.55,)
# cbar.set_label("Maximum Drought Duration", fontsize=10,labelpad=8,weight='bold')
cbar.ax.tick_params(labelsize=12)
plt.savefig('/g/data/w35/ma9839/FOR_AILIE/Results/maxduration_spi1_spi_python_'+cmap)

plt.show()

    
















