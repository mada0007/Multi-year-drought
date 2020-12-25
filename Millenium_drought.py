## This script will be used in millenium drought analysis for AILIE's

## Author: Mustapha Adamu

## Date:   20th June 2020

#*************************************************************************

# importing need libraries 

import sys
import os
import xarray as xr
import cartopy.crs as ccrs  # This a library for making 2D spatial plots in python
import matplotlib
# matplotlib.use("TKAgg")
import matplotlib.pyplot as plt  # Also for plotting in python
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy as cart
#import dask.array
#from dask.diagnostics import ProgressBar
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from cartopy.util import add_cyclic_point
import numpy as np
#from scipy.stats import kurtosis, skew
#from sklearn.utils import resample
import scipy
import scipy.signal
import scipy.stats as s
import cartopy as cart

import matplotlib as mpl

import cmocean as cm  #special library for making beautiful colormaps
import glob
import sys
import os
import cartopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import xarray as xr


import grid_tools
import timeseries_tools

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#**************************************************************************

def spi(series, x):
    import numpy as np
    from scipy.stats import gamma
    from timeseries_tools import moving_average
    from scipy.interpolate import interp1d

    # Take monthly data and compute into moving averages of length, X
    # Note that series will be length n-(x-1) following the application of
    # the moving average
    series = moving_average(series, x)

    # The data must be stratified into months so that the SPI may be computed
    # for each month so data is deseasonalised.
    # Reshape the array to stratify into months.
    lens = len(series)
    lens = lens / 12
    lens = int(lens)
    series = series.reshape([lens, 12])

    # Create dummy arrays to store the SPI data
    z = np.zeros([lens, 12], float)
    print(z.shape)

    # Compute the SPI, one month at a time
    for i in range(0, 12):
        tmp = series[:, i]
        tmpz = spicalc(tmp)
        print(len(tmpz))
        z[:, i] = tmpz

    # Reshape the array into it's original time series
    return z

## function to compute SPI from time series obtained from AILIE

def spicalc(data):
    import numpy as np
    from scipy.stats import gamma

    # set any values that are not NaN but below zero, to NaN
    data[data < 0.0] = np.nan

    # remove any NaNs (i.e. missing numbers) from data so only real numbers exist
    tmp = data[~np.isnan(data)]
    print(len(tmp))

    # if there are less than 10 real datum with which to fit the distribution,
    # then return an array of NaN, otherwise, do the calculations

    spireturn = np.zeros(len(data)) + np.nan

    if len(tmp) > 10:

        # compute the shape and scale parameters using more than one non-zero data point
        # otherwise computation of the log will fail
        tmpnonz = tmp[np.where(tmp > 0.0)]
        if len(tmpnonz) > 1:
            A = np.log(np.mean(tmpnonz)) - (np.sum(np.log(tmpnonz)) / len(tmpnonz))
            shp = (1.0 / (4 * A)) * (1 + ((1 + ((4 * A) / 3)) ** 0.5))
            scl = np.mean(tmpnonz) / shp
            gam = gamma.cdf(tmpnonz, shp, scale=scl)
        else:
            # if there are no or one non-zero number, then the probability of non-zero numbers
            # is set as 0 or 1/len(tmp) (depending on len(tmpnonz))
            gam = len(tmpnonz) / len(tmp)

        # fit the gamma distribution, G(x), already calculated as gam if there is more than
        # one non-zero number in the time series
        # if there are zero values, the cdf becomes H(x) = q + (1-q)G(x) where q is the
        # probability of a zero value in the time series
        numzero = len(tmp[np.where(tmp == 0.0)])

        if numzero > 0:
            q = numzero / len(tmp)
            gam = q + (1 - q) * gam
            gcdf = np.zeros(len(tmp))
            i = np.where(tmp > 0.0)
            j = np.where(tmp == 0.0)
            gcdf[i] = gam
            gcdf[j] = q
        else:
            gcdf = gam

        # define the constants for the approximation
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        # compute the SPI values when the gamma cdf is non-uniform
        if len(gcdf[np.where(gcdf == 1.0)]) == 0:
            t = np.where(gcdf <= 0.5, (np.log(1 / (gcdf ** 2))) ** 0.5, (np.log(1 / ((1.0 - gcdf) ** 2)) ** 0.5))
            ztmp = (t - ((c0 + c1 * t + c2 * (t ** 2)) / (1 + d1 * t + d2 * (t ** 2) + d3 * (t ** 3))))
            s = np.where(gcdf <= 0.5, -1 * ztmp, ztmp)

        # if the grid cell is always dry (i.e. precip of zero, then SPI returns 0s as dry
        # is always "normal"
        # print(s)
        else:
            s = np.zeros(len(gcdf))

        # spireturn[~np.isnan(data)] = s

    return s


## import data and find compute SE Australia Areal Average:::

## cordinates to select SE_ AUSTRALIA
#-- SE_AUSTRALIA
en_lat_bottom = -45
en_lat_top = -33
en_lon_left = 135
en_lon_right = 153


def get_area_mean(tas, lat_bottom, lat_top, lon_left, lon_right):
    """The array of mean temperatures in a region at all time points."""
    return tas.loc[:, lat_bottom:lat_top, lon_left:lon_right].mean(dim=('lat','lon'))


precip = xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/AWAP_25.nc').precip

## SE AUSTRALIA _TIME 
SE_AUS = get_area_mean(precip,en_lat_bottom,en_lat_top,en_lon_left,en_lon_right) ## SE_AUS_TS

SE_AUS = np.load('/g/data/w35/ma9839/OBS_SPI/AWAP_SPI/SPI1_ts.npy')

SE_spi = xr.DataArray(SE_AUS, coords=dict(time=precip.time) ,dims=['time'])

D1 = SE_spi.sel(time=slice('1911', '1920'))

D2 = SE_spi.sel(time=slice('1921', '1930'))

D3 = SE_spi.sel(time=slice('1931', '1940'))

D4 = SE_spi.sel(time=slice('1941', '1950'))

D5 = SE_spi.sel(time=slice('1951', '1960'))

D6 = SE_spi.sel(time=slice('1961', '1970'))

D7 = SE_spi.sel(time=slice('1971', '1980'))

D8 = SE_spi.sel(time=slice('1981', '1990'))

D9 = SE_spi.sel(time=slice('1991', '2000'))

D10 = SE_spi.sel(time=slice('2001', '2010'))

D11 = SE_spi.sel(time=slice('2011', '2019'))

decades = [D1,D2,D3,D4,D5,D6,D7,D8,D9,D10, D11]
thresh = -1

## Loop to get lenght of dry months

dry_decs=[]

for data in decades:

    dd = np.where(data< thresh)

    dd = data[dd]

    dry_decs.append(len(dd))



thresh = 1

## Loop to get lenght of dry months

wet_decs=[]

for data in decades:

    dd = np.where(data>thresh)

    dd = data[dd]

    wet_decs.append(len(dd))


#*****************************************************************************
# Working on gridded dataset
lon = precip.lon
lat = precip.lat
time= precip.time
data = xr.open_dataset('/g/data/w35/ma9839/OBS_SPI/AWAP_SPI/AWAP_SPI1.nc', decode_times=False).SPI1
spi_data = xr.DataArray(data, coords={'time':precip.time, 'lat':precip.lat, 'lon':precip.lon}, dims=['time','lat','lon'])
reg_data =  xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/reg_as.nc').p
print(reg_data.shape)


## for millenium drought
D12 = spi_data.sel(time=slice('2001', '2009'))
grids_per_drought = np.zeros((precip.mean(dim='time')).shape) * np.nan
for x in range(len(data.lat)):

    for y in range(len(data.lon)):

            ts = D12[:, x, y]

            t_ind = np.where(ts < -1)  ## geting index of gridcell with below threshold of drought



            grids_per_drought[x, y] = len(ts[t_ind]) 
            # grids_per_drought[x, y] = (len(ts[t_ind]) / len(ts)) * 100
 

            # print(grids_per_drought[x,y],x,y)
millenium_dry = xr.DataArray(grids_per_drought, coords={'lat': lat, 'lon': lon},
                  dims=['lat', 'lon'])


## for climatology 
D12 = spi_data
grids_per_drought = np.zeros((precip.mean(dim='time')).shape) * np.nan
for x in range(len(lat)):

    for y in range(len(lon)):

            ts = D12[:, x, y]

            t_ind = np.where(ts < -1)  ## geting index of gridcell with below threshold of drought


            grids_per_drought[x, y] = len(ts[t_ind]) 
            # grids_per_drought[x, y] = (len(ts[t_ind]) / len(ts)) * 100


            # print(len(ts[t_ind]))

dry_clim = xr.DataArray(grids_per_drought, coords={'lat': lat, 'lon': lon},
                  dims=['lat', 'lon'])


# levels = [20,100,150,600]

## for climatology 

print('NOW Plotting')

all_data = [dry_decs, wet_decs, dry_clim, millenium_dry] ## list of data for plotting 1st 2 timeseries


#******************************************************************************

years = ['D1(1911-1919)','D2(1920-1929)','D3(1930-1939)','D4(1940-1949)','D5(1950-1959)','D6(1960-1969)','D7(1970-1979)','D8(1980-1989)','D9(1990-1999)','D10(2000-2009)','D11(2010-2019)']
decades = ['D1','D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11']
levels = np.arange(0,40,5)

proj = ccrs.PlateCarree()
# fig, ax = plt.subplots(2, 2, figsize=(10,10), gridspec_kw={'wspace': 0.1, 'hspace': -0.1},
#                           subplot_kw=dict(projection=proj))
# axes = ax.flatten()

import itertools
import matplotlib.patches as mpatches

def flip(items, ncol):
    'function for flipping legends: https://stackoverflow.com/questions/10101141/matplotlib-legend-add-items-across-columns-instead-of-down'
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

y=np.arange(0,35,5)
x=np.arange(1,12,1)
fig = plt.figure(figsize=(12,12),)
ax1 = plt.subplot(221,)
ax1.bar(x, all_data[0], label=years)
ax1.set_xticks(x)
ax1.set_yticks(y)
ax1.set_xticklabels(decades)
ax1.set_title('Months with SPI below -1', fontsize=12,weight='bold')
ax1.grid(linestyle='--')

ax1.tick_params(axis='both', which='major', labelsize=10)

#**********************
#function to create legend handles::::::::::;
p_all = []
for i in range(len(years)):
    p = mpatches.Patch(color='white', label=years[i])

    p_all.append(p)

#*************************

ax2 = plt.subplot(222,)

ax2.bar(x, all_data[1])
ax2.set_xticks(x)
ax2.set_yticks(y)
ax2.grid(linestyle='--')
ax2.set_xticklabels(decades)
ax2.set_title('Months with SPI above 1', fontsize=12,weight='bold')
ax2.legend(loc=(0.73, 0.5),fontsize=7,handles=p_all, ncol=1)



ax3 = plt.subplot(223, projection=proj)

h = all_data[2].plot.contourf(ax=ax3,cmap='rainbow',levels =[0,80,100, 120, 130, 140, 150, 160,165, 170, 175, 180, 185, 190, 195, 200,220, 240, 260, 300, 350 ] ,add_labels=False,add_colorbar=False,)
ax3.coastlines()
# ax3.set_extent([110, 155,-48, -9.5, ], ccrs.PlateCarree())
ax3.set_title('Drought Months (1911-2019)', fontsize=12,weight='bold')
ax3.set_yticks([-43, -35, -25, -15])
ax3.set_xticks([110,120,130,140,150])
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax3.xaxis.set_major_formatter(lon_formatter, )
ax3.yaxis.set_major_formatter(lat_formatter)
ax3.tick_params(axis='both', which='major', labelsize=10)
ax3.grid(linestyle='--',linewidth=1, color='k',alpha=0.5)
ax3.add_feature(cart.feature.OCEAN, zorder=1,facecolor=cartopy.feature.COLORS['land_alt1'])

ax3.axis('tight')
# plt.title('PERCENTAGE OF MONTHS IN SD DURING MD (AS)')

print(all_data[3])
levels  = np.arange(0,40,5)

ax4 = plt.subplot(224,  projection=proj)
h1 = all_data[3].plot.contourf(ax=ax4,cmap='rainbow',levels=levels,add_labels=False,add_colorbar=False,)
ax4.coastlines(resolution='110m', zorder=3)
# ax4.set_extent([110, 155,-48, -9.5, ], ccrs.PlateCarree())
ax4.set_title('Millenium Drought (2001-2009)', fontsize=12,weight='bold')
ax4.set_yticks([-43, -35, -25, -15])
ax4.set_xticks([110,120,130,140,150])
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax4.xaxis.set_major_formatter(lon_formatter, )
ax4.yaxis.set_major_formatter(lat_formatter)
ax4.tick_params(axis='both', which='major', labelsize=10)
ax4.grid(linestyle='--', linewidth=1, color='k',alpha=0.5)
ax4.add_feature(cart.feature.OCEAN, zorder=1,facecolor=cartopy.feature.COLORS['land_alt1'])
ax4.axis('tight')

# plt.title('PERCENTAGE OF MONTHS IN SD DURING MD (AS)')

cbar = fig.colorbar(h, ax=ax3,orientation='horizontal', shrink=0.55,)
cbar.set_label("no. of months with SPI < -1", fontsize=10,labelpad=8,weight='bold')

cbar = fig.colorbar(h1, ax=ax4,orientation='horizontal', shrink=0.55,)
cbar.set_label("no. of months with SPI < -1", fontsize=10,labelpad=8,weight='bold')
# axes[3].set_ylabel('APR-SEP',fontsize=18)

    #cbar.ax.set_ylabel('Correlation', rotation=270,**csfont, fontsize='large')
cbar.ax.tick_params(labelsize=12)
# plt.subplots_adjust(left=1, right=0)
# plt.subplots_adjust(left=0.5, right=0.5,)
plt.savefig('/g/data/w35/ma9839/FOR_AILIE/Results/main_test_1')

plt.show()

    



















print(len(precip), len(data))