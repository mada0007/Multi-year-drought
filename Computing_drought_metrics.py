

from netCDF4 import Dataset
import numpy as np
import glob
import sys
import xarray as xr
import os

### Set paths ###
data_path = "/g/data/w35/ma9839/DATA_OBS/"              # CMIP models path
spi_path = '/g/data/w35/ma9839/OBS_SPI/AWAP_SPI/'           # SPI 1 data files
lib_path = '/g/data/w35/ma9839/RQ2_functions' # SPI functions
from drought_metrics_SPI import *


sys.path.append(os.path.abspath(lib_path))

### Set variable ###
var_name = "pr"
var_path = "Precip"

#######################
### Set experiments ###
#######################

# CMIP5 experiments to process (loops through these, can set several)
experiment = ['histortical']  ########### not sure what this does and how to represnt in my case#****************

#################
### Set years ###
#################

# Set years
start_yr = 1979
end_yr = 2008




# year indices to extract correct data years
# This is a bit redundant, leave for now and fix later
# yr_ind = np.array(range((0, (end_yr-start_yr)*12 + 12))  #Need to add one to end year, silly python


############################
#### Set drought limits ####
############################

# SPI scale (loops through these, can set several)

scale = 3  # [1,3,6]

# Threshold (see drought_metrics_SPI function for details)
severity = ['moderate']

# Name of NC-variable in SPI file (also used to create file names)
fun = "SPI"

# If want to additionally return drought metrics for specific drought lenghts, set this
bins = np.arange(1, 12 + 1)  # drought durations to count

# Use observations to determine drought threshold?
obs_ref = False  # NOT IMPLEMENTED properly in the code itself atm, only for creating output path

#########################
### Calculate metrics ###
#########################

# Loop through severities (thresholds)
for v in range(len(severity)):

    # Loop through experiments

    for k in range(1):

        # List all model names
        models = sorted((os.listdir(data_path)))[0]
        print(models)

        print(models)

        # Loop through models
        for m in range(len(models)):
            if models[m].startswith('.'):  # get rid of missing data
                continue

            # if not models[m].endswith('.nc'):
            #     continue

            ### Find CMIP5 files ###
            # files = (glob.glob(data_path + "/" + models[m]))

            # print(files)

            # ### Load data ###

            # # Model data
            # dset = xr.open_dataset(files[0],decode_times=False)
            # fh = Dataset(files[0], mode='r')



            # all_data = fh.variables[precip]

            # data = all_data
            # mask = all_data.mask

            data = xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/AWAP_25.nc').precip
            print(data)





            lat = data.lat

            lon = data.lon

            # fh.close()

            #  Mask missing values
            miss_val = -99999.0
            # data[mask==True] = miss_val

            # Loop through SPI scales
            # for s in range(len(scale)):

            ### Load SPI data ###

            # Find SPI files
            # spifile = (glob.glob(spi_path + "/" + models[m]))


            # Modify for AWAP
            spifile = xr.open_dataset('/g/data/w35/ma9839/OBS_SPI/AWAP_SPI/AWAP_1x1.nc', decode_times=False)
            print(spifile)

            # Load data
            # spi_fh = Dataset(spifile[0], mode='r')
            spi_all = spifile.SPI1

            print(spi_all.shape, data.shape)


            # print(spi_all)

            # R-generated SPI value file does not include the first few months if scale > 1
            # (these would be all NA and were not written to the netcdf)
            # Add these layers to spi_data so matches the shape of data

            spi_data = np.zeros((len(data), len(lat), len(lon))) * np.nan
            spi_mask = np.ones((len(data), len(lat), len(lon)), dtype=bool)

            #        spi_data[scale[s] - 1: len(data)] = spi_all.data
            #       spi_mask[scale[s] - 1: len(data)] = spi_all.mask

            # Not sure why some files have northing instead of latitude...
            #  try:
            #     spi_lat = spi_fh.variables['latitude'][:]
            #  except:
            #     spi_lat = spi_fh.variables['northing'][:]

            # spi_fh.close()

            # Not sure why but python reads some files upside down
            # Flip latitude if reads map upside down so model matches SPI data

            # If model upside down
            # if (lat[0] < 0 and spi_lat[0] > 0):
            #  print "Flipping MODEL data, experiment: ", experiment[k], " model:", models[m]
            #   data = data[:, ::-1, :]
            # replace lat with spi_lat (otherwise written to file the wrong way round)
            #  lat = spi_lat
            # If SPI upside down
            #  elif (lat[0] > 0 and spi_lat[0] < 0):
            #    print "Flipping SPI data, experiment: ", experiment[k], " model:", models[m]
            #  spi_data = spi_data[:, ::-1, :]

            ### Initialise output arrays ###
            duration = np.zeros((len(data), len(lat), len(lon))) + miss_val  # * np.nan
            magnitude = np.zeros((len(data), len(lat), len(lon))) + miss_val  # * np.nan
            intensity = np.zeros((len(data), len(lat), len(lon))) + miss_val  # * np.nan
            rel_intensity = np.zeros((len(data), len(lat), len(lon))) + miss_val  # * np.nan
            timing = np.zeros((len(data), len(lat), len(lon))) + miss_val  # * np.nan

            count_duration = np.zeros((len(bins), len(lat), len(lon))) + miss_val  # * np.nan
            count_magnitude = np.zeros((len(bins), len(lat), len(lon))) + miss_val  # * np.nan
            count_intensity = np.zeros((len(bins), len(lat), len(lon))) + miss_val  # * np.nan
            count_rel_intensity = np.zeros((len(bins), len(lat), len(lon))) + miss_val  # * np.nan

            spi_magnitude = np.zeros((len(data), len(lat), len(lon))) + miss_val  # * np.nan
            spi_count_magnitude = np.zeros((len(bins), len(lat), len(lon))) + miss_val  # * np.nan
            spi_intensity = np.zeros((len(data), len(lat), len(lon))) + miss_val  # * np.nan
            spi_count_intensity = np.zeros((len(bins), len(lat), len(lon))) + miss_val  # * np.nan

            ### Calculate metrics ###

            print ('Scale ', '/', ' Experiment ', '/',
                   len(experiment), ', model: ', m + 1, "/", len(models))



            # Loop through grid cells
            for i in range(len(lat)):

                for j in range(len(lon)):
                    print(i,j)
                    mod_vec = data[:, i, j]

                    # Extract data for pixel
                    spi_vec = spi_all[:, i, j]
                    print(len(spi_vec), len(mod_vec))

                    # if any(~spi_mask[:, i, j]):
                    #     # # Extract data for pixel
                    #     mod_vec = data[:, i, j]
                    #     spi_vec = spi_data[:, i, j]
                    spi_vec[np.isnan(spi_vec)]=0
                    mod_vec[np.isnan(mod_vec)]=0


                    # Set spi_vec to NaN, where spi values missing
                    # ind = np.where(spi_mask[:, i, j] == True)[0]
                    # spi_vec[ind] = np.nan
                    #print(spi_vec)





                    #Calculate drought metrics
                    metric = drought_metrics_SPI(mod_vec=mod_vec, spi_vec=spi_vec,
                                                 lib_path=lib_path, severity=severity[v],
                                                 scale=scale, count=bins, miss_val=miss_val)
                    #print("metric_intensity is:",metric['rel_intensity'])
                    #print("metric-duration is:", metric['duration'])
                    #print("metric_count_rel_intensity:", metric['count_rel_intensity'])



                    ### Write metrics to variables ###

                    # Metrics for all events
                    duration[range(np.size(metric['duration'])), i, j] = metric[
                        'duration']  # total drought duration (months)
                    print((metric['duration']))



                    magnitude[range(np.size(metric['magnitude'])), i, j] = metric['magnitude']  # average magnitude

                    intensity[range(np.size(metric['intensity'])), i, j] = metric['intensity']  # average intensity
                    #print((metric['intensity']))

                    rel_intensity[range(np.size(metric['rel_intensity'])), i, j] = metric[
                        'rel_intensity']  # average intensity
                    #print(metric['rel_intensity'])

                    timing[range(np.size(metric['timing'])), i, j] = metric['timing']  # drought timing (month index)

                    # Metrics for specific drought lengths
                    count_duration[range(np.size(metric['count_duration'])), i, j] = metric[
                        'count_duration']  # total drought duration (months)
                    #print(metric['count_duration'])


                    count_magnitude[range(np.size(metric['count_magnitude'])), i, j] = metric[
                        'count_magnitude']  # total drought duration (months)
                    #print(count_magnitude)

                    count_intensity[range(np.size(metric['count_intensity'])), i, j] = metric[
                        'count_intensity']  # total drought duration (months)

                    count_rel_intensity[range(np.size(metric['count_rel_intensity'])), i, j] = metric[
                        'count_rel_intensity']  # total drought duration (months)

                    # Metrics using SPI units
                    spi_magnitude[range(np.size(metric['SPI_magnitude'])), i, j] = metric[
                        'SPI_magnitude']  # total drought duration (months)

                    spi_count_magnitude[range(np.size(metric['SPI_count_magnitude'])), i, j] = metric[
                        'SPI_count_magnitude']  # total drought duration (months)

                    spi_intensity[range(np.size(metric['SPI_intensity'])), i, j] = metric[
                        'SPI_intensity']  # total drought duration (months)
                    #print(spi_intensity)

                    spi_count_intensity[range(np.size(metric['SPI_count_intensity'])), i, j] = metric[
                        'SPI_count_intensity']  # total drought duration (months)

        ##############################
            ### Write result to NetCDF ###
            ##############################

            # Create output path name
            out_path = ("/g/data/w35/ma9839/OBS_DROUGHT3" + models[m])
            print(out_path)

            # If using obs ref
            if obs_ref:
                out_path = out_path + "_" + obs_var  # not implemented atm...

            # Add info on SPI scale and threshold
            out_path = out_path + '/SPI_3' + severity[v] + '_scale_' + str(scale) + '/'

            # Create out directory if doesn't exist
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            # Create output file name
            out_file = (out_path + '/' + models[m])

            # Open a new netCDF file for writing
            ncfile = Dataset(out_file, 'w', format="NETCDF4_CLASSIC")

            # Create the output data
            # Create the x, y and time dimensions
            ncfile.createDimension('lat', lat.shape[0])
            ncfile.createDimension('lon', lon.shape[0])
            ncfile.createDimension('time', len(data))
            ncfile.createDimension('count', len(bins))

            # Create dimension variables

            longitude = ncfile.createVariable("lon", 'f8', ('lon',))
            latitude = ncfile.createVariable("lat", 'f8', ('lat',))
            time = ncfile.createVariable("time", 'i4', ('time',))
            count = ncfile.createVariable("count", 'i4', ('count',))

            # Create data variables
            data_dur = ncfile.createVariable('duration', 'i4', ('time', 'lat', 'lon'), fill_value=miss_val)

            data_mag = ncfile.createVariable('magnitude', 'f8', ('time', 'lat', 'lon'), fill_value=miss_val)
            data_int = ncfile.createVariable('intensity', 'f8', ('time', 'lat', 'lon'), fill_value=miss_val)
            data_rel_int = ncfile.createVariable('rel_intensity', 'f8', ('time', 'lat', 'lon'), fill_value=miss_val)
            data_tim = ncfile.createVariable('timing', 'i4', ('time', 'lat', 'lon'), fill_value=miss_val)

            data_cdur = ncfile.createVariable('count_duration', 'i4', ('count', 'lat', 'lon'), fill_value=miss_val)
            data_cmag = ncfile.createVariable('count_magnitude', 'f8', ('count', 'lat', 'lon'), fill_value=miss_val)
            data_cint = ncfile.createVariable('count_intensity', 'f8', ('count', 'lat', 'lon'), fill_value=miss_val)
            data_rel_cint = ncfile.createVariable('count_rel_intensity', 'f8', ('count', 'lat', 'lon'), fill_value=miss_val)

            data_smag = ncfile.createVariable('spi_magnitude', 'f8', ('time', 'lat', 'lon'), fill_value=miss_val)
            data_scmag = ncfile.createVariable('spi_count_magnitude', 'f8', ('count', 'lat', 'lon'), fill_value=miss_val)
            data_sint = ncfile.createVariable('spi_intensity', 'f8', ('time', 'lat', 'lon'), fill_value=miss_val)
            data_scint = ncfile.createVariable('spi_count_intensity', 'f8', ('count', 'lat', 'lon'), fill_value=miss_val)

            # Set variable attributes
            longitude.units = 'degrees_east'
            latitude.units = 'degrees_north'

            data_dur.long_name = 'drought event duration (no. months)'
            data_mag.long_name = 'drought event magnitude (mm)'
            data_int.long_name = 'drought event intensity (mm)'
            data_rel_int.long_name = 'drought event relative intensity (%)'
            data_tim.long_name = 'drought event timing (month index)'

            data_cdur.long_name = 'number of drought events of length specified in count'
            data_cmag.long_name = 'mean magnituge of drought events of length count (mm)'
            data_cint.long_name = 'mean intensity of drought events of length count (mm)'
            data_rel_cint.long_name = 'mean relative intensity of drought events of length count (%)'

            data_smag.long_name = 'drought event magnitude in SPI units (sd)'
            data_scmag.long_name = 'mean magnituge of drought events of length count in SPI units (sd)'
            data_sint.long_name = 'drought event intensity in SPI units (sd)'
            data_scint.long_name = 'mean intensity of drought events of length count in SPI units (sd)'

            # Write data to variable
            longitude[:] = lon
            latitude[:] = lat
            time[:] = range(1, len(data) + 1)
            count[:] = bins

            data_dur[:, :, :] = duration
            data_mag[:, :, :] = magnitude
            data_int[:, :, :] = intensity
            data_rel_int[:, :, :] = rel_intensity
            data_tim[:, :, :] = timing

            data_cdur[:, :, :] = count_duration
            data_cmag[:, :, :] = count_magnitude
            data_cint[:, :, :] = count_intensity
            data_rel_cint[:, :, :] = count_rel_intensity

            data_smag[:, :, :] = spi_magnitude
            data_scmag[:, :, :] = spi_count_magnitude
            data_sint[:, :, :] = spi_intensity
            data_scint[:, :, :] = spi_count_intensity

            # Close the file
            ncfile.close()
