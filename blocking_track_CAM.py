#%%
###### This code is to track all blocking events with CAM outputs ######
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import cv2
import copy
import matplotlib.path as mpath
import pickle
import glob
from netCDF4 import Dataset
from datetime import date, timedelta
import xarray as xr

#%%
### A function to calculate distance between two grid points on earth ###
from math import radians, cos, sin, asin, sqrt
 
def haversine(lon1, lat1, lon2, lat2): # longitude1，latitude1，longitude2，latitude2 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6378 # earth radius
    return c * r * 1000



def blocking_type_v1(lwa_a_peak, lwa_c_peak, peaking_labels, peaking_x_index, ny, nx, dx):
    """ Method1: Only focus on the peaking date, compare the lwa_a and lwa_c """
    import numpy as np
    
    x_range = int(15/dx)+1

    lwa_a_sum = 0
    lwa_c_sum = 0
    
   
    ### Only focus on the block core region ###
    lwa_a_peak_roll = np.roll(lwa_a_peak,   int(nx/2)-int(peaking_x_index), axis=1)
    lwa_c_peak_roll = np.roll(lwa_c_peak,   int(nx/2)-int(peaking_x_index), axis=1)
    peaking_labels_roll = np.roll(peaking_labels, int(nx/2)-int(peaking_x_index), axis=1)

    ### We define the block core region is the +- 15 degrees around the peaking LWA location ###
    lwa_a_peak_core = lwa_a_peak_roll[  :, int(nx/2)-int(x_range):int(nx/2)+int(x_range)+1]
    lwa_c_peak_core =lwa_c_peak_roll[   :, int(nx/2)-int(x_range):int(nx/2)+int(x_range)+1]
    peaking_labels_core = peaking_labels_roll[   :, int(nx/2)-int(x_range):int(nx/2)+int(x_range)+1] 
    
    ### Sum up the lwa_a and lwa_c within the block center ###
    lwa_a_block = np.zeros((ny, 2*x_range+1))
    lwa_c_block = np.zeros((ny, 2*x_range+1))
    lwa_a_block[peaking_labels_core == True]  = lwa_a_peak_core[peaking_labels_core== True]
    lwa_c_block[peaking_labels_core == True]  = lwa_c_peak_core[peaking_labels_core == True]
    
    lwa_a_sum += lwa_a_block.sum()
    lwa_c_sum += lwa_c_block.sum()
    
    
    ### if the anticyclonic LWA is much stronger than cytclonic LWA, then it is defined as ridge ###
    ### if the anticyclonic LWA is comparable with cyclonic LWA, then it is defined as dipole ###
    ### if the anticyclonic LWA is weaker than cyclonic LWA, then it is defined as trough events ###
    if lwa_a_sum > 10 * lwa_c_sum :
        Btype = 0
    elif lwa_c_sum > 2 * lwa_a_sum:
        Btype = 1
    else:
        Btype = 2

    return Btype


def list2arr(var_list, n_events, max_dur):
    import numpy as np
    var_arr = np.full((n_events, max_dur), 0, dtype=np.int32)
    for i, xx in enumerate(var_list):
        L = len(xx)
        var_arr[i, :L] = xx
    return var_arr


#%%
### Read the daily Z500-based LWA ###
experiment_name = "CAM6_F2000climo_f19f19_plev_A"

file_LWA = "/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/LWA_Z500.nc" 
ds = xr.open_dataset(file_LWA)
LWA_Z = ds['LWA_Z500'].values
ds.close()

file_LWA_A = "/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/LWA_Z500_A.nc" 
ds = xr.open_dataset(file_LWA_A)
LWA_Z_A = ds['LWA_Z500_A'].values
ds.close()

file_LWA_C = "/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/LWA_Z500_C.nc" 
ds = xr.open_dataset(file_LWA_C)
LWA_Z_C = ds['LWA_Z500_C'].values
ds.close()

## Read lon and lat ##
path = glob.glob(r"/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/std*.nc")
path.sort()
path=path[365*10:365*80]
N=len(path)   #The number of nc files

### Read basic variables ###
file0 = Dataset(path[0],'r')
lon = file0.variables['lon'][:]
lat = file0.variables['lat'][:]
nlon = len(lon)
nlat = len(lat)
lat_mid = int(nlat/2)
lat_SH = lat[0:lat_mid]
lat_NH = lat[lat_mid:]
nlat_SH = len(lat_SH)
nlat_NH =len(lat_NH)
file0.close()


a= 6.378*1.e6 ## Earth radius in meters
slat0   = np.sin(lat*np.pi/180)
clat0   = np.cos(lat*np.pi/180)
clat_2d = abs(clat0[:, np.newaxis] * np.ones((nlat,nlon)))                    ## 2-D cos(fi) array
slat_2d = slat0[:, np.newaxis] * np.ones((nlat,nlon))                         ## 2-D sin(fi) array
dlat = (lat[1] - lat[0]) * np.pi /180                                            ## latitude spacing
dlon = (lon[1] - lon[0]) * np.pi /180                                            ## longitude spacing
dphi = a * dlat * clat_2d                                                     ## 2-D length differential element: a*cos(fi)*dlat
ds   = a**2 * clat_2d * dlat * dlon                                           ## 2-D area differential element:   a^2*cos(fi)*dlat*dlon
ds_NH = ds[lat_mid:,:] 
ds_SH = ds[0:lat_mid,:]

    
LWA_Z = LWA_Z[365*10:365*80,lat_mid:,:] 
LWA_Z_A = LWA_Z_A[365*10:365*80,lat_mid:,:] 
LWA_Z_C = LWA_Z_C[365*10:365*80,lat_mid:,:] 
nlat_hemi = nlat_NH
lat_hemi=lat_NH
ds_hemi = ds_NH

### Time Management, remove Feb2, 90 years in total ###
Datestamp0 = pd.date_range(start="2011-01-01",end="2080-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp0)})
Month0 = Date0['date'].dt.month 
Year0 = Date0['date'].dt.year
Day0 = Date0['date'].dt.day
Datestamp=[]
for d in np.arange(len(Datestamp0)):
    if Month0[d]==2 and Day0[d]==29:
        continue
    else:
        Datestamp.append(Datestamp0[d])
Date = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Month = Date['date'].dt.month 
Year = Date['date'].dt.year
Day = Date['date'].dt.day
Date = list(Date['date'])
nday = len(Date)


#%%
###### Detect Blocking ######
Blocking_total_lat = []
Blocking_total_lon = []
Blocking_total_date = []
B_freq = np.zeros((nlat_NH, nlon))         ### The final blocking frequency/total number 
B_freq2 = np.zeros((nday ,nlat_NH, nlon))  ### Record whether a grid point is under a blocking for a certain day (only 1 and 0 in this 3D array)

### Amplitude threshold and Duration threshold ###
LWA_max_lon = np.zeros((nlon*nday))
for t in np.arange(nday):    
    for lo in np.arange(nlon):
        LWA_max_lon[t*nlon+lo] = np.max(LWA_Z[t,:,lo])
Thresh = np.percentile(LWA_max_lon[:],50)

Duration = 5
print("Thresh is ")
print(Thresh)

### Wave Event ###
WE = np.zeros((nday,nlat_NH,nlon),dtype='int8') 
WE[LWA_Z>Thresh] = 255                    # Wave event

################### connected component-labeling algorithm ################
num_labels = np.zeros(nday)
labels = np.zeros((nday,nlat_NH,nlon))
for d in np.arange(nday):
    num_labels[d], labels[d,:,:], stats, centroids  = cv2.connectedComponentsWithStats(WE[d,:,:], connectivity=4)
    
####### connect the label around 0, since they are labeled separately ########
####### but actually they should belong to the same label  ########
labels_new = copy.copy(labels)
for d in np.arange(nday):
    if np.any(labels_new[d,:,0]) == 0 or np.any(labels_new[d,:,-1]) == 0:   ## If there are no events at either 0 or 357.5, then we don't need to do any connection
        continue
            
    column_0 = np.zeros((nlat_NH,3))       ## We assume there are at most three wave events at column 0 (actuaaly most of the time there is just one)
    column_end = np.zeros((nlat_NH,3))
    label_0 = np.zeros(3)
    label_end = np.zeros(3)
    
    ## Get the wave event at column 0 (0) ##
    start_lat0 = 0
    for i in np.arange(3):
        for la in np.arange(start_lat0, nlat_NH):
            if labels_new[d,la,0]==0:
                continue
            if labels_new[d,la,0]!=0:
                label_0[i]=labels_new[d,la,0]
                column_0[la,i]=labels_new[d,la,0]
                if labels_new[d,la+1,0]!=0:
                    continue
                if labels_new[d,la+1,0]==0:
                    start_lat0 = la+1
                    break 

        ## Get the wave event at column -1 (357.5) ## 
        start_lat1 = 0
        for j in np.arange(3):
            for la in np.arange(start_lat1, nlat_NH):
                if labels_new[d,la,-1]==0:
                    continue
                if labels_new[d,la,-1]!=0:
                    label_end[j]=labels_new[d,la,-1]
                    column_end[la,j]=labels_new[d,la,-1]
                    if labels_new[d,la+1,-1]!=0:
                        continue
                    if labels_new[d,la+1,-1]==0:
                        start_lat1 = la+1
                        break                       
            ## Compare the two cloumns at 0 and 357.5, and connect the label if the two are indeed connected
            if (column_end[:,i]*column_0[:,j]).mean() == 0:
                continue                
            if (column_end*column_0).mean() != 0:
                num_labels[d]-=1
                if label_0[i] < label_end[j]:
                    labels_new[d][labels_new[d]==label_end[j]] = label_0[i]
                    labels_new[d][labels_new[d]>label_end[j]] = (labels_new[d]-1)[labels_new[d]>label_end[j]]            
                if label_0[i] > label_end[j]:
                    labels_new[d][labels_new[d]==label_0[i]] = label_end[j]
                    labels_new[d][labels_new[d]>label_0[i]] = (labels_new[d]-1)[labels_new[d]>label_0[i]]
                    
                    
############ Now we get the maximum LWA location of each individule event ###########
############ Also we get the area or width of each individual event #########
lat_d = []; lon_d = []; lwa_max=[]; lwa_total =[]; lwa_mean = []; lwa_label=[]
lon_w = []; lon_e = []; area = []
lat_n = []; lat_s = []
lon_wide = []; lat_wide = []
for d in np.arange(nday):
    if int(num_labels[d]-1)==0:
        lat_d.append([np.nan])
        lon_d.append([np.nan])
        lon_w.append([np.nan])
        lon_e.append([np.nan])
        lat_n.append([np.nan])
        lat_s.append([np.nan])
        area.append([np.nan])
        lon_wide.append([np.nan])
        lat_wide.append([np.nan])
        lwa_max.append([np.nan])
        lwa_total.append([np.nan])
        lwa_mean.append([np.nan])
        lwa_label.append([np.nan])
        continue

    lat_list=[];    lon_list=[];   lwa_max_list=[]; lwa_total_list = []; lwa_mean_list = []; lwa_label_list = []
    lon_w_list = [];lon_e_list=[]; area_list = []
    lon_wide_list = []
    lat_n_list = []; lat_s_list=[]
    lat_wide_list = []
    
    for n in np.arange(0,int(num_labels[d]-1)):
        LWA_d = np.zeros((nlat_NH, nlon))
        LWA_d[labels_new[d]==n+1]=LWA_Z[d][labels_new[d]==n+1]   ### isolate that wave event ###
        
        ### Get the maximum location ###
        if len(np.array(np.where( LWA_d==LWA_d.max() ))[0])>1:           
            lat_list.append( lat_hemi[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[0][0]])
            lon_list.append( lon[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[1][0]])
            lwa_max_list.append( LWA_d.max())
        else:
            lat_list.append( lat_hemi[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[0]])
            lon_list.append( lon[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[1]])
            lwa_max_list.append( LWA_d.max())
                
        ### Get the total LWA ###
        lwa_total_list.append(LWA_d[labels_new[d]==n+1].sum())
        
        ### Get the mean LWA ###
        lwa_mean_list.append(np.mean(LWA_d[labels_new[d]==n+1]))
        
        ### Get all LWA of this event ###
        lwa_label_list.append(LWA_d)
            
        ### Get the west east boundary ###            
        for lo in np.arange(nlon):
            if (np.any(LWA_d[:,lo])) and (not np.any(LWA_d[:,lo-1])):
                lon_w_list.append(lon[lo])
            else:
                lon_w_list.append(0)
                
            if (not np.any(LWA_d[:,lo])) and (np.any(LWA_d[:,lo-1])):
                lon_e_list.append(lon[lo-1]) 
            else:
                lon_e_list.append(360)
                    
        ### Get the width from west to east ###
        if lon_e_list[-1]-lon_w_list[-1] > 0:
            lon_wide_list.append(lon_e_list[-1]-lon_w_list[-1])
        else:
            lon_wide_list.append(360+(lon_e_list[-1]-lon_w_list[-1]))

           
        ### Get the north south boundary ###
        for la in np.arange(nlat_hemi):
            if (not np.any(LWA_d[la,:])) and (np.any(LWA_d[la-1,:])):
                lat_n_list.append( lat_hemi[la])
            if (np.any(LWA_d[la,:])) and (not np.any(LWA_d[la-1,:])):
                lat_s_list.append( lat_hemi[la-1])
        ### Get the width from north to south ###
        lat_wide_list.append( lat_n_list[-1]-lat_s_list[-1]   ) 
            
                 
        ### Get the total area ###
        area_list.append(np.sum(ds_hemi[labels_new[d]==n+1]))
                    
                    
    lat_d.append(lat_list);   lon_d.append(lon_list); lwa_max.append(lwa_max_list); lwa_total.append(lwa_total_list); lwa_mean.append(lwa_mean_list); lwa_label.append(lwa_label_list)
    lon_w.append(lon_w_list); lon_e.append(lon_e_list);  area.append(area_list)
    lat_n.append(lat_n_list); lat_s.append(lat_s_list)
    lon_wide.append(lon_wide_list); lat_wide.append(lat_wide_list)


with open("/scratch/bell/liu3315/cesm_output/"+ experiment_name + "/LWA_Z500/Blocking/lon_d", "wb") as fp:
    pickle.dump(lon_d, fp)
with open("/scratch/bell/liu3315/cesm_output/"+ experiment_name + "/LWA_Z500/Blocking/lat_d", "wb") as fp:
    pickle.dump(lat_d, fp)

    
np.save("/scratch/bell/liu3315/cesm_output/"+ experiment_name + "/LWA_Z500/Blocking/labels_new.npy",labels_new)


#%%            
########### during the consecutive two days, find the pair events (with the shortest dististance) #############
########### the distance is limited to 18 degree longitude and 13.5 degree latitude #########
next_index = []
lon_thresh = 18
lat_thresh = 13.5
for d in np.arange(nday-1):
    next_index_day = np.full(len(lon_d[d]), np.nan)
    ### create a matrix that contains the distance between each WE during the consective two days ###
    shift_L = np.zeros((len(lon_d[d]),len(lon_d[d+1]) ))
    for i in np.arange(len(lon_d[d])):
        for j in np.arange(len(lon_d[d+1])):
            shift_L[i,j] = haversine(lon_d[d][i], lat_d[d][i], lon_d[d+1][j], lat_d[d+1][j])    
    
    ### pair the events from the shortest distance among them ###
    for dd in np.arange( min(len(lon_d[d]), len(lon_d[d+1])) ):
        WE_i, WE_j = np.unravel_index(np.argmin(shift_L), shift_L.shape)
        
        ### note the distance between two paris is also limited a thrshold ###
        lon_shift = abs(lon_d[d+1][WE_j] - lon_d[d][WE_i])
        lat_shift = abs(lat_d[d+1][WE_j] - lat_d[d][WE_i])
        ### correct the longitude shift due the periodic boundary ###        
        if lon_shift > 180:
            lon_shift = abs(lon_shift-360)
        if lon_shift < -180:
            lon_shift = abs(lon_shift + 360)
            
        if lon_shift<lon_thresh and lat_shift<lat_thresh:
            next_index_day[WE_i] = WE_j  ### these two events can be paired! ###
            shift_L[WE_i, :] = np.inf    ### make the distance related to these two events to infinity so that we can search the next shortest distance and avoid this pair ###
            shift_L[:, WE_j] = np.inf
        else:
            shift_L[WE_i, :] = np.inf
            shift_L[:, WE_j] = np.inf
            
    next_index.append(next_index_day)
            
#%%
########### Now we begin to track wave events #########
Blocking_lat = []; Blocking_lon = []; Blocking_date = []; Blocking_date_index = []; Blocking_lwa_max = []; Blocking_lwa_total = []; Blocking_lwa_mean = []; Blocking_lwa_label = []
Blocking_lon_wide = []; Blocking_lat_wide = []; Blocking_area = []; Blocking_label = []; Blocking_label_sum = []
Blocking_type = []; Blocking_month = []; Blocking_year = [];  Blocking_duration = []

Blocking_peaking_lat = [];  Blocking_peaking_lon = []
Blocking_peaking_date = []; Blocking_peaking_date_index=[]; Blocking_peaking_month = []; Blocking_peaking_year = []
Blocking_peaking_lwa_max = []; Blocking_peaking_lwa_total = []; Blocking_peaking_lwa_mean = []; Blocking_peaking_lwa_label = []
Blocking_peaking_label = []; Blocking_peaking_lon_wide = []; Blocking_peaking_lat_wide = []; Blocking_peaking_area = []
    
for d in np.arange(nday-1):
    ### if all wave events within this day are tracked, then go to the next day ###
    if np.all(np.isnan(lon_d[d])):
        continue  
            
    for i in np.arange(len(lon_d[d])):
        
        ### if this wave evnet is tracked, then go to the next wave event ###
        if np.isnan(lon_d[d][i]):
            continue
        
        ### Tracking starts ###
        day = 0
        track_lon = []; track_lat = []; track_lon_index = []; track_lat_index = []; 
        track_date = []; track_date_index = []; track_month = []; track_year = []
        track_lwa_max = []; track_lwa_total = []; track_lwa_mean = []; track_lwa_label = []
        track_lon_wide = []; track_area = []; track_lat_wide = []; track_label = []
        
        B_count = np.zeros((nlat_NH, nlon))
        B = np.zeros((nlat_NH, nlon))
        B_count2 = []
        
        B_count[labels_new[d+day]==i+1]+=1
        B[labels_new[d+day]==i+1]=1            
        B_count2.append(B)
        
        track_lon.append(lon_d[d+day][i]); track_lon_index.append(i)               
        track_lat.append(lat_d[d+day][i]); track_lat_index.append(i)            
        track_date.append(Date[d+day]); track_date_index.append(d+day); track_month.append(str(Date[d+day])[5:7]); track_year.append(str(Date[d+day])[0:4])
        track_lwa_max.append(lwa_max[d+day][i]); track_lwa_total.append(lwa_total[d+day][i]); track_lwa_mean.append(lwa_mean[d+day][i]); track_lwa_label.append(lwa_label[d+day][i])
        track_lon_wide.append(lon_wide[d+day][i]); track_lat_wide.append(lat_wide[d+day][i])
        track_area.append(area[d+day][i])
        track_label.append(labels_new[d+day]==i+1)

        
        next_index_pair= next_index[d+day][i] ### find the pair event at next day, it could be nan ###
        if ~np.isnan(next_index_pair):
            next_index_pair = int(next_index_pair)
            
        while ~np.isnan(next_index_pair) and abs(lon_d[d+day+1][next_index_pair]-track_lon[0]) < 1.5*18 and abs(lat_d[d+day+1][next_index_pair]-track_lat[0]) < 1.5*13.5 and track_lat[-1] > 40 and track_lon_wide[-1]>15:
        # while ~np.isnan(next_index_pair) and track_lat[-1] > 30 and track_lon_wide[-1]>15:
            ### if this event does have a pair next day, and the total displacement is within 1.5*18 lons and 1.5*13.5 lats, and it is beyond 30N or 30S, then keep tracking ###
            track_date.append(Date[d+day+1]); track_date_index.append(d+day+1); track_month.append(str(Date[d+day+1])[5:7]); track_year.append(str(Date[d+day+1])[0:4])
            B_count[labels_new[d+day+1]==next_index_pair+1]+=1
            B = np.zeros((nlat_NH, nlon))
            B[labels_new[d+day+1]==next_index_pair+1]=1            
            B_count2.append(B)
                
            track_lon.append(lon_d[d+day+1][next_index_pair])
            track_lon_index.append(next_index_pair)
            track_lat.append(lat_d[d+day+1][next_index_pair])
            track_lat_index.append(next_index_pair)
            track_lon_wide.append(lon_wide[d+day+1][next_index_pair])
            track_lat_wide.append(lat_wide[d+day+1][next_index_pair])
            track_area.append(area[d+day+1][next_index_pair])
            track_lwa_max.append(lwa_max[d+day+1][next_index_pair]); track_lwa_total.append(lwa_total[d+day+1][next_index_pair]); track_lwa_mean.append(lwa_mean[d+day+1][next_index_pair]); track_lwa_label.append(lwa_label[d+day+1][next_index_pair])
            track_label.append(labels_new[d+day+1]==next_index_pair+1)

            ### interate the day ###
            day+=1
            
            ### if this the last day, then jump out ###
            if d+day+1>nday-1:
                break
            
            ### if not, then find a next pair ###
            next_index_pair= next_index[d+day][next_index_pair]
            if ~np.isnan(next_index_pair):
                next_index_pair = int(next_index_pair)
        
            
        if day+1 >= Duration:
            Blocking_lon.append(track_lon)
            Blocking_lat.append(track_lat)
            Blocking_date.append(track_date)
            Blocking_date_index.append(track_date_index)
            Blocking_month.append(track_month)
            Blocking_year.append(track_year)
            Blocking_lwa_max.append(track_lwa_max)
            Blocking_lwa_total.append(track_lwa_total)
            Blocking_lwa_mean.append(track_lwa_mean)
            Blocking_lwa_label.append(track_lwa_label)
            Blocking_lon_wide.append(track_lon_wide)
            Blocking_lat_wide.append(track_lat_wide)
            Blocking_area.append(track_area)
            Blocking_label_sum.append(B_count)
            Blocking_label.append(track_label)
            
            B_freq += B_count
            for dd in np.arange(len(B_count2)):
                B_freq2[d+dd,:,:]+= B_count2[dd]
                
            # Find the peaking date information of this wave event #
            lwa_peak = max(track_lwa_max)
            lwa_peak_index = track_lwa_max.index(lwa_peak)
            
            Blocking_peaking_lwa_max.append(lwa_peak); Blocking_peaking_lwa_total.append(track_lwa_total[lwa_peak_index]); Blocking_peaking_lwa_mean.append(track_lwa_mean[lwa_peak_index]); Blocking_peaking_lwa_label.append(track_lwa_label[lwa_peak_index])
            Blocking_peaking_lon.append(track_lon[lwa_peak_index]); Blocking_peaking_lat.append(track_lat[lwa_peak_index])
            Blocking_peaking_date.append(track_date[lwa_peak_index]); Blocking_peaking_date_index.append(track_date_index[lwa_peak_index])
            Blocking_peaking_month.append(track_month[lwa_peak_index]); Blocking_peaking_year.append(track_year[lwa_peak_index])
            Blocking_peaking_lon_wide.append(track_lon_wide[lwa_peak_index]); Blocking_peaking_lat_wide.append(track_lat_wide[lwa_peak_index])
            Blocking_peaking_area.append(track_area[lwa_peak_index])
            Blocking_peaking_label.append(track_label[lwa_peak_index])
            Blocking_duration.append(day+1)
        
            peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==track_lon[lwa_peak_index])))
            Blocking_type.append( blocking_type_v1(LWA_Z_A[track_date_index[lwa_peak_index],:,:], LWA_Z_C[track_date_index[lwa_peak_index],:,:], track_label[lwa_peak_index], peaking_lon_index, nlat_hemi, nlon, dlon*180/np.pi) )
                    
        ### Once we successfully detected a wave event, we mark that with nan to avoid repeating ###
        for dd in np.arange(day+1):
            lon_d[d+dd][track_lon_index[dd]] = np.nan
            lat_d[d+dd][track_lat_index[dd]] = np.nan
    print(d)
          

#%%
### save the data ###

file_blocking = "/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/Blocking.nc"

### save the Blocking data to nc ###    
n_events = len(Blocking_lon)
max_dur = max(len(xx) for xx in Blocking_lon)

file = xr.Dataset(
    data_vars={
        "Blocking_lon": (("event", "time"), list2arr(Blocking_lon, n_events, max_dur)),
        "Blocking_lat": (("event", "time"), list2arr(Blocking_lat, n_events, max_dur)),
        "Blocking_date": (("event", "time"), list2arr(Blocking_date_index, n_events, max_dur)),
        "Blocking_area": (("event", "time"), list2arr(Blocking_area, n_events, max_dur)),
        "Blocking_lon_wide": (("event", "time"), list2arr(Blocking_lon_wide, n_events, max_dur)),
        "Blocking_lat_wide": (("event", "time"), list2arr(Blocking_lat_wide, n_events, max_dur)),
        "Blocking_lwa_max": (("event", "time"), list2arr(Blocking_lwa_max, n_events, max_dur)),
        "Blocking_lwa_total": (("event", "time"), list2arr(Blocking_lwa_total, n_events, max_dur)),
        "Blocking_lwa_mean": (("event", "time"), list2arr(Blocking_lwa_mean, n_events, max_dur)),
        "Blocking_month": (("event","time"), list2arr(Blocking_month, n_events, max_dur)),
        "Blocking_year": (("event","time"), list2arr(Blocking_year, n_events, max_dur)),


        "Blocking_peaking_lon": (("event"), np.array(Blocking_peaking_lon)),
        "Blocking_peaking_lat": (("event"), np.array(Blocking_peaking_lat)),
        "Blocking_peaking_lwa_max": (("event"), np.array(Blocking_peaking_lwa_max)),
        "Blocking_peaking_lwa_total": (("event"), np.array(Blocking_peaking_lwa_total)),
        "Blocking_peaking_lwa_mean": (("event"), np.array(Blocking_peaking_lwa_mean)),
        "Blocking_peaking_month": (("event"), np.array(Blocking_peaking_month)),
        "Blocking_peaking_year": (("event"), np.array(Blocking_peaking_year)),
        "Blocking_peaking_date": (("event"), np.array(Blocking_peaking_date_index)),
        "Blocking_peaking_lon_wide": (("event"), np.array(Blocking_peaking_lon_wide)),
        "Blocking_peaking_lat_wide": (("event"), np.array(Blocking_peaking_lat_wide)),
        "Blocking_peaking_area": (("event"), np.array(Blocking_peaking_area)),

        "Blocking_duration": (("event"), np.array(Blocking_duration)),
        "Blocking_type": (("event"), np.array(Blocking_type)),
        
    },
    coords={
        "event":  np.arange(n_events),
        "time": np.arange(max_dur),
        "lon": np.arange(nlon),
        "lat": np.arange(nlat),

    },

)
file.to_netcdf(file_blocking)


with open("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/Blocking_date", "wb") as fp:
    pickle.dump(Blocking_date, fp)
       
with open("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/Blocking_lon", "wb") as fp:
    pickle.dump(Blocking_lon, fp)
    
with open("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/Blocking_lat", "wb") as fp:
    pickle.dump(Blocking_lat, fp)
    
with open("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/Blocking_lon_wide", "wb") as fp:
    pickle.dump(Blocking_lon_wide, fp)
    
with open("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/Blocking_area", "wb") as fp:
    pickle.dump(Blocking_area, fp)

with open("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/Blocking_label", "wb") as fp:
    pickle.dump(Blocking_label, fp)


np.save("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/B_freq2.npy", B_freq2)
np.save("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/B_freq.npy",B_freq)





       
   
# %%
