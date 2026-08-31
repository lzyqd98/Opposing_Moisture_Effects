#%%
###### This code is to track blocking events with JRA55 dataset ######
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


### A function to calculate distance between two grid points on earth ###
from math import radians, cos, sin, asin, sqrt
 
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # earth radius
    return c * r * 1000

#%%
### Read basic variables ###
path = glob.glob(r"/scratch/bell/liu3315/JRA55/LWA/*.nc")
path.sort()
N=len(path)   #The number of nc files

file0 = Dataset(path[0],'r')
lon = file0.variables['lon'][:]
lat = file0.variables['lat'][:]
midlat = int(len(lat)/2)
lat_SH = lat[0:midlat ]
lat_NH = lat[midlat :]
lat1= 24
nlon = len(lon)
nlat = len(lat)
nlat_SH = len(lat_SH)
nlat_NH =len(lat_NH)
file0.close()

### Read the daily Z500-based LWA ###
file_path = "/scratch/bell/liu3315/JRA55/var_connect/"
ds = xr.open_dataset(file_path+"LWA_Z500.nc")
LWA_Z = ds['LWA_Z500'].values[:,midlat:,:]
time = ds['time'].values
nday = len(time)

Datestamp = pd.date_range(start="1980-01-01",end="2020-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Month = Date0['date'].dt.month 
Year = Date0['date'].dt.year
Day = Date0['date'].dt.day
Date = list(Date0['date'])
nday = len(Date)

#%%
###### Detect Blocking ######
Blocking_total_lat = []
Blocking_total_lon = []
Blocking_total_date = []
B_freq = np.zeros((nlat_NH, nlon))         ### The final blocking frequency/total number 
B_freq2 = np.zeros((nday ,nlat_NH, nlon))  ### Record whether a grid point is under a blocking for a certain day (only 1 and 0 in this 3D array)

### Threshold and Duration ###
LWA_max_lon = np.zeros((nlon*nday))
for t in np.arange(nday):    
    for lo in np.arange(nlon):
        LWA_max_lon[t*nlon+lo] = np.max(LWA_Z[t,:,lo])
Thresh = np.median(LWA_max_lon[:])
# Thresh = 60
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
    if np.any(labels_new[d,:,0]) == 0 or np.any(labels_new[d,:,-1]) == 0:   ## If there are no events at the data boundary, then we don't need to do any connection
        continue
            
    column_0 = np.zeros((nlat_NH,3))       ## We assume there are at most three wave events at column 0 (actually most of the time there is just one)
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
lat_d = []; lon_d = []; max_d = []
lon_w = []; lon_e = []; area = []
lon_wide = []
for d in np.arange(nday):
    if int(num_labels[d]-1)==0:
        lat_list=np.zeros((1));lon_list=np.zeros((1))
        lat_list[0] = np.nan; lon_list[0] = np.nan
        lat_d.append(lat_list)
        lon_d.append(lon_list)
        lon_w.append(lon_list)
        lon_e.append(lon_list)
        area.append(lon_list)
        lon_wide.append(lon_list)
        continue
    
    lat_list=np.zeros(( int(num_labels[d]-1) ));    lon_list=np.zeros(( int(num_labels[d]-1) ));   max_list=np.zeros(( int(num_labels[d]-1) ))
    lon_w_list = np.zeros(( int(num_labels[d]-1) ));lon_e_list=np.zeros(( int(num_labels[d]-1) )); area_list = np.zeros((int(num_labels[d]-1))) 
    lon_wide_list = np.zeros(( int(num_labels[d]-1) ))
    for n in np.arange(0,int(num_labels[d]-1)):
        LWA_d = np.zeros((nlat_NH, nlon))
        LWA_d[labels_new[d]==n+1]=LWA_Z[d][labels_new[d]==n+1]   ### isolate that wave event ###
        ### Get the maximum location ###
        if len(np.array(np.where( LWA_d==LWA_d.max() ))[0])>1:           
            lat_list[n] = lat_NH[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[0][0]]
            lon_list[n] = lon[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[1][0]]
            max_list[n] = LWA_d.max()
        else:
            lat_list[n] = lat_NH[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[0]]
            lon_list[n] = lon[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[1]]
            max_list[n] = LWA_d.max()
        ### Get the west east boundary longitude and area ###
        for lo in np.arange(nlon):
            if (np.any(LWA_d[:,lo])) and (not np.any(LWA_d[:,lo-1])):
                lon_w_list[n] = lon[lo]
            if (not np.any(LWA_d[:,lo])) and (np.any(LWA_d[:,lo-1])):
                lon_e_list[n] = lon[lo-1]
        ### Get the width from west to east ###
        if lon_e_list[n]-lon_w_list[n] > 0:
            lon_wide_list[n] = lon_e_list[n]-lon_w_list[n]
        else:
            lon_wide_list[n] = 360+(lon_e_list[n]-lon_w_list[n])
        ### Get the total area ###
        area_count = np.zeros((nlat_NH, nlon))
        area_count[labels_new[d]==n+1]=1
        area_list[n] = np.sum(area_count)
                    
    lat_d.append(lat_list);   lon_d.append(lon_list);    max_d.append(max_list)
    lon_w.append(lon_w_list); lon_e.append(lon_e_list);  area.append(area_list)
    lon_wide.append(lon_wide_list)

    
#%%            
########### Now we begin to track wave events #########
########### A blocking event requires at least 5 consecutive wave events ########
########### The distance between two wave events should be less than 13.5 of latitudes and 18 of longitudes #######
########### Also, the blocking event should be large enough, the width of such event is at least 15 degress width (Typical Rossby Deformation Radius) #######

Blocking_lat = []; Blocking_lon = []; Blocking_date = []
Blocking_lon_wide = []; Blocking_area = []; Blocking_label = []
for d in np.arange(nday-1):
    
    ### calculate the distance between each WE during the consective two days
    shift_L = np.zeros((len(lon_d[d]),len(lon_d[d+1]) ))
    for i in np.arange(len(lon_d[d])):
        for j in np.arange(len(lon_d[d+1])):
            shift_L[i,j] = haversine(lon_d[d][i], lat_d[d][i], lon_d[d+1][j], lat_d[d+1][j])    
    shift_L[np.isnan(shift_L)] = np.inf  ### This step is to make sure skip the WE already intentified as a block
    if np.all(np.isinf(shift_L)):
        continue
    
    for dd in np.arange( min(len(lon_d[d]), len(lon_d[d+1])) ):
        
        WE_i_ori, WE_j_ori = np.unravel_index(np.argmin(shift_L), shift_L.shape)
        WE_i, WE_j = np.unravel_index(np.argmin(shift_L), shift_L.shape)
        
        day = 0
        track_lon = []; track_lat = []; track_date = []
        track_lon_index = []; track_lat_index = []
        track_lon_wide = []; track_area = []; track_label = []
        B_count = np.zeros((nlat_NH, nlon))
        B = np.zeros((nlat_NH, nlon))
        B_count2 = []
    
        B_count[labels_new[d+day]==WE_i+1]+=1
        B[labels_new[d+day]==WE_i+1]=1            
        B_count2.append(B)
    
        track_lon.append(lon_d[d+day][WE_i])
        track_lon_index.append(WE_i)
        lon_shift = lon_d[d+day+1][WE_j]-track_lon[-1]      
        if lon_shift > 180:
            lon_shift = lon_shift-360
        if lon_shift < -180:
            lon_shift = lon_shift + 360
                
        track_lat.append(lat_d[d+day][WE_i])
        track_lat_index.append(WE_i)
        lat_shift = lat_d[d+day+1][WE_j]-track_lat[-1]
       
        total_shift = abs(lon_shift) + abs(lat_shift)


        track_date.append(Date[d+day])
        track_lon_wide.append(lon_wide[d+day][WE_i])
        track_area.append(area[d+day][WE_i])
        track_label.append(labels_new[d+day]==WE_i+1)
                
        while abs(lon_shift) < 18 and abs(lat_shift) < 13.5 and total_shift < 31.5 and track_lat[-1]>30:
            track_date.append(Date[d+day+1])

            B_count[labels_new[d+day+1]==WE_j+1]+=1
            B = np.zeros((nlat_NH, nlon))
            B[labels_new[d+day+1]==WE_j+1]=1            
            B_count2.append(B)
                
            track_lon.append(lon_d[d+day+1][WE_j])
            track_lon_index.append(WE_j)
            track_lat.append(lat_d[d+day+1][WE_j])
            track_lat_index.append(WE_j)
            track_lon_wide.append(lon_wide[d+day+1][WE_j])
            track_area.append(area[d+day+1][WE_j])
            track_label.append(labels_new[d+day+1]==WE_j+1)
                    
            day+=1 ### now we interate to the next day
            if d+day+1>nday-1:
                break
            
            shift_L_ite = np.zeros((len(lon_d[d+day]),len(lon_d[d+day+1]) ))
            for i in np.arange(len(lon_d[d+day])):
                for j in np.arange(len(lon_d[d+day+1])):
                    shift_L_ite[i,j] = haversine(lon_d[d+day][i], lat_d[d+day][i], lon_d[d+day+1][j], lat_d[d+day+1][j])
            shift_L_ite[np.isnan(shift_L_ite)] = np.inf  ### This step is to make sure skip the WE already intentified as a block
            if np.all(np.isinf(shift_L_ite)):
                break
            
            n=0
            for k in np.arange(   min(len(lon_d[d+day]), len(lon_d[d+day+1])) ):
                n+=1
                WE_ik, WE_jk = np.unravel_index(np.argmin(shift_L_ite), shift_L_ite.shape)
                if WE_ik == WE_j:
                    WE_i = WE_j
                    WE_j = WE_jk
                    break
                else:
                    ### give the original row and column very lagre values so that they cannot be the minimum anymore
                    shift_L_ite[WE_ik, :] = np.inf
                    shift_L_ite[:, WE_jk] = np.inf
            
            ### if no pair for this point, then this search  stops ###
            if n == min( len(lon_d[d+day]), len(lon_d[d+day+1])  ):
                break
            
            
            lon_shift = lon_d[d+day+1][WE_j]-track_lon[-1]               
            if lon_shift > 180:
                lon_shift = lon_shift-360
            if lon_shift < -180:
                lon_shift = lon_shift + 360
                
            lat_shift = lat_d[d+day+1][WE_j]-track_lat[-1]
            total_shift = abs(lon_shift) + abs(lat_shift)
        
        
        
    
        ### For this tracked wave event evolution, we want to how many days the longitude width is longer than 15 degree longitude ###           
        n_large_wave = 0
        for j in np.arange(len(track_lon_wide)):
            if track_lon_wide[j]>15:
                n_large_wave+=1
                
        ### Decide if it is a blocking event: it should persist longer than at least 5 days ###
        ### Also, the area/width should be large enough ### 
        if day+1 >= Duration and n_large_wave>=5:
            Blocking_lon.append(track_lon)
            Blocking_lat.append(track_lat)
            Blocking_date.append(track_date)
            Blocking_lon_wide.append(track_lon_wide)
            Blocking_area.append(track_area)
            Blocking_label.append(track_label)
            B_freq += B_count
            for dd in np.arange(len(B_count2)):
                B_freq2[d+dd,:,:]+= B_count2[dd]
            
            ### Once we successfully detected a blocking, we make the orginal distance very large to avoid repeat ###
            shift_L[WE_i_ori, :] = np.inf
            shift_L[:, WE_j_ori] = np.inf
            
            for dd in np.arange(day+1):
                lon_d[d+dd][track_lon_index[dd]] = np.nan
                lat_d[d+dd][track_lat_index[dd]] = np.nan
            
    print(d)


   
          

#%%

with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_date", "wb") as fp:
    pickle.dump(Blocking_date, fp)
       
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_lon", "wb") as fp:
    pickle.dump(Blocking_lon, fp)
    
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_lat", "wb") as fp:
    pickle.dump(Blocking_lat, fp)
    
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_lon_wide", "wb") as fp:
    pickle.dump(Blocking_lon_wide, fp)
    
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_area", "wb") as fp:
    pickle.dump(Blocking_area, fp)
    
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_label", "wb") as fp:
    pickle.dump(Blocking_label, fp)
  

np.save("/scratch/bell/liu3315/JRA55/blocks/B_freq.npy", B_freq2)
np.save("/scratch/bell/liu3315/JRA55/blocks/B_freq_d.npy",B_freq)




# %%
