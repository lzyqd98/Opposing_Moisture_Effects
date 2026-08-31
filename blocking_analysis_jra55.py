#%%
###### This code is to study the blocking diversity (separation of different types of blocks) ######
from math import pi
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
import scipy.stats as stats
import cartopy


#%%
### read basic data ###
path = glob.glob(r"/scratch/bell/liu3315/JRA55/lwa/*.nc")
path.sort()
N=len(path)   

file0 = Dataset(path[0],'r')
lon = file0.variables['lon'][:]
lat = file0.variables['lat'][:]
zlev = file0.variables['lev_z'][:]
midlat = int(len(lat)/2)
lat_SH = lat[0:midlat]
lat_NH = lat[midlat:]
nlon = len(lon)
nlat = len(lat)
nzlev = len(zlev)
nlat_SH = len(lat_SH)
nlat_NH =len(lat_NH)
dlat=lat[1]-lat[0]
dlon=lon[1]-lon[0]
file0.close()

### read blocking data ###
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_date", "rb") as fp:
    Blocking_date = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_lat", "rb") as fp:
    Blocking_lat = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_lon", "rb") as fp:
    Blocking_lon = pickle.load(fp)    
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_lon_wide", "rb") as fp:
    Blocking_lon_wide = pickle.load(fp) 
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_area", "rb") as fp:
    Blocking_area = pickle.load(fp) 
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_label", "rb") as fp:
    Blocking_label = pickle.load(fp)  
    
B_freq = np.load("/scratch/bell/liu3315/JRA55/blocks/B_freq.npy")

### Time Management ###
Datestamp = pd.date_range(start="1980-01-01",end="2020-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Month = Date0['date'].dt.month 
Year = Date0['date'].dt.year
Day = Date0['date'].dt.day
Date = list(Date0['date'])
nday = len(Date)

#%%
### get the blocking peaking date and location and wave activity ###
Blocking_peaking_date = []
Blocking_peaking_date_index = []
Blocking_peaking_lon = []
Blocking_peaking_lat = []
Blocking_peaking_LWA = []
Blocking_duration =[]
Blocking_velocity = [] 
Blocking_peaking_lon_wide = []
Blocking_peaking_area = []
for n in np.arange(len(Blocking_date)):
    start = Date.index(Blocking_date[n][0])
    end = Date.index(Blocking_date[n][-1])
    duration = len(Blocking_date[n])
    LWA_event_max = np.zeros((duration))

    for d in np.arange(duration):
        index = start+d
        lo = np.squeeze(np.array(np.where( lon[:]==Blocking_lon[n][d])))
        la = np.squeeze(np.array(np.where( lat[:]==Blocking_lat[n][d])))    
        file = Dataset(path[index],'r')
        LWA_event_max[d]  = file.variables['LWA_Z500'][la,lo]
        file.close()
        
    Blocking_peaking_date_index=int(np.squeeze(np.array(np.where( LWA_event_max==np.max(LWA_event_max) ))))
    Blocking_peaking_LWA.append(np.max(LWA_event_max))
    Blocking_peaking_date.append(Blocking_date[n][Blocking_peaking_date_index])
    Blocking_peaking_lon.append(Blocking_lon[n][Blocking_peaking_date_index])
    Blocking_peaking_lat.append(Blocking_lat[n][Blocking_peaking_date_index])
    Blocking_peaking_lon_wide.append(Blocking_lon_wide[n][Blocking_peaking_date_index])
    Blocking_peaking_area.append(Blocking_area[n][Blocking_peaking_date_index])
    Blocking_duration.append(duration)
    Blocking_velocity.append( haversine(Blocking_lon[n][0], Blocking_lat[n][0], Blocking_lon[n][-1], Blocking_lat[n][-1])/(duration*24*60*60) )
    
    print(n)


with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_date_index", "wb") as fp:
    pickle.dump(Blocking_peaking_date_index, fp)       
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_date", "wb") as fp:
    pickle.dump(Blocking_peaking_date, fp)    
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_lat", "wb") as fp:
    pickle.dump(Blocking_peaking_lat, fp)    
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_lon", "wb") as fp:
    pickle.dump(Blocking_peaking_lon, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_lon_wide", "wb") as fp:
    pickle.dump(Blocking_peaking_lon_wide, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_area", "wb") as fp:
    pickle.dump(Blocking_peaking_area, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_LWA", "wb") as fp:
    pickle.dump(Blocking_peaking_LWA, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_velocity", "wb") as fp:
    pickle.dump(Blocking_velocity, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_duration", "wb") as fp:
    pickle.dump(Blocking_duration, fp)

### Directly read data ###
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_date", "rb") as fp:
#     Blocking_peaking_date = pickle.load(fp)
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_lat", "rb") as fp:
#     Blocking_peaking_lat = pickle.load(fp)
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_lon", "rb") as fp:
#     Blocking_peaking_lon = pickle.load(fp)    
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_lon_wide", "rb") as fp:
#     Blocking_peaking_lon_wide = pickle.load(fp)    
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_area", "rb") as fp:
#     Blocking_peaking_area = pickle.load(fp) 
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_peaking_LWA", "rb") as fp:
#     Blocking_peaking_LWA = pickle.load(fp) 
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_velocity", "rb") as fp:
#     Blocking_velocity = pickle.load(fp)
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_duration", "rb") as fp:
#     Blocking_duration = pickle.load(fp)


#%%
##### Now we separate 3 types of blocks (ridge, trough, dipole) ###
#### Method: Use the peaking date, calculate the total LWA_AC and LWA_C of the block region ####
Blocking_ridge_date = [];  Blocking_ridge_lon = []; Blocking_ridge_lat=[];  Blocking_ridge_peaking_date = [];   Blocking_ridge_peaking_lon = []; Blocking_ridge_peaking_lat=[];  Blocking_ridge_duration = [];  Blocking_ridge_velocity = [];    Blocking_ridge_area = [];   Blocking_ridge_peaking_LWA = [];   Blocking_ridge_A = []; Blocking_ridge_C = [];   Blocking_ridge_label =[]
Blocking_trough_date = []; Blocking_trough_lon =[]; Blocking_trough_lat=[]; Blocking_trough_peaking_date = [];  Blocking_trough_peaking_lon =[]; Blocking_trough_peaking_lat=[]; Blocking_trough_duration = []; Blocking_trough_velocity = [];   Blocking_trough_area = [];  Blocking_trough_peaking_LWA = [];  Blocking_trough_A = []; Blocking_trough_C = []; Blocking_trough_label =[]
Blocking_dipole_date = []; Blocking_dipole_lon =[]; Blocking_dipole_lat=[]; Blocking_dipole_peaking_date = [];  Blocking_dipole_peaking_lon =[]; Blocking_dipole_peaking_lat=[]; Blocking_dipole_duration= []; Blocking_dipole_velocity =[];    Blocking_dipole_area = [];   Blocking_dipole_peaking_LWA = [];  Blocking_dipole_A = []; Blocking_dipole_C = []; Blocking_dipole_label = []
lat_range=int(int((90-np.max(Blocking_peaking_lat))/dlat)*2+1)
lon_range=int(30/dlon)+1


for n in np.arange(len(Blocking_lon)):

    LWA_AC_sum = 0
    LWA_C_sum = 0
    Blocking_A = []
    Blocking_C = []
        
    ### peaking date information ###
    peaking_date_index = Date.index(Blocking_peaking_date[n])
    peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_peaking_lon[n])))
    peaking_lat_index = np.squeeze(np.array(np.where( lat[:]==Blocking_peaking_lat[n]))) 
    
    t = np.squeeze(np.where(np.array(Blocking_date[n]) == np.array(Blocking_peaking_date[n] )))
    
    file = Dataset(path[peaking_date_index],'r')
    LWA_max  = file.variables['LWA_Z500'][peaking_lat_index,peaking_lon_index]
    LWA_AC  = file.variables['LWA_Z500_A'][midlat:,:]
    LWA_C  = file.variables['LWA_Z500_C'][midlat:,:]
    file.close()
        
    LWA_AC = np.roll(LWA_AC, int(nlon/2)-peaking_lon_index, axis=1)
    LWA_C = np.roll(LWA_C,   int(nlon/2)-peaking_lon_index, axis=1)
    WE = np.roll(Blocking_label[n][t], int(nlon/2)-peaking_lon_index, axis=1)
    lon_roll = np.roll(lon,   int(nlon/2)-peaking_lon_index)
    
    LWA_AC = LWA_AC[  :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    LWA_C = LWA_C[    :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    WE = WE[   :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    
    LWA_AC_d = np.zeros((nlat_NH, lon_range))
    LWA_C_d = np.zeros((nlat_NH, lon_range))
    LWA_AC_d[WE == True]  = LWA_AC[WE== True]
    LWA_C_d[WE == True]  =  LWA_C[WE == True]
    
    
    LWA_AC_sum += LWA_AC_d.sum()
    LWA_C_sum += LWA_C_d.sum()
    Blocking_A.append(LWA_AC_d.sum())
    Blocking_C.append(LWA_C_d.sum())

    if LWA_AC_sum > 10 * LWA_C_sum :
        Blocking_ridge_date.append(Blocking_date[n]);                 Blocking_ridge_lon.append(Blocking_lon[n]);                  Blocking_ridge_lat.append(Blocking_lat[n])
        Blocking_ridge_peaking_date.append(Blocking_peaking_date[n]); Blocking_ridge_peaking_lon.append(Blocking_peaking_lon[n]);  Blocking_ridge_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_ridge_peaking_LWA.append(LWA_max)
        Blocking_ridge_duration.append(len(Blocking_date[n]));        Blocking_ridge_velocity.append(Blocking_velocity[n]);        Blocking_ridge_area.append(Blocking_peaking_area[n]); Blocking_ridge_label.append(Blocking_label[n])
        Blocking_ridge_A.append(Blocking_A);                          Blocking_ridge_C.append(Blocking_C)
    elif LWA_C_sum > 2 * LWA_AC_sum:
        Blocking_trough_date.append(Blocking_date[n]);                 Blocking_trough_lon.append(Blocking_lon[n]);                 Blocking_trough_lat.append(Blocking_lat[n])
        Blocking_trough_peaking_date.append(Blocking_peaking_date[n]); Blocking_trough_peaking_lon.append(Blocking_peaking_lon[n]); Blocking_trough_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_trough_peaking_LWA.append(LWA_max)
        Blocking_trough_duration.append(len(Blocking_date[n]));        Blocking_trough_velocity.append(Blocking_velocity[n]);       Blocking_trough_area.append(Blocking_peaking_area[n]); Blocking_trough_label.append(Blocking_label[n])
        Blocking_trough_A.append(Blocking_A);                          Blocking_trough_C.append(Blocking_C)
    else:
        Blocking_dipole_date.append(Blocking_date[n]);                 Blocking_dipole_lon.append(Blocking_lon[n]);                 Blocking_dipole_lat.append(Blocking_lat[n])
        Blocking_dipole_peaking_date.append(Blocking_peaking_date[n]); Blocking_dipole_peaking_lon.append(Blocking_peaking_lon[n]); Blocking_dipole_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_dipole_peaking_LWA.append(LWA_max)
        Blocking_dipole_duration.append(len(Blocking_date[n]));        Blocking_dipole_velocity.append(Blocking_velocity[n]);       Blocking_dipole_area.append(Blocking_peaking_area[n]); Blocking_dipole_label.append(Blocking_label[n])
        Blocking_dipole_A.append(Blocking_A);                          Blocking_dipole_C.append(Blocking_C)
    print(n)


Blocking_diversity_date= [];   Blocking_diversity_lon= []; Blocking_diversity_lat= []; Blocking_diversity_date= []; Blocking_diversity_peaking_date= []; Blocking_diversity_peaking_lon= [];  Blocking_diversity_peaking_lat=[]; Blocking_diversity_peaking_LWA=[]; Blocking_diversity_duration=[]; Blocking_diversity_area=[]; Blocking_diversity_velocity=[]; Blocking_diversity_A = []; Blocking_diversity_C = []; Blocking_diversity_label = []
Blocking_diversity_date.append(Blocking_ridge_date);   Blocking_diversity_lon.append(Blocking_ridge_lon); Blocking_diversity_lat.append(Blocking_ridge_lat); Blocking_diversity_peaking_date.append(Blocking_ridge_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_ridge_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_ridge_peaking_lon); Blocking_diversity_peaking_LWA.append(Blocking_ridge_peaking_LWA); Blocking_diversity_velocity.append(Blocking_ridge_velocity); Blocking_diversity_duration.append(Blocking_ridge_duration); Blocking_diversity_area.append(Blocking_ridge_area);  Blocking_diversity_A.append(Blocking_ridge_A) ;         Blocking_diversity_C.append(Blocking_ridge_C)   ;  Blocking_diversity_label.append(Blocking_ridge_label)
Blocking_diversity_date.append(Blocking_trough_date);  Blocking_diversity_lon.append(Blocking_trough_lon); Blocking_diversity_lat.append(Blocking_trough_lat); Blocking_diversity_peaking_date.append(Blocking_trough_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_trough_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_trough_peaking_lon);Blocking_diversity_peaking_LWA.append(Blocking_trough_peaking_LWA); Blocking_diversity_velocity.append(Blocking_trough_velocity); Blocking_diversity_duration.append(Blocking_trough_duration); Blocking_diversity_area.append(Blocking_trough_area);  Blocking_diversity_A.append(Blocking_trough_A) ; Blocking_diversity_C.append(Blocking_trough_C) ;  Blocking_diversity_label.append(Blocking_trough_label)    
Blocking_diversity_date.append(Blocking_dipole_date);  Blocking_diversity_lon.append(Blocking_dipole_lon); Blocking_diversity_lat.append(Blocking_dipole_lat); Blocking_diversity_peaking_date.append(Blocking_dipole_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_dipole_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_dipole_peaking_lon); Blocking_diversity_peaking_LWA.append(Blocking_dipole_peaking_LWA); Blocking_diversity_velocity.append(Blocking_dipole_velocity); Blocking_diversity_duration.append(Blocking_dipole_duration); Blocking_diversity_area.append(Blocking_dipole_area); Blocking_diversity_A.append(Blocking_dipole_A) ; Blocking_diversity_C.append(Blocking_dipole_C) ;  Blocking_diversity_label.append(Blocking_dipole_label)

with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_date", "wb") as fp:
    pickle.dump(Blocking_diversity_date, fp)       
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_lon", "wb") as fp:
    pickle.dump(Blocking_diversity_lon, fp)    
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_lat", "wb") as fp:
    pickle.dump(Blocking_diversity_lat, fp)    
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_peaking_date", "wb") as fp:
    pickle.dump(Blocking_diversity_peaking_date, fp)     
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_peaking_lon", "wb") as fp:
    pickle.dump(Blocking_diversity_peaking_lon, fp)      
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_peaking_lat", "wb") as fp:
    pickle.dump(Blocking_diversity_peaking_lat, fp)  
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_peaking_LWA", "wb") as fp:
    pickle.dump(Blocking_diversity_peaking_LWA, fp) 
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_duration", "wb") as fp:
    pickle.dump(Blocking_diversity_duration, fp) 
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_velocity", "wb") as fp:
    pickle.dump(Blocking_diversity_velocity, fp) 
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_area", "wb") as fp:
    pickle.dump(Blocking_diversity_area, fp)     
with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_label", "wb") as fp:
    pickle.dump(Blocking_diversity_label, fp)   


### Directly read data ###
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_date", "rb") as fp:
#     Blocking_diversity_date = pickle.load(fp)      
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_lon", "rb") as fp:
#     Blocking_diversity_lon = pickle.load(fp)    
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_lat", "rb") as fp:
#     Blocking_diversity_lat = pickle.load(fp)    
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_peaking_date", "rb") as fp:
#     Blocking_diversity_peaking_date = pickle.load(fp)     
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_peaking_lon", "rb") as fp:
#     Blocking_diversity_peaking_lon = pickle.load(fp)      
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_peaking_lat", "rb") as fp:
#     Blocking_diversity_peaking_lat = pickle.load(fp)  
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_peaking_LWA", "rb") as fp:
#     Blocking_diversity_peaking_LWA = pickle.load(fp) 
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_duration", "rb") as fp:
#     Blocking_diversity_duration = pickle.load(fp) 
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_velocity", "rb") as fp:
#     Blocking_diversity_velocity = pickle.load(fp) 
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_area", "rb") as fp:
#     Blocking_diversity_area = pickle.load(fp)     
# with open("/scratch/bell/liu3315/JRA55/blocks/Blocking_diversity_label", "rb") as fp:
#     Blocking_diversity_label = pickle.load(fp)  


#%%
### Horizontal composite ###
LWA_Blocking_diversity_com = []
dAdt_Blocking_diversity_com = []
dAdt_lrghr_Blocking_diversity_com = []
dAdt_A_lrghr_Blocking_diversity_com = []
dAdt_C_lrghr_Blocking_diversity_com = []
dAdt_cnvhr_Blocking_diversity_com = []
dAdt_A_cnvhr_Blocking_diversity_com = []
dAdt_C_cnvhr_Blocking_diversity_com = []
dTdt_Blocking_diversity_com = []
dTdt_lrghr_Blocking_diversity_com = []
dTdt_cnvhr_Blocking_diversity_com = []
Z_Blocking_diversity_com = []
for i in np.array([0, 1, 2]):
    lat_range=int(int((90-np.max(Blocking_diversity_peaking_lat[i]))/dlat)*2+1); lon_range=int(60/dlon)+1
    LWA_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dAdt_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dAdt_lrghr_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dAdt_A_lrghr_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dAdt_C_lrghr_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dAdt_cnvhr_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dAdt_A_cnvhr_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dAdt_C_cnvhr_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    
    dTdt_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dTdt_lrghr_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dTdt_cnvhr_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    Z_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))

    for n in np.arange(len(Blocking_diversity_peaking_date[i])):
        
        LWA_d = np.zeros((nlat,nlon)); dTdt_d = np.zeros((nlat,nlon)); dTdt_lrghr_d=np.zeros((nlat,nlon)); dTdt_cnvhr_d=np.zeros((nlat,nlon)); Z_d = np.zeros((nlat,nlon))
        dAdt_d = np.zeros((nlat,nlon)); dAdt_lrghr_d= np.zeros((nlat,nlon)); dAdt_cnvhr_d= np.zeros((nlat,nlon))
        dAdt_A_lrghr_d = np.zeros((nlat,nlon)); dAdt_C_lrghr_d = np.zeros((nlat,nlon))
        dAdt_A_cnvhr_d = np.zeros((nlat,nlon)); dAdt_C_cnvhr_d = np.zeros((nlat,nlon))
        
        ### peaking date information ###
        peaking_date_index = Date.index(Blocking_diversity_peaking_date[i][n])
        peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_diversity_peaking_lon[i][n])))
        peaking_lat_index = np.squeeze(np.array(np.where( lat[:]==Blocking_diversity_peaking_lat[i][n])))
        
        file = Dataset(path[peaking_date_index],'r')

        LWA_d[:,:] = file.variables['LWA_column'][:,:]
        dAdt_d[:,:] = file.variables['dAdt_moist_column'][:,:]
        dAdt_lrghr_d[:,:] = file.variables['dAdt_lrghr_column'][:,:]
        dAdt_A_lrghr_d[:,:] = file.variables['dAdt_lrghr_A_column'][:,:]
        dAdt_C_lrghr_d[:,:] = file.variables['dAdt_lrghr_C_column'][:,:]
        dAdt_cnvhr_d[:,:] = file.variables['dAdt_cnvhr_column'][:,:]
        dAdt_A_cnvhr_d[:,:] = file.variables['dAdt_cnvhr_A_column'][:,:]
        dAdt_C_cnvhr_d[:,:] = file.variables['dAdt_cnvhr_C_column'][:,:]
        dTdt_d[:,:] = file.variables['dTdt_moist_column'][:,:] #5 875hPa, 6 850hPa, 14 600hPa, 16 500hPa
        dTdt_lrghr_d[:,:] = file.variables['dTdt_lrghr_column'][:,:]
        dTdt_cnvhr_d[:,:] = file.variables['dTdt_cnvhr_column'][:,:]
        Z_d[:,:] = file.variables['Z500'][:,:]
        
        ### shift the array to make the conpoiste
        LWA_d = np.roll(LWA_d, int(nlon/2)-peaking_lon_index, axis=1)
        dTdt_d = np.roll(dTdt_d, int(nlon/2)-peaking_lon_index, axis=1)
        dTdt_lrghr_d = np.roll(dTdt_lrghr_d, int(nlon/2)-peaking_lon_index, axis=1)
        dTdt_cnvhr_d = np.roll(dTdt_cnvhr_d, int(nlon/2)-peaking_lon_index, axis=1)
        dAdt_d = np.roll(dAdt_d, int(nlon/2)-peaking_lon_index, axis=1)
        dAdt_lrghr_d = np.roll(dAdt_lrghr_d, int(nlon/2)-peaking_lon_index, axis=1)
        dAdt_A_lrghr_d = np.roll(dAdt_A_lrghr_d, int(nlon/2)-peaking_lon_index, axis=1)
        dAdt_C_lrghr_d = np.roll(dAdt_C_lrghr_d, int(nlon/2)-peaking_lon_index, axis=1)
        dAdt_cnvhr_d = np.roll(dAdt_cnvhr_d, int(nlon/2)-peaking_lon_index, axis=1)
        dAdt_A_cnvhr_d = np.roll(dAdt_A_cnvhr_d, int(nlon/2)-peaking_lon_index, axis=1)
        dAdt_C_cnvhr_d = np.roll(dAdt_C_cnvhr_d, int(nlon/2)-peaking_lon_index, axis=1)
        Z_d = np.roll(Z_d, int(nlon/2)-peaking_lon_index, axis=1)
        
        LWA_Blocking_diversity[n,:,:] = LWA_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,    int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dTdt_Blocking_diversity[n,:,:] = dTdt_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dTdt_lrghr_Blocking_diversity[n,:,:] = dTdt_lrghr_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dTdt_cnvhr_Blocking_diversity[n,:,:] = dTdt_cnvhr_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dAdt_Blocking_diversity[n,:,:] = dAdt_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dAdt_lrghr_Blocking_diversity[n,:,:] = dAdt_lrghr_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dAdt_A_lrghr_Blocking_diversity[n,:,:] = dAdt_A_lrghr_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dAdt_C_lrghr_Blocking_diversity[n,:,:] = dAdt_C_lrghr_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dAdt_cnvhr_Blocking_diversity[n,:,:] = dAdt_cnvhr_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dAdt_A_cnvhr_Blocking_diversity[n,:,:] = dAdt_A_cnvhr_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dAdt_C_cnvhr_Blocking_diversity[n,:,:] = dAdt_C_cnvhr_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        Z_Blocking_diversity[n,:,:] = Z_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,        int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        print(n)
        
    LWA_Blocking_diversity_m = LWA_Blocking_diversity.mean(axis=0)
    dAdt_Blocking_diversity_m = np.nanmean(dAdt_Blocking_diversity, axis=0)
    dAdt_lrghr_Blocking_diversity_m = np.nanmean(dAdt_lrghr_Blocking_diversity, axis=0)
    dAdt_A_lrghr_Blocking_diversity_m = np.nanmean(dAdt_A_lrghr_Blocking_diversity, axis=0)
    dAdt_C_lrghr_Blocking_diversity_m = np.nanmean(dAdt_C_lrghr_Blocking_diversity, axis=0)
    dAdt_A_cnvhr_Blocking_diversity_m = np.nanmean(dAdt_A_cnvhr_Blocking_diversity, axis=0)
    dAdt_C_cnvhr_Blocking_diversity_m = np.nanmean(dAdt_C_cnvhr_Blocking_diversity, axis=0)
    dAdt_cnvhr_Blocking_diversity_m = np.nanmean(dAdt_cnvhr_Blocking_diversity, axis=0)
    dTdt_Blocking_diversity_m = np.nanmean(dTdt_Blocking_diversity, axis=0)
    dTdt_lrghr_Blocking_diversity_m = np.nanmean(dTdt_lrghr_Blocking_diversity, axis=0)
    dTdt_cnvhr_Blocking_diversity_m = np.nanmean(dTdt_cnvhr_Blocking_diversity, axis=0)
    Z_Blocking_diversity_m = Z_Blocking_diversity.mean(axis=0)


    LWA_Blocking_diversity_com.append(LWA_Blocking_diversity_m)
    dAdt_Blocking_diversity_com.append(dAdt_Blocking_diversity_m)
    dAdt_lrghr_Blocking_diversity_com.append(dAdt_lrghr_Blocking_diversity_m)
    dAdt_A_lrghr_Blocking_diversity_com.append(dAdt_A_lrghr_Blocking_diversity_m)
    dAdt_C_lrghr_Blocking_diversity_com.append(dAdt_C_lrghr_Blocking_diversity_m)
    dAdt_A_cnvhr_Blocking_diversity_com.append(dAdt_A_cnvhr_Blocking_diversity_m)
    dAdt_C_cnvhr_Blocking_diversity_com.append(dAdt_C_cnvhr_Blocking_diversity_m)
    dAdt_cnvhr_Blocking_diversity_com.append(dAdt_cnvhr_Blocking_diversity_m)
    dTdt_Blocking_diversity_com.append(dTdt_Blocking_diversity_m)
    dTdt_lrghr_Blocking_diversity_com.append(dTdt_lrghr_Blocking_diversity_m)
    dTdt_cnvhr_Blocking_diversity_com.append(dTdt_cnvhr_Blocking_diversity_m)
    Z_Blocking_diversity_com.append(Z_Blocking_diversity_m)

with open("/scratch/bell/liu3315/JRA55/blocks/LWA_Blocking_diversity_com", "wb") as fp:
    pickle.dump(LWA_Blocking_diversity_com, fp) 
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dAdt_Blocking_diversity_com, fp) 
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_lrghr_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dAdt_lrghr_Blocking_diversity_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_cnvhr_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dAdt_cnvhr_Blocking_diversity_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dTdt_Blocking_diversity_com, fp) 
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_lrghr_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dTdt_lrghr_Blocking_diversity_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_cnvhr_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dTdt_cnvhr_Blocking_diversity_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/Z_Blocking_diversity_com", "wb") as fp:
    pickle.dump(Z_Blocking_diversity_com, fp) 
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_A_lrghr_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dAdt_A_lrghr_Blocking_diversity_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_C_lrghr_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dAdt_C_lrghr_Blocking_diversity_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_A_cnvhr_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dAdt_A_cnvhr_Blocking_diversity_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_C_cnvhr_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dAdt_C_cnvhr_Blocking_diversity_com, fp)

######################################################################################################################################
#%%
###### Vertical composite #######
nzlev=47
lat_range=int(10/dlat)+1
lon_range=int(60/dlon)+1

dTdt_diversity_cros_com=[]
dTdt_lrghr_diversity_cros_com=[]
dTdt_cnvhr_diversity_cros_com=[]
dAdt_diversity_cros_com=[]
dAdt_lrghr_diversity_cros_com=[]
dAdt_A_lrghr_diversity_cros_com=[]
dAdt_C_lrghr_diversity_cros_com=[]
dAdt_A_cnvhr_diversity_cros_com=[]
dAdt_C_cnvhr_diversity_cros_com=[]
dAdt_cnvhr_diversity_cros_com=[]
LWA_diversity_cros_com=[]

for i in np.array([0, 2]):
    dAdt_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    dAdt_lrghr_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    dAdt_A_lrghr_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    dAdt_C_lrghr_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    dAdt_A_cnvhr_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    dAdt_C_cnvhr_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    dAdt_cnvhr_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon)) 
    LWA_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    dTdt_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon)) 
    dTdt_lrghr_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    dTdt_cnvhr_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    
    for n in np.arange(len(Blocking_diversity_peaking_date[i])):
        ### peaking date information ###
        peaking_date_index = Date.index(Blocking_diversity_peaking_date[i][n])
        peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_diversity_peaking_lon[i][n])))
        peaking_lat_index = np.squeeze(np.array(np.where( lat[:]==Blocking_diversity_peaking_lat[i][n])))
        
        file = Dataset(path[peaking_date_index],'r')
        LWA1 = file.variables['LWA'][:,:,:] 
        dAdt1 = file.variables['dAdt_moist'][:,:,:] 
        dAdt_lrghr1 = file.variables['dAdt_lrghr'][:,:,:]
        dAdt_A_lrghr1 = file.variables['dAdt_lrghr_A'][:,:,:]
        dAdt_C_lrghr1 = file.variables['dAdt_lrghr_C'][:,:,:]
        dAdt_A_cnvhr1 = file.variables['dAdt_cnvhr_A'][:,:,:]
        dAdt_C_cnvhr1 = file.variables['dAdt_cnvhr_C'][:,:,:]
        dAdt_cnvhr1 = file.variables['dAdt_cnvhr'][:,:,:]
        dTdt1 = file.variables['dTdt_moist'][:,:,:] 
        dTdt_lrghr1 = file.variables['dTdt_lrghr'][:,:,:]
        dTdt_cnvhr1 = file.variables['dTdt_cnvhr'][:,:,:]
        file.close()
        
        
        ### shift the array to make the conpoiste ###
        dAdt2 = np.roll(dAdt1, int(nlon/2)-peaking_lon_index, axis=2)
        dAdt_lrghr2 = np.roll(dAdt_lrghr1, int(nlon/2)-peaking_lon_index, axis=2)
        dAdt_A_lrghr2 = np.roll(dAdt_A_lrghr1, int(nlon/2)-peaking_lon_index, axis=2)
        dAdt_C_lrghr2 = np.roll(dAdt_C_lrghr1, int(nlon/2)-peaking_lon_index, axis=2)
        dAdt_A_cnvhr2 = np.roll(dAdt_A_cnvhr1, int(nlon/2)-peaking_lon_index, axis=2)
        dAdt_C_cnvhr2 = np.roll(dAdt_C_cnvhr1, int(nlon/2)-peaking_lon_index, axis=2)
        dAdt_cnvhr2 = np.roll(dAdt_cnvhr1, int(nlon/2)-peaking_lon_index, axis=2)
        dTdt2 = np.roll(dTdt1, int(nlon/2)-peaking_lon_index, axis=2)
        dTdt_lrghr2 = np.roll(dTdt_lrghr1, int(nlon/2)-peaking_lon_index, axis=2)
        dTdt_cnvhr2 = np.roll(dTdt_cnvhr1, int(nlon/2)-peaking_lon_index, axis=2)
        LWA2 = np.roll(LWA1, int(nlon/2)-peaking_lon_index, axis=2)
        lon1 = np.roll(lon, int(nlon/2)-peaking_lon_index)
        
        ### get a +- 30 lon wide domain, average the +-10 latitudes, now it's a Z_lon cross section ###
        dAdt2 = np.mean(dAdt2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        dAdt_lrghr2 = np.mean(dAdt_lrghr2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        dAdt_A_lrghr2 = np.mean(dAdt_A_lrghr2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        dAdt_C_lrghr2 = np.mean(dAdt_C_lrghr2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        dAdt_A_cnvhr2 = np.mean(dAdt_A_cnvhr2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        dAdt_C_cnvhr2 = np.mean(dAdt_C_cnvhr2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        dAdt_cnvhr2 = np.mean(dAdt_cnvhr2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        dTdt2 = np.mean(dTdt2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        dTdt_lrghr2 = np.mean(dTdt_lrghr2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        dTdt_cnvhr2 = np.mean(dTdt_cnvhr2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        LWA2 = np.mean(LWA2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        lat2 =  lat[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1]
        lon2 = lon1[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        
        dAdt_diversity_cros[n,:,:] = dAdt2
        dAdt_lrghr_diversity_cros[n,:,:] = dAdt_lrghr2
        dAdt_A_lrghr_diversity_cros[n,:,:] = dAdt_A_lrghr2
        dAdt_C_lrghr_diversity_cros[n,:,:] = dAdt_C_lrghr2
        dAdt_A_cnvhr_diversity_cros[n,:,:] = dAdt_A_cnvhr2
        dAdt_C_cnvhr_diversity_cros[n,:,:] = dAdt_C_cnvhr2
        dAdt_cnvhr_diversity_cros[n,:,:] = dAdt_cnvhr2
        dTdt_lrghr_diversity_cros[n,:,:] = dTdt_lrghr2
        dTdt_cnvhr_diversity_cros[n,:,:] = dTdt_cnvhr2
        dTdt_diversity_cros[n,:,:] = dTdt2
        LWA_diversity_cros[n,:,:] = LWA2
        print(n)

    dAdt_diversity_cros_m = dAdt_diversity_cros.mean(axis=0) 
    dAdt_lrghr_diversity_cros_m = dAdt_lrghr_diversity_cros.mean(axis=0)
    dAdt_A_lrghr_diversity_cros_m = dAdt_A_lrghr_diversity_cros.mean(axis=0)
    dAdt_C_lrghr_diversity_cros_m = dAdt_C_lrghr_diversity_cros.mean(axis=0)
    dAdt_A_cnvhr_diversity_cros_m = dAdt_A_cnvhr_diversity_cros.mean(axis=0)
    dAdt_C_cnvhr_diversity_cros_m = dAdt_C_cnvhr_diversity_cros.mean(axis=0)
    dAdt_cnvhr_diversity_cros_m = dAdt_cnvhr_diversity_cros.mean(axis=0)
    dTdt_lrghr_diversity_cros_m = dTdt_lrghr_diversity_cros.mean(axis=0)
    dTdt_cnvhr_diversity_cros_m = dTdt_cnvhr_diversity_cros.mean(axis=0)   
    dTdt_diversity_cros_m = np.nanmean(dTdt_diversity_cros,axis=0)
    LWA_diversity_cros_m = LWA_diversity_cros.mean(axis=0)
    

    dTdt_diversity_cros_com.append(dTdt_diversity_cros_m)
    dTdt_lrghr_diversity_cros_com.append(dTdt_lrghr_diversity_cros_m)
    dTdt_cnvhr_diversity_cros_com.append(dTdt_cnvhr_diversity_cros_m)
    dAdt_diversity_cros_com.append(dAdt_diversity_cros_m)
    dAdt_lrghr_diversity_cros_com.append(dAdt_lrghr_diversity_cros_m)
    dAdt_A_lrghr_diversity_cros_com.append(dAdt_A_lrghr_diversity_cros_m)
    dAdt_C_lrghr_diversity_cros_com.append(dAdt_C_lrghr_diversity_cros_m)
    dAdt_A_cnvhr_diversity_cros_com.append(dAdt_A_cnvhr_diversity_cros_m)
    dAdt_C_cnvhr_diversity_cros_com.append(dAdt_C_cnvhr_diversity_cros_m)
    dAdt_cnvhr_diversity_cros_com.append(dAdt_cnvhr_diversity_cros_m)
    LWA_diversity_cros_com.append(LWA_diversity_cros_m)


with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_diversity_cros_com", "wb") as fp:
    pickle.dump(dAdt_diversity_cros_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_lrghr_diversity_cros_com", "wb") as fp:
    pickle.dump(dAdt_lrghr_diversity_cros_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_A_lrghr_diversity_cros_com", "wb") as fp:
    pickle.dump(dAdt_A_lrghr_diversity_cros_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_C_lrghr_diversity_cros_com", "wb") as fp:
    pickle.dump(dAdt_C_lrghr_diversity_cros_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_A_cnvhr_diversity_cros_com", "wb") as fp:
    pickle.dump(dAdt_A_cnvhr_diversity_cros_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_C_cnvhr_diversity_cros_com", "wb") as fp:
    pickle.dump(dAdt_C_cnvhr_diversity_cros_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_cnvhr_diversity_cros_com", "wb") as fp:
    pickle.dump(dAdt_cnvhr_diversity_cros_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_diversity_cros_com", "wb") as fp:
    pickle.dump(dTdt_diversity_cros_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_lrghr_diversity_cros_com", "wb") as fp:
    pickle.dump(dTdt_lrghr_diversity_cros_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_cnvhr_diversity_cros_com", "wb") as fp:
    pickle.dump(dTdt_cnvhr_diversity_cros_com, fp)
with open("/scratch/bell/liu3315/JRA55/blocks/LWA_diversity_cros_com", "wb") as fp:
    pickle.dump(LWA_diversity_cros_com, fp)


######################################################################################################################################
#%%
###### Figure 1 of the manuscript ######
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_Blocking_diversity_com", "rb") as fp:
    dAdt_Blocking_diversity_com = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_lrghr_Blocking_diversity_com", "rb") as fp:
    dAdt_lrghr_Blocking_diversity_com = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_A_lrghr_Blocking_diversity_com", "rb") as fp:
    dAdt_A_lrghr_Blocking_diversity_com = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_C_lrghr_Blocking_diversity_com", "rb") as fp:
    dAdt_C_lrghr_Blocking_diversity_com = pickle.load(fp)
    
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_cnvhr_Blocking_diversity_com", "rb") as fp:
    dAdt_cnvhr_Blocking_diversity_com = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_A_cnvhr_Blocking_diversity_com", "rb") as fp:
    dAdt_A_cnvhr_Blocking_diversity_com = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_C_cnvhr_Blocking_diversity_com", "rb") as fp:
    dAdt_C_cnvhr_Blocking_diversity_com = pickle.load(fp)

with open("/scratch/bell/liu3315/JRA55/blocks/Z_Blocking_diversity_com", "rb") as fp:
    Z_Blocking_diversity_com = pickle.load(fp)
    
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_diversity_cros_com", "rb") as fp:
    dAdt_diversity_cros_com = pickle.load(fp)    
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_lrghr_diversity_cros_com", "rb") as fp:
    dAdt_lrghr_diversity_cros_com = pickle.load(fp)   
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_A_lrghr_diversity_cros_com", "rb") as fp:
    dAdt_A_lrghr_diversity_cros_com = pickle.load(fp)   
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_C_lrghr_diversity_cros_com", "rb") as fp:
    dAdt_C_lrghr_diversity_cros_com = pickle.load(fp)   
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_cnvhr_diversity_cros_com", "rb") as fp:
    dAdt_cnvhr_diversity_cros_com = pickle.load(fp)   
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_A_cnvhr_diversity_cros_com", "rb") as fp:
    dAdt_A_cnvhr_diversity_cros_com = pickle.load(fp)   
with open("/scratch/bell/liu3315/JRA55/blocks/dAdt_C_cnvhr_diversity_cros_com", "rb") as fp:
    dAdt_C_cnvhr_diversity_cros_com = pickle.load(fp)  
with open("/scratch/bell/liu3315/JRA55/blocks/LWA_diversity_cros_com", "rb") as fp:
    LWA_diversity_cros_com = pickle.load(fp)
    

with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_Blocking_diversity_com", "rb") as fp:
    dTdt_Blocking_diversity_com = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_lrghr_Blocking_diversity_com", "rb") as fp:
    dTdt_lrghr_Blocking_diversity_com = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_cnvhr_Blocking_diversity_com", "rb") as fp:
    dTdt_cnvhr_Blocking_diversity_com = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_diversity_cros_com", "rb") as fp:
    dTdt_diversity_cros_com = pickle.load(fp)  
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_lrghr_diversity_cros_com", "rb") as fp:
    dTdt_lrghr_diversity_cros_com = pickle.load(fp)
with open("/scratch/bell/liu3315/JRA55/blocks/dTdt_cnvhr_diversity_cros_com", "rb") as fp:
    dTdt_cnvhr_diversity_cros_com = pickle.load(fp)  
 
minlev = Z_Blocking_diversity_com[2].min()
maxlev = Z_Blocking_diversity_com[2].max()
# levs_Z = np.linspace(5250,5800,11)
levs_Z = np.linspace(5249,5740,11)
minlev = dAdt_Blocking_diversity_com[2].min()
maxlev = dAdt_Blocking_diversity_com[2].max()
levs_dAdt = np.linspace(-5, 5, 21)
levs_dTdt = np.linspace(0, 2, 15)

lat_range=int(10/dlat)+1
lon_range=int(60/dlon)+1
duration = 6  

i=2
fig = plt.figure(figsize=[10,12])
ax = fig.add_subplot(4,3,1)
a = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dTdt_Blocking_diversity_com[i])), dTdt_Blocking_diversity_com[i], levs_dTdt, cmap='hot_r',extend='both')  
ax.contour(np.arange(0,lon_range),np.arange(0,len(dTdt_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
ax.set_yticks([0,8.5,17, 25.5,33])
ax.set_yticklabels(['-20','-10','lat_c','+10','+20'])
ax.set_xticks([0,8,16,24,32,40,48])
ax.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ax.set_title("Total diabatic heating \n(a)", pad=5, fontsize=12)
# ax.set_xlabel('relative longitude',fontsize=12)
ax.set_ylabel('relative latitude',fontsize=12)     

bx = fig.add_subplot(4,3,2)
b = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dTdt_lrghr_Blocking_diversity_com[i])), dTdt_lrghr_Blocking_diversity_com[i], levs_dTdt, cmap='hot_r',extend='both')  
bx.contour(np.arange(0,lon_range),np.arange(0,len(dTdt_lrghr_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
bx.set_yticks([0,8,16, 24,32])
bx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
bx.set_xticks([0,8,16,24,32,40,48])
bx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
bx.set_title("Large-scale condensation \n(b)" , pad=5, fontsize=12)

cx = fig.add_subplot(4,3,3)
c = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dTdt_cnvhr_Blocking_diversity_com[i])), dTdt_cnvhr_Blocking_diversity_com[i], levs_dTdt, cmap='hot_r',extend='both')  
cx.contour(np.arange(0,lon_range),np.arange(0,len(dTdt_cnvhr_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
cx.set_yticks([0,8,16, 24,32])
cx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
cx.set_xticks([0,8,16,24,32,40,48])
cx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
cx.set_title("Convective heating \n(c)" , pad=5, fontsize=12)

cbar = fig.add_axes([0.93,0.70,0.01,0.18])
cb = plt.colorbar(a, cax=cbar, ticks=[0,0.5,1,1.5,2]) 
cb.set_label('diabatic heating rate (K/day)',fontsize=10)


dx = fig.add_subplot(4,3,4)
d = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[i])), dAdt_Blocking_diversity_com[i], levs_dAdt, cmap='RdBu_r',extend='both')  
dx.contour(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
dx.set_yticks([0,8.5,17, 25.5,33])
dx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
dx.set_xticks([0,8,16,24,32,40,48])
dx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
dx.set_title("(d)", pad=5, fontsize=12)
# ax.set_xlabel('relative longitude',fontsize=12)
dx.set_ylabel('relative latitude',fontsize=12)     

ex = fig.add_subplot(4,3,5)
e = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_lrghr_Blocking_diversity_com[i])), dAdt_lrghr_Blocking_diversity_com[i], levs_dAdt, cmap='RdBu_r',extend='both')  
ex.contour(np.arange(0,lon_range),np.arange(0,len(dAdt_lrghr_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
ex.set_yticks([0,8,16, 24,32])
ex.set_yticklabels(['-20','-10','lat_c','+10','+20'])
ex.set_xticks([0,8,16,24,32,40,48])
ex.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ex.set_title("(e)" , pad=5, fontsize=12)

fx = fig.add_subplot(4,3,6)
f = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_cnvhr_Blocking_diversity_com[i])), dAdt_cnvhr_Blocking_diversity_com[i], levs_dAdt, cmap='RdBu_r',extend='both')  
fx.contour(np.arange(0,lon_range),np.arange(0,len(dAdt_cnvhr_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
fx.set_yticks([0,8,16, 24,32])
fx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
fx.set_xticks([0,8,16,24,32,40,48])
fx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
fx.set_title("(f)" , pad=5, fontsize=12)

cbar = fig.add_axes([0.93,0.50,0.01,0.18])
cb = plt.colorbar(d, cax=cbar, ticks=[-5,-4,-3,-2,-1,0,1,2,3,4,5]) 
cb.set_label('LWA tendency (ms-1/day)',fontsize=10)


maxlevel = np.max(dAdt_diversity_cros_com[0])
minlevel = np.min(dAdt_diversity_cros_com[0]) 
levs_dAdt_cross = np.linspace(-15, 15, 21)
levs_dTdt_cross = np.linspace(0, 2, 15)

maxlevel = np.max(LWA_diversity_cros_com[0])
minlevel = np.min(LWA_diversity_cros_com[0]) 
levs_LWA_cross = np.linspace(0, 200, 15)

zlev = np.arange(1000,31000,1000)
zlev1=0; zlev2=12

gx = fig.add_subplot(4,3,7)
g=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dTdt_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_cross,cmap='hot_r',extend ='both')
gx.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
gx.set_ylabel('height (km)',fontsize=12)
gx.set_ylim(1000,10000)
gx.set_yticks([1000,3000,5000,7000,9000])
gx.set_yticklabels(['1','3','5','7','9'])
gx.set_xticks([0,8,16,24,32,40,48])
gx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
gx.set_title("(g)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

hx = fig.add_subplot(4,3,8)
h=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dTdt_lrghr_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_cross,cmap='hot_r',extend ='both')
hx.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
hx.set_ylim(1000,10000)
hx.set_yticks([1000,3000,5000,7000,9000])
hx.set_yticklabels(['1','3','5','7','9'])
hx.set_xticks([0,8,16,24,32,40,48])
hx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
hx.set_title("(h)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

ix = fig.add_subplot(4,3,9)
i=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dTdt_cnvhr_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_cross,cmap='hot_r',extend ='both')
ix.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
ix.set_ylim(1000,10000)
ix.set_yticks([1000,3000,5000,7000,9000])
ix.set_yticklabels(['1','3','5','7','9'])
ix.set_xticks([0,8,16,24,32,40,48])
ix.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ix.set_title("(i)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

cbar = fig.add_axes([0.93,0.30,0.01,0.18])
cb = plt.colorbar(g, cax=cbar, ticks=[0,0.5,1,1.5,2]) 
cb.set_label('diabatic heating (K/day)',fontsize=10)

jx = fig.add_subplot(4,3,10)
j=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dAdt_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
jx.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
jx.set_ylabel('height (km)',fontsize=12)
jx.set_ylim(1000,10000)
jx.set_yticks([1000,3000,5000,7000,9000])
jx.set_yticklabels(['1','3','5','7','9'])
jx.set_xticks([0,8,16,24,32,40,48])
jx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
jx.set_title("(j)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

kx = fig.add_subplot(4,3,11)
k=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dAdt_lrghr_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
kx.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
kx.set_ylim(1000,10000)
kx.set_yticks([1000,3000,5000,7000,9000])
kx.set_yticklabels(['1','3','5','7','9'])
kx.set_xticks([0,8,16,24,32,40,48])
kx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
kx.set_title("(k)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

lx = fig.add_subplot(4,3,12)
l=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dAdt_cnvhr_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
lx.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
lx.set_ylim(1000,10000)
lx.set_yticks([1000,3000,5000,7000,9000])
lx.set_yticklabels(['1','3','5','7','9'])
lx.set_xticks([0,8,16,24,32,40,48])
lx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
lx.set_title("(l)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

cbar = fig.add_axes([0.93,0.10,0.01,0.18])
cb = plt.colorbar(j, cax=cbar, ticks=[-15,-12,-9,-6,-3,0,3,6,9,12,15]) 
cb.set_label('LWA tendency (ms-1/day)',fontsize=10)
#%%
### Figure 2 of the manuscript ###

levs_Z = np.linspace(5250,5800,11)
levs_dAdt = np.linspace(-5, 5, 21)
levs_dTdt = np.linspace(0, 2, 15)

lat_range=int(10/dlat)+1
lon_range=int(60/dlon)+1

i=0
fig = plt.figure(figsize=[10,12])
ax = fig.add_subplot(4,3,1)
a = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dTdt_Blocking_diversity_com[i])), dTdt_Blocking_diversity_com[i], levs_dTdt, cmap='hot_r',extend='both')  
ax.contour(np.arange(0,lon_range),np.arange(0,len(dTdt_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
ax.set_yticks([0,8.5,17, 25.5,33])
ax.set_yticklabels(['-20','-10','lat_c','+10','+20'])
ax.set_xticks([0,8,16,24,32,40,48])
ax.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ax.set_title("Total diabatic heating \n(a)", pad=5, fontsize=12)
# ax.set_xlabel('relative longitude',fontsize=12)
ax.set_ylabel('relative latitude',fontsize=12)     

bx = fig.add_subplot(4,3,2)
b = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dTdt_lrghr_Blocking_diversity_com[i])), dTdt_lrghr_Blocking_diversity_com[i], levs_dTdt, cmap='hot_r',extend='both')  
bx.contour(np.arange(0,lon_range),np.arange(0,len(dTdt_lrghr_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
bx.set_yticks([0,8,16, 24,32])
bx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
bx.set_xticks([0,8,16,24,32,40,48])
bx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
bx.set_title("Large-scale condensation \n(b)" , pad=5, fontsize=12)

cx = fig.add_subplot(4,3,3)
c = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dTdt_cnvhr_Blocking_diversity_com[i])), dTdt_cnvhr_Blocking_diversity_com[i], levs_dTdt, cmap='hot_r',extend='both')  
cx.contour(np.arange(0,lon_range),np.arange(0,len(dTdt_cnvhr_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
cx.set_yticks([0,8,16, 24,32])
cx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
cx.set_xticks([0,8,16,24,32,40,48])
cx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
cx.set_title("Convective heating \n(c)" , pad=5, fontsize=12)

cbar = fig.add_axes([0.93,0.70,0.01,0.18])
cb = plt.colorbar(a, cax=cbar, ticks=[0,0.5,1,1.5,2]) 
cb.set_label('diabatic heating rate (K/day)',fontsize=10)


dx = fig.add_subplot(4,3,4)
d = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[i])), dAdt_Blocking_diversity_com[i], levs_dAdt, cmap='RdBu_r',extend='both')  
dx.contour(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
dx.set_yticks([0,8.5,17, 25.5,33])
dx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
dx.set_xticks([0,8,16,24,32,40,48])
dx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
dx.set_title("(d)", pad=5, fontsize=12)
# ax.set_xlabel('relative longitude',fontsize=12)
dx.set_ylabel('relative latitude',fontsize=12)     

ex = fig.add_subplot(4,3,5)
e = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_lrghr_Blocking_diversity_com[i])), dAdt_lrghr_Blocking_diversity_com[i], levs_dAdt, cmap='RdBu_r',extend='both')  
ex.contour(np.arange(0,lon_range),np.arange(0,len(dAdt_lrghr_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
ex.set_yticks([0,8,16, 24,32])
ex.set_yticklabels(['-20','-10','lat_c','+10','+20'])
ex.set_xticks([0,8,16,24,32,40,48])
ex.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ex.set_title("(e)" , pad=5, fontsize=12)

fx = fig.add_subplot(4,3,6)
f = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_cnvhr_Blocking_diversity_com[i])), dAdt_cnvhr_Blocking_diversity_com[i], levs_dAdt, cmap='RdBu_r',extend='both')  
fx.contour(np.arange(0,lon_range),np.arange(0,len(dAdt_cnvhr_Blocking_diversity_com[i])), Z_Blocking_diversity_com[i], levs_Z, colors='k')  
fx.set_yticks([0,8,16, 24,32])
fx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
fx.set_xticks([0,8,16,24,32,40,48])
fx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
fx.set_title("(f)" , pad=5, fontsize=12)

cbar = fig.add_axes([0.93,0.50,0.01,0.18])
cb = plt.colorbar(d, cax=cbar, ticks=[-5,-4,-3,-2,-1,0,1,2,3,4,5]) 
cb.set_label('LWA tendency (ms-1/day)',fontsize=10)


levs_dAdt_cross = np.linspace(-15, 15, 21)
levs_dTdt_cross = np.linspace(0, 2, 15)
levs_LWA_cross = np.linspace(0, 200, 15)

zlev = np.arange(1000,31000,1000)
zlev1=0; zlev2=12

gx = fig.add_subplot(4,3,7)
g=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dTdt_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_cross,cmap='hot_r',extend ='both')
gx.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
gx.set_ylabel('height (km)',fontsize=12)
gx.set_ylim(1000,10000)
gx.set_yticks([1000,3000,5000,7000,9000])
gx.set_yticklabels(['1','3','5','7','9'])
gx.set_xticks([0,8,16,24,32,40,48])
gx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
gx.set_title("(g)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

hx = fig.add_subplot(4,3,8)
h=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dTdt_lrghr_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_cross,cmap='hot_r',extend ='both')
hx.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
hx.set_ylim(1000,10000)
hx.set_yticks([1000,3000,5000,7000,9000])
hx.set_yticklabels(['1','3','5','7','9'])
hx.set_xticks([0,8,16,24,32,40,48])
hx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
hx.set_title("(h)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

ix = fig.add_subplot(4,3,9)
i=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dTdt_cnvhr_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_cross,cmap='hot_r',extend ='both')
ix.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
ix.set_ylim(1000,10000)
ix.set_yticks([1000,3000,5000,7000,9000])
ix.set_yticklabels(['1','3','5','7','9'])
ix.set_xticks([0,8,16,24,32,40,48])
ix.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ix.set_title("(i)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

cbar = fig.add_axes([0.93,0.30,0.01,0.18])
cb = plt.colorbar(g, cax=cbar, ticks=[0,0.5,1,1.5,2]) 
cb.set_label('diabatic heating (K/day)',fontsize=10)

jx = fig.add_subplot(4,3,10)
j=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dAdt_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
jx.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
jx.set_ylabel('height (km)',fontsize=12)
jx.set_ylim(1000,10000)
jx.set_yticks([1000,3000,5000,7000,9000])
jx.set_yticklabels(['1','3','5','7','9'])
jx.set_xticks([0,8,16,24,32,40,48])
jx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
jx.set_title("(j)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

kx = fig.add_subplot(4,3,11)
k=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dAdt_lrghr_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
kx.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
kx.set_ylim(1000,10000)
kx.set_yticks([1000,3000,5000,7000,9000])
kx.set_yticklabels(['1','3','5','7','9'])
kx.set_xticks([0,8,16,24,32,40,48])
kx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
kx.set_title("(k)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

lx = fig.add_subplot(4,3,12)
l=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dAdt_cnvhr_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
lx.contour(np.arange(0,lon_range), zlev[zlev1:zlev2], LWA_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
lx.set_ylim(1000,10000)
lx.set_yticks([1000,3000,5000,7000,9000])
lx.set_yticklabels(['1','3','5','7','9'])
lx.set_xticks([0,8,16,24,32,40,48])
lx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
lx.set_title("(l)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

cbar = fig.add_axes([0.93,0.10,0.01,0.18])
cb = plt.colorbar(j, cax=cbar, ticks=[-15,-12,-9,-6,-3,0,3,6,9,12,15]) 
cb.set_label('LWA tendency (ms-1/day)',fontsize=10)
# %%
