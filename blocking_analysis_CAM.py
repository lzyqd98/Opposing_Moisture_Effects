#%%
###### This code is to plot blocking events and study their basic featuers ######
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
import xarray as xr

#%%
### read basic data ###
experiment_name = "CAM6_F2000climo_f19f19_plev_A"

path = glob.glob(r"/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/LWA_Z500/std*.nc")
path.sort()
path=path[365*10:365*80]
N=len(path) 

### Read basic variables ###
file0 = Dataset(path[0],'r')
lon = file0.variables['lon'][:]
lat = file0.variables['lat'][:]
lat_mid = int(len(lat)/2)
lat_SH = lat[0:lat_mid]
lat_NH = lat[lat_mid:]
nlon = len(lon)
nlat = len(lat)
nlat_SH = len(lat_SH)
nlat_NH =len(lat_NH)
dlat = 180/(nlat-1) 
dlon = (lon[1] - lon[0]) 

file0.close()

file_blocking = "/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/Blocking.nc"
Block_all = xr.open_dataset(file_blocking)
Block_ridge = Block_all.where(Block_all.Blocking_type == 0, drop=True)
Block_trough = Block_all.where(Block_all.Blocking_type == 1, drop=True)
Block_dipole = Block_all.where(Block_all.Blocking_type == 2, drop=True)
Block_all.close()


Block_type = Block_dipole
Blocking_date= Block_type["Blocking_date"].values
Blocking_duration = Block_type["Blocking_duration"].values
Blocking_lon = Block_type["Blocking_lon"].values
Blocking_lat = Block_type["Blocking_lat"].values
B_freq = np.load("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/B_freq.npy")

### Time Management ###
### CAM simulation doesnot have -02/29 ###
Datestamp0 = pd.date_range(start="2011-01-01",end="2080-12-31")
# Datestamp0 = pd.date_range(start="2002-01-01",end="2025-12-31")
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
###### Code for doing the horizontal composite ######

path1 = glob.glob(r"/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/std*.nc")
path1.sort()
path1=path1[365*10:365*80]

path2 = glob.glob(r"/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/LWA/std*.nc")
path2.sort()
path2=path2[365*10:365*80]

LWA_Blocking_diversity_com = []
dAdt_Blocking_diversity_com = []
dTdt_Blocking_diversity_com = []
Z_Blocking_diversity_com = []
for type_i in [Block_ridge, Block_trough, Block_dipole]:
    Block_type = type_i
    n_event = len(Block_type["Blocking_peaking_date"])
    
    lat_range=int(int((90-np.max(Block_type.Blocking_peaking_lat.values[:]))/dlat)*2+1); lon_range=int(60/dlon)+1
    
    LWA_Blocking_diversity = np.zeros((n_event,lat_range, lon_range))
    dTdt_Blocking_diversity = np.zeros((n_event,lat_range, lon_range))
    dAdt_Blocking_diversity = np.zeros((n_event,lat_range, lon_range))
    Z_Blocking_diversity = np.zeros((n_event,lat_range, lon_range))

    for n in np.arange(n_event):
        
        LWA_d = np.zeros((nlat,nlon)); dTdt_d = np.zeros((nlat,nlon)); Z_d = np.zeros((nlat,nlon))
        dAdt_d = np.zeros((nlat,nlon))
        
        ### peaking date information ###
        peaking_date_index = int(Block_type.Blocking_peaking_date.values[n])
        peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Block_type.Blocking_peaking_lon.values[n])))
        peaking_lat_index = np.squeeze(np.array(np.where( lat[:]==Block_type.Blocking_peaking_lat.values[n])))
        
        file = Dataset(path1[peaking_date_index],'r')
        Z_d[:,:] = file.variables['hgt'][5,:,:]
        file.close()
        
        
        file2 = Dataset(path2[peaking_date_index],'r')
        dTdt_d = file2.variables['dTdt_moist'][:,:] #2 850hpa; 5 500hPa
        
        H = 7000
        n_plev = 33
        zlev = np.arange(0,32001,1000)
        rho = np.array([np.exp(-zlev[i]/H) for i in np.arange(n_plev)]) 
        rho = rho[:,np.newaxis,np.newaxis] * np.ones((n_plev,nlat,nlon))  ## Density in 3D ##
        dTdt_d = dTdt_d * rho[:]                                   ## we only use 1km-31km interia points
        dTdt_d = dTdt_d[:,:,:].sum(axis = 0) / rho[:].sum(axis=0)    


        dAdt_d = file2.variables['dAdt_moist_column'][:,:] 
        LWA_d = file2.variables['LWA_column'][:,:]
        file2.close()

        ### shift the array to make the conpoiste
        LWA_d = np.roll(LWA_d, int(nlon/2)-peaking_lon_index, axis=1)
        dTdt_d = np.roll(dTdt_d, int(nlon/2)-peaking_lon_index, axis=1)
        dAdt_d = np.roll(dAdt_d, int(nlon/2)-peaking_lon_index, axis=1)
        Z_d = np.roll(Z_d, int(nlon/2)-peaking_lon_index, axis=1)

        
        LWA_Blocking_diversity[n,:,:] = LWA_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,    int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dTdt_Blocking_diversity[n,:,:] = dTdt_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dAdt_Blocking_diversity[n,:,:] = dAdt_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        Z_Blocking_diversity[n,:,:] = Z_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,        int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]

        print(n)
        
    LWA_Blocking_diversity_m = LWA_Blocking_diversity.mean(axis=0)
    dTdt_Blocking_diversity_m = np.nanmean(dTdt_Blocking_diversity, axis=0)
    dAdt_Blocking_diversity_m = np.nanmean(dAdt_Blocking_diversity, axis=0)
    Z_Blocking_diversity_m = Z_Blocking_diversity.mean(axis=0)
        
    LWA_Blocking_diversity_com.append(LWA_Blocking_diversity_m)
    dAdt_Blocking_diversity_com.append(dAdt_Blocking_diversity_m)
    dTdt_Blocking_diversity_com.append(dTdt_Blocking_diversity_m)
    Z_Blocking_diversity_com.append(Z_Blocking_diversity_m)


with open("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/LWA_Blocking_diversity_com", "wb") as fp:
    pickle.dump(LWA_Blocking_diversity_com, fp)
with open("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/dAdt_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dAdt_Blocking_diversity_com, fp) 
with open("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/dTdt_Blocking_diversity_com", "wb") as fp:
    pickle.dump(dTdt_Blocking_diversity_com, fp) 
with open("/scratch/bell/liu3315/cesm_output/"+experiment_name+"/LWA_Z500/Blocking/Z_Blocking_diversity_com", "wb") as fp:
    pickle.dump(Z_Blocking_diversity_com, fp) 
    
    

#%%
### Read data  ###

with open("/scratch/bell/liu3315/cesm_output/CAM4_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/LWA_Blocking_diversity_com", "rb") as fp:
    LWA_Blocking_diversity_com_CAM4 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM4_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dAdt_Blocking_diversity_com", "rb") as fp:
    dAdt_Blocking_diversity_com_CAM4 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM4_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dAdt_LC_Blocking_diversity_com", "rb") as fp:
    dAdt_LC_Blocking_diversity_com_CAM4 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM4_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dAdt_CON_Blocking_diversity_com", "rb") as fp:
    dAdt_CON_Blocking_diversity_com_CAM4 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM4_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dTdt_Blocking_diversity_com", "rb") as fp:
    dTdt_Blocking_diversity_com_CAM4 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM4_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dTdt_LC_Blocking_diversity_com", "rb") as fp:
    dTdt_LC_Blocking_diversity_com_CAM4 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM4_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dTdt_CON_Blocking_diversity_com", "rb") as fp:
    dTdt_CON_Blocking_diversity_com_CAM4 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM4_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/Z_Blocking_diversity_com", "rb") as fp:
    Z_Blocking_diversity_com_CAM4 = pickle.load(fp)

with open("/scratch/bell/liu3315/cesm_output/CAM5_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/LWA_Blocking_diversity_com", "rb") as fp:
    LWA_Blocking_diversity_com_CAM5 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM5_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dAdt_Blocking_diversity_com", "rb") as fp:
    dAdt_Blocking_diversity_com_CAM5 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM5_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dAdt_LC_Blocking_diversity_com", "rb") as fp:
    dAdt_LC_Blocking_diversity_com_CAM5 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM5_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dAdt_CON_Blocking_diversity_com", "rb") as fp:
    dAdt_CON_Blocking_diversity_com_CAM5 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM5_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dTdt_Blocking_diversity_com", "rb") as fp:
    dTdt_Blocking_diversity_com_CAM5 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM5_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dTdt_LC_Blocking_diversity_com", "rb") as fp:
    dTdt_LC_Blocking_diversity_com_CAM5 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM5_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dTdt_CON_Blocking_diversity_com", "rb") as fp:
    dTdt_CON_Blocking_diversity_com_CAM5 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM5_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/Z_Blocking_diversity_com", "rb") as fp:
    Z_Blocking_diversity_com_CAM5 = pickle.load(fp)
    
with open("/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/LWA_Blocking_diversity_com", "rb") as fp:
    LWA_Blocking_diversity_com_CAM6 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dAdt_Blocking_diversity_com", "rb") as fp:
    dAdt_Blocking_diversity_com_CAM6 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dAdt_LC_Blocking_diversity_com", "rb") as fp:
    dAdt_LC_Blocking_diversity_com_CAM6 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dAdt_CON_Blocking_diversity_com", "rb") as fp:
    dAdt_CON_Blocking_diversity_com_CAM6 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dTdt_Blocking_diversity_com", "rb") as fp:
    dTdt_Blocking_diversity_com_CAM6 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dTdt_LC_Blocking_diversity_com", "rb") as fp:
    dTdt_LC_Blocking_diversity_com_CAM6 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/dTdt_CON_Blocking_diversity_com", "rb") as fp:
    dTdt_CON_Blocking_diversity_com_CAM6 = pickle.load(fp)
with open("/scratch/bell/liu3315/cesm_output/CAM6_F2000climo_f19f19_plev_A/LWA_Z500/Blocking/Z_Blocking_diversity_com", "rb") as fp:
    Z_Blocking_diversity_com_CAM6 = pickle.load(fp)
    
dAdt_Blocking_dipole_com_QG = np.load("/scratch/bell/liu3315/Moist_QG/dAdt_moist_QG_composite_dipole.npy")
dAdt_Blocking_ridge_com_QG = np.load("/scratch/bell/liu3315/Moist_QG/dAdt_moist_QG_composite_ridge.npy")
dTdt_Blocking_dipole_com_QG = np.load("/scratch/bell/liu3315/Moist_QG/dTdt_moist_QG_composite_dipole.npy")
dTdt_Blocking_ridge_com_QG = np.load("/scratch/bell/liu3315/Moist_QG/dTdt_moist_QG_composite_ridge.npy")
PV_Blocking_dipole_com_QG = np.load("/scratch/bell/liu3315/Moist_QG/PV_moist_QG_composite_dipole.npy")
PV_Blocking_ridge_com_QG = np.load("/scratch/bell/liu3315/Moist_QG/PV_moist_QG_composite_ridge.npy")
QG_x = np.load("/scratch/bell/liu3315/Moist_QG/QG_x.npy")
QG_y = np.load("/scratch/bell/liu3315/Moist_QG/QG_y.npy")


#%%
### Figure 4 ###
# levs_Z = np.linspace(5300, 5800, 15)
levs_Z = np.linspace(5200, 5700, 15)
levs_dAdt = np.linspace(-4.5e-5, 4.5e-5, 21)
levs_dAdt = np.linspace(-4.5e-5, 4.5e-5, 21)
levs_dTdt = np.linspace(0, 2e-5, 21)

levs_qgpv = np.linspace(-1.2e-4, 1.5e-4, 11)
levs_dAdt_QG = np.linspace(-0.0003, 0.0003, 11)
levs_dTdt_QG = np.linspace(0, 4e-5, 21)

lat_range = int(((90-np.max(Block_dipole.Blocking_peaking_lat.values[:]))/dlat)*2+1)
lon_range = int(60/dlon)+1

duration = 6
fontsize = 14
index = 0
fig = plt.figure(figsize=(13.5,10))

# Layout:
# QG | QG colorbar | gap | CAM4 | CAM5 | CAM6 | CAM colorbar
gs = fig.add_gridspec(
    3, 7,
    width_ratios=[1, 0.03, 0.12, 1, 1, 1, 0.03],
    wspace=0.07,
    hspace=0.18
)


# ============================================================
# Row 1: Total diabatic heating
# ============================================================

ax1 = fig.add_subplot(gs[0,0])

cf_qg = ax1.contourf(
    QG_x, QG_y,
    dAdt_Blocking_ridge_com_QG,
    levs_dAdt_QG,
    cmap='RdBu_r',
    extend='both'
)

ax1.contour(
    QG_x, QG_y,
    PV_Blocking_ridge_com_QG,
    levs_qgpv,
    colors='k',
    alpha=0.7,
    linewidths=2.5,
    linestyles='solid'
)

ax1.set_title("Two-layer Moist QG\n(a)", pad=5, fontsize=fontsize)
ax1.set_ylabel('Total diabatic heating', fontsize=fontsize)


ax2 = fig.add_subplot(gs[0,3])

cf_cam = ax2.contourf(
    np.arange(lon_range),
    np.arange(len(dAdt_Blocking_diversity_com_CAM4[index])),
    dAdt_Blocking_diversity_com_CAM4[index],
    levs_dAdt,
    cmap='RdBu_r',
    extend='both'
)

ax2.contour(
    np.arange(lon_range),
    np.arange(len(dAdt_Blocking_diversity_com_CAM4[index])),
    Z_Blocking_diversity_com_CAM4[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax2.set_title("CAM4\n(b)", pad=5, fontsize=fontsize)


ax3 = fig.add_subplot(gs[0,4])

lat_range = len(dAdt_Blocking_diversity_com_CAM5[index])

ax3.contourf(
    np.arange(lon_range),
    np.arange(lat_range),
    dAdt_Blocking_diversity_com_CAM5[index],
    levs_dAdt,
    cmap='RdBu_r',
    extend='both'
)

ax3.contour(
    np.arange(lon_range),
    np.arange(lat_range),
    Z_Blocking_diversity_com_CAM5[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax3.set_title("CAM5\n(c)", pad=5, fontsize=fontsize)


ax4 = fig.add_subplot(gs[0,5])

ax4.contourf(
    np.arange(lon_range),
    np.arange(len(dAdt_Blocking_diversity_com_CAM6[index])),
    dAdt_Blocking_diversity_com_CAM6[index],
    levs_dAdt,
    cmap='RdBu_r',
    extend='both'
)

ax4.contour(
    np.arange(lon_range),
    np.arange(len(dAdt_Blocking_diversity_com_CAM6[index])),
    Z_Blocking_diversity_com_CAM6[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax4.set_title("CAM6\n(d)", pad=5, fontsize=fontsize)


# ============================================================
# Row 2: Large-scale condensation
# ============================================================

ax5 = fig.add_subplot(gs[1,0])

ax5.contourf(
    QG_x, QG_y,
    dAdt_Blocking_ridge_com_QG,
    levs_dAdt_QG,
    cmap='RdBu_r',
    extend='both'
)

ax5.contour(
    QG_x, QG_y,
    PV_Blocking_ridge_com_QG,
    levs_qgpv,
    colors='k',
    alpha=0.7,
    linewidths=2.5,
    linestyles='solid'
)

ax5.set_title("(e)", pad=5, fontsize=fontsize)
ax5.set_ylabel('Large-scale condensation', fontsize=fontsize)


ax6 = fig.add_subplot(gs[1,3])

ax6.contourf(
    np.arange(lon_range),
    np.arange(len(dAdt_LC_Blocking_diversity_com_CAM4[index])),
    dAdt_LC_Blocking_diversity_com_CAM4[index],
    levs_dAdt,
    cmap='RdBu_r',
    extend='both'
)

ax6.contour(
    np.arange(lon_range),
    np.arange(len(dAdt_LC_Blocking_diversity_com_CAM4[index])),
    Z_Blocking_diversity_com_CAM4[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax6.set_title("(f)", pad=5, fontsize=fontsize)


ax7 = fig.add_subplot(gs[1,4])

lat_range = len(dAdt_LC_Blocking_diversity_com_CAM5[index])

ax7.contourf(
    np.arange(lon_range),
    np.arange(lat_range),
    dAdt_LC_Blocking_diversity_com_CAM5[index],
    levs_dAdt,
    cmap='RdBu_r',
    extend='both'
)

ax7.contour(
    np.arange(lon_range),
    np.arange(lat_range),
    Z_Blocking_diversity_com_CAM5[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax7.set_title("(g)", pad=5, fontsize=fontsize)


ax8 = fig.add_subplot(gs[1,5])

ax8.contourf(
    np.arange(lon_range),
    np.arange(len(dAdt_LC_Blocking_diversity_com_CAM6[index])),
    dAdt_LC_Blocking_diversity_com_CAM6[index],
    levs_dAdt,
    cmap='RdBu_r',
    extend='both'
)

ax8.contour(
    np.arange(lon_range),
    np.arange(len(dAdt_LC_Blocking_diversity_com_CAM6[index])),
    Z_Blocking_diversity_com_CAM6[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax8.set_title("(h)", pad=5, fontsize=fontsize)


# ============================================================
# Row 3: Moist convection
# ============================================================

ax9 = fig.add_subplot(gs[2,0])

ax9.contourf(
    QG_x, QG_y,
    dAdt_Blocking_ridge_com_QG*0.0 - 1e-10,
    levs_dAdt_QG,
    cmap='RdBu_r',
    extend='both'
)

ax9.contour(
    QG_x, QG_y,
    PV_Blocking_ridge_com_QG,
    levs_qgpv,
    colors='k',
    alpha=0.7,
    linewidths=2.5,
    linestyles='solid'
)

ax9.set_title("(i)", pad=5, fontsize=fontsize)
ax9.set_ylabel('Moist convection', fontsize=fontsize)


ax10 = fig.add_subplot(gs[2,3])

ax10.contourf(
    np.arange(lon_range),
    np.arange(len(dAdt_CON_Blocking_diversity_com_CAM4[index])),
    dAdt_CON_Blocking_diversity_com_CAM4[index],
    levs_dAdt,
    cmap='RdBu_r',
    extend='both'
)

ax10.contour(
    np.arange(lon_range),
    np.arange(len(dAdt_CON_Blocking_diversity_com_CAM4[index])),
    Z_Blocking_diversity_com_CAM4[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax10.set_title("(j)", pad=5, fontsize=fontsize)


ax11 = fig.add_subplot(gs[2,4])

lat_range = len(dAdt_CON_Blocking_diversity_com_CAM5[index])

ax11.contourf(
    np.arange(lon_range),
    np.arange(lat_range),
    dAdt_CON_Blocking_diversity_com_CAM5[index],
    levs_dAdt,
    cmap='RdBu_r',
    extend='both'
)

ax11.contour(
    np.arange(lon_range),
    np.arange(lat_range),
    Z_Blocking_diversity_com_CAM5[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax11.set_title("(k)", pad=5, fontsize=fontsize)


ax12 = fig.add_subplot(gs[2,5])

ax12.contourf(
    np.arange(lon_range),
    np.arange(len(dAdt_CON_Blocking_diversity_com_CAM6[index])),
    dAdt_CON_Blocking_diversity_com_CAM6[index],
    levs_dAdt,
    cmap='RdBu_r',
    extend='both'
)

ax12.contour(
    np.arange(lon_range),
    np.arange(len(dAdt_CON_Blocking_diversity_com_CAM6[index])),
    Z_Blocking_diversity_com_CAM6[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax12.set_title("(l)", pad=5, fontsize=fontsize)


# ============================================================
# Remove x/y ticks
# ============================================================

axes = [ax1, ax2, ax3, ax4,
        ax5, ax6, ax7, ax8,
        ax9, ax10, ax11, ax12]

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])


# ============================================================
# Colorbars
# ============================================================

# QG colorbar
cax_qg = fig.add_subplot(gs[:,1])
cb_qg = fig.colorbar(cf_qg,cax=cax_qg)
# cb_qg.set_label(r'Diabatic heating (s$^{-1}$)',fontsize=fontsize)
# cb_qg.ax.tick_params(labelsize=fontsize-1)
cb_qg.set_ticks([-3e-4,-2e-4,-1e-4,0, 1e-4, 2e-4,3e-4])
cb_qg.set_ticklabels(['-3', '-2', '-1', '0', '1', '2', '3'])
cb_qg.set_label(r'LWA tendency ($\times10^{-4}$ m s$^{-2}$)',
                 fontsize=fontsize)

# CAM colorbar
cax_cam = fig.add_subplot(gs[:,6])
cb_cam = fig.colorbar(cf_cam,cax=cax_cam)
# cb_cam.set_label(r'Diabatic heating rate(K/s)',fontsize=fontsize)
# cb_cam.ax.tick_params(labelsize=fontsize-1)
cb_cam.set_ticks([-4e-5,-3e-5, -2e-5, -1e-5,0, 1e-5, 2e-5,  3e-5, 4e-5])
cb_cam.set_ticklabels(['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'])
cb_cam.set_label(r'LWA tendency ($\times10^{-5}$ m s$^{-2}$)',
                 fontsize=fontsize)

# ============================================================
# Overall title
# ============================================================

fig.suptitle(
    'LWA Tendency due to Diabatic Heating in Ridge Blocks across the Model Hierarchy',
    fontsize=18,
    y=0.97
)

fig.subplots_adjust(
    left=0.07,
    right=0.96,
    top=0.90,
    bottom=0.06
)

plt.show()

#%%
### Figure 3 ###
# levs_Z = np.linspace(5300, 5800, 15)
levs_Z = np.linspace(5200, 5700, 15)
levs_dAdt = np.linspace(-4.5e-5, 4.5e-5, 21)
levs_dTdt = np.linspace(0, 2e-5, 21)

levs_qgpv = np.linspace(-1.2e-4, 1.5e-4, 11)
levs_dAdt_QG = np.linspace(-0.0003, 0.0003, 11)
levs_dTdt_QG = np.linspace(0, 4e-5, 21)

lat_range = int(((90-np.max(Block_dipole.Blocking_peaking_lat.values[:]))/dlat)*2+1)
lon_range = int(60/dlon)+1

duration = 6
fontsize = 14
index = 0
fig = plt.figure(figsize=(13.5,10))

# Layout:
# QG | QG colorbar | gap | CAM4 | CAM5 | CAM6 | CAM colorbar
gs = fig.add_gridspec(
    3, 7,
    width_ratios=[1, 0.03, 0.12, 1, 1, 1, 0.03],
    wspace=0.07,
    hspace=0.18
)


# ============================================================
# Row 1: Total diabatic heating
# ============================================================

ax1 = fig.add_subplot(gs[0,0])

cf_qg = ax1.contourf(
    QG_x, QG_y,
    dTdt_Blocking_ridge_com_QG,
    levs_dTdt_QG,
    cmap='hot_r',
    extend='both'
)

ax1.contour(
    QG_x, QG_y,
    PV_Blocking_ridge_com_QG,
    levs_qgpv,
    colors='k',
    alpha=0.7,
    linewidths=2.5,
    linestyles='solid'
)

ax1.set_title("Two-layer Moist QG\n(a)", pad=5, fontsize=fontsize)
ax1.set_ylabel('Total diabatic heating', fontsize=fontsize)


ax2 = fig.add_subplot(gs[0,3])

cf_cam = ax2.contourf(
    np.arange(lon_range),
    np.arange(len(dTdt_Blocking_diversity_com_CAM4[index])),
    dTdt_Blocking_diversity_com_CAM4[index],
    levs_dTdt,
    cmap='hot_r',
    extend='both'
)

ax2.contour(
    np.arange(lon_range),
    np.arange(len(dTdt_Blocking_diversity_com_CAM4[index])),
    Z_Blocking_diversity_com_CAM4[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax2.set_title("CAM4\n(b)", pad=5, fontsize=fontsize)


ax3 = fig.add_subplot(gs[0,4])

lat_range = len(dTdt_Blocking_diversity_com_CAM5[index])

ax3.contourf(
    np.arange(lon_range),
    np.arange(lat_range),
    dTdt_Blocking_diversity_com_CAM5[index],
    levs_dTdt,
    cmap='hot_r',
    extend='both'
)

ax3.contour(
    np.arange(lon_range),
    np.arange(lat_range),
    Z_Blocking_diversity_com_CAM5[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax3.set_title("CAM5\n(c)", pad=5, fontsize=fontsize)


ax4 = fig.add_subplot(gs[0,5])

ax4.contourf(
    np.arange(lon_range),
    np.arange(len(dTdt_Blocking_diversity_com_CAM6[index])),
    dTdt_Blocking_diversity_com_CAM6[index],
    levs_dTdt,
    cmap='hot_r',
    extend='both'
)

ax4.contour(
    np.arange(lon_range),
    np.arange(len(dTdt_Blocking_diversity_com_CAM6[index])),
    Z_Blocking_diversity_com_CAM6[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax4.set_title("CAM6\n(d)", pad=5, fontsize=fontsize)


# ============================================================
# Row 2: Large-scale condensation
# ============================================================

ax5 = fig.add_subplot(gs[1,0])

ax5.contourf(
    QG_x, QG_y,
    dTdt_Blocking_ridge_com_QG,
    levs_dTdt_QG,
    cmap='hot_r',
    extend='both'
)

ax5.contour(
    QG_x, QG_y,
    PV_Blocking_ridge_com_QG,
    levs_qgpv,
    colors='k',
    alpha=0.7,
    linewidths=2.5,
    linestyles='solid'
)

ax5.set_title("(e)", pad=5, fontsize=fontsize)
ax5.set_ylabel('Large-scale condensation', fontsize=fontsize)


ax6 = fig.add_subplot(gs[1,3])

ax6.contourf(
    np.arange(lon_range),
    np.arange(len(dTdt_LC_Blocking_diversity_com_CAM4[index])),
    dTdt_LC_Blocking_diversity_com_CAM4[index],
    levs_dTdt,
    cmap='hot_r',
    extend='both'
)

ax6.contour(
    np.arange(lon_range),
    np.arange(len(dTdt_LC_Blocking_diversity_com_CAM4[index])),
    Z_Blocking_diversity_com_CAM4[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax6.set_title("(f)", pad=5, fontsize=fontsize)


ax7 = fig.add_subplot(gs[1,4])

lat_range = len(dTdt_LC_Blocking_diversity_com_CAM5[index])

ax7.contourf(
    np.arange(lon_range),
    np.arange(lat_range),
    dTdt_LC_Blocking_diversity_com_CAM5[index],
    levs_dTdt,
    cmap='hot_r',
    extend='both'
)

ax7.contour(
    np.arange(lon_range),
    np.arange(lat_range),
    Z_Blocking_diversity_com_CAM5[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax7.set_title("(g)", pad=5, fontsize=fontsize)


ax8 = fig.add_subplot(gs[1,5])

ax8.contourf(
    np.arange(lon_range),
    np.arange(len(dTdt_LC_Blocking_diversity_com_CAM6[index])),
    dTdt_LC_Blocking_diversity_com_CAM6[index],
    levs_dTdt,
    cmap='hot_r',
    extend='both'
)

ax8.contour(
    np.arange(lon_range),
    np.arange(len(dTdt_LC_Blocking_diversity_com_CAM6[index])),
    Z_Blocking_diversity_com_CAM6[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax8.set_title("(h)", pad=5, fontsize=fontsize)


# ============================================================
# Row 3: Moist convection
# ============================================================

ax9 = fig.add_subplot(gs[2,0])

ax9.contourf(
    QG_x, QG_y,
    dTdt_Blocking_ridge_com_QG*0.0 + 1e-10,
    levs_dTdt_QG,
    cmap='hot_r',
    extend='both'
)

ax9.contour(
    QG_x, QG_y,
    PV_Blocking_ridge_com_QG,
    levs_qgpv,
    colors='k',
    alpha=0.7,
    linewidths=2.5,
    linestyles='solid'
)

ax9.set_title("(i)", pad=5, fontsize=fontsize)
ax9.set_ylabel('Moist convection', fontsize=fontsize)


ax10 = fig.add_subplot(gs[2,3])

ax10.contourf(
    np.arange(lon_range),
    np.arange(len(dTdt_CON_Blocking_diversity_com_CAM4[index])),
    dTdt_CON_Blocking_diversity_com_CAM4[index],
    levs_dTdt,
    cmap='hot_r',
    extend='both'
)

ax10.contour(
    np.arange(lon_range),
    np.arange(len(dTdt_CON_Blocking_diversity_com_CAM4[index])),
    Z_Blocking_diversity_com_CAM4[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax10.set_title("(j)", pad=5, fontsize=fontsize)


ax11 = fig.add_subplot(gs[2,4])

lat_range = len(dTdt_CON_Blocking_diversity_com_CAM5[index])

ax11.contourf(
    np.arange(lon_range),
    np.arange(lat_range),
    dTdt_CON_Blocking_diversity_com_CAM5[index],
    levs_dTdt,
    cmap='hot_r',
    extend='both'
)

ax11.contour(
    np.arange(lon_range),
    np.arange(lat_range),
    Z_Blocking_diversity_com_CAM5[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax11.set_title("(k)", pad=5, fontsize=fontsize)


ax12 = fig.add_subplot(gs[2,5])

ax12.contourf(
    np.arange(lon_range),
    np.arange(len(dTdt_CON_Blocking_diversity_com_CAM6[index])),
    dTdt_CON_Blocking_diversity_com_CAM6[index],
    levs_dTdt,
    cmap='hot_r',
    extend='both'
)

ax12.contour(
    np.arange(lon_range),
    np.arange(len(dTdt_CON_Blocking_diversity_com_CAM6[index])),
    Z_Blocking_diversity_com_CAM6[index],
    levs_Z,
    colors='k',
    alpha=0.7,
    linewidths=2.5
)

ax12.set_title("(l)", pad=5, fontsize=fontsize)


# ============================================================
# Remove x/y ticks
# ============================================================

axes = [ax1, ax2, ax3, ax4,
        ax5, ax6, ax7, ax8,
        ax9, ax10, ax11, ax12]

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])


# ============================================================
# Colorbars
# ============================================================

# QG colorbar
cax_qg = fig.add_subplot(gs[:,1])
cb_qg = fig.colorbar(cf_qg,cax=cax_qg)
# cb_qg.set_label(r'Diabatic heating (s$^{-1}$)',fontsize=fontsize)
# cb_qg.ax.tick_params(labelsize=fontsize-1)
cb_qg.set_ticks([0, 1e-5, 2e-5,3e-5,4e-5])
cb_qg.set_ticklabels(['0', '1', '2', '3', '4'])
cb_qg.set_label(r'Diabatic heating ($\times10^{-5}$ K s$^{-1}$)',
                 fontsize=fontsize)

# CAM colorbar
cax_cam = fig.add_subplot(gs[:,6])
cb_cam = fig.colorbar(cf_cam,cax=cax_cam)
# cb_cam.set_label(r'Diabatic heating rate(K/s)',fontsize=fontsize)
# cb_cam.ax.tick_params(labelsize=fontsize-1)
cb_cam.set_ticks([0, 0.5e-5, 1e-5, 1.5e-5,  2e-5])
cb_cam.set_ticklabels(['0', '0.5', '1', '1.5', '2'])
cb_cam.set_label(r'Diabatic heating ($\times10^{-5}$ K s$^{-1}$)',
                 fontsize=fontsize)

# ============================================================
# Overall title
# ============================================================

fig.suptitle(
    'Diabatic Heating in Ridge Blocks across the Model Hierarchy',
    fontsize=18,
    y=0.97
)

fig.subplots_adjust(
    left=0.07,
    right=0.96,
    top=0.90,
    bottom=0.06
)

plt.show()


# %%
