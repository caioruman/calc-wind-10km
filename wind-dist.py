import numpy as np
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt
import os
import argparse

from datetime import date, datetime, timedelta

from glob import glob
from rpn.rpn import RPN
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn import level_kinds

from netCDF4 import Dataset
import time

'''
  - Calculates the probability distribution for the wind using model levels
  - Calculates the weibull distribution and return the parameters

  - Open the RPN files for each month
  - Group the data in one array
  - Get the timeseries for each station
  - Divide the timeseries between SHF > 0 and SHF < 0 
  - K-mean clustering of those two timeseries
'''

parser=argparse.ArgumentParser(description='Separates the wind profiles based on PBL Stability', formatter_class=argparse.RawTextHelpFormatter)
#parser.add_argument("-op", "--opt-arg", type=str, dest='opcional', help="Algum argumento opcional no programa", default=False)
parser.add_argument("anoi", type=int, help="Ano", default=0)
parser.add_argument("anof", type=int, help="Anof", default=0)
parser.add_argument("exp", type=str, help="Simulation Name", default=0)
args=parser.parse_args()

datai = args.anoi
dataf = args.anof
exp = args.exp


def main():
  
  #exp = "cPanCan_011deg_ERA5_80lvl_rerun"
      
  folder = "/home/cruman/Documents/Scripts/calc-wind-10km"
  folder = "/home/cruman/projects/rrg-sushama-ab/cruman/Data/Phase2"
#  Cedar
#  main_folder = "/home/cruman/projects/rrg-sushama-ab/teufel/{0}".format(exp)
#  Beluga
  main_folder = "/home/cruman/projects/rrg-sushama-ab/cruman/storage_model/Output/{0}".format(exp)

  # constants
  Rd = 287  # Gas constant of dry air
  g = 9.80665 # gravity

  # to be put in a loop later. 
  for year in range(datai, dataf+1):    

    for month in range(1,13):        

      arq_dm_month = np.sort(glob("{0}/Samples/{1}_{2}{3:02d}/dm*".format(main_folder, exp, year, month)))
      arq_pm_month = np.sort(glob("{0}/Samples/{1}_{2}{3:02d}/pm*".format(main_folder, exp, year, month)))      

      m = 0
      for arq_dm, arq_pm in zip(arq_dm_month, arq_pm_month):
        m += 1

        name = "71925__Cambridge_Bay__NT_YCB"
        if os.path.exists("{0}/CSV_{5}/{1}/{2}{3:02d}/{1}_{2}{3:02d}{4:02d}_wind_neg.csv".format(folder, name, year, month, m, exp)):
          print("{0}-{1}-{2} already calculated. skipping.".format(year, month, m))
          continue

      # Reading SHF. File shape: (time, soil type, lat, lon)
        with RPN(arq_pm) as r:
          print("Opening file {0}".format(arq_pm))
          shf = np.squeeze(r.variables["AHF"][:])
          # I want the grid average, getting the field number 5
          shf = shf[:,4,:,:]
                  
          tskin = np.squeeze(r.variables["TSKN"][:])
          tskin = tskin[:,4,:,:]
          
          if 'lons2d' not in locals():
            lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()  
              

        # Reading Wind and Temperature (time, level, lat, lon)
        with RPN(arq_dm) as r:
          
          print("Opening file {0}".format(arq_dm))

          #mslp = np.squeeze(r.variables['PN'][:])
#          mslp = r.variables['PN'][:]
          #print(mslp)
          #print(mslp.shape)
          #sys.exit()
          uu = np.squeeze(r.variables["UU"][:])
          vv = np.squeeze(r.variables["VV"][:])                        

          tt = np.squeeze(r.variables["TT"][:])
          t2m = tt[:,-1,:,:]+273.15
          tt = tt[:,:-1,:,:]+273.15

          # position of elements: (time, z, x, y)
          # top altitude is the 0 array index
          gz = r.variables['GZ'][:]          
          gz_0 = gz[:,-1,:,:]
          # removing the last level (surface)
          gz = gz[:,:-1,:,:]

          # spliting between tt/hu and uu levels
          gz_tt = gz[:,1::2,:,:]
          gz_uu = gz[:,::2,:,:]

          hu = r.variables['HU'][:] # specific humidity
          hu_0 = hu[:,-1,:,:]
          hu = hu[:,:-1,:,:]

          mslp = np.squeeze(r.variables['PN'][:])
                  
          utest = r.variables["UU"]
          #ttest = r.variables["TT"]
          #print([lev for lev in utest.sorted_levels])
          levels = [lev for lev in utest.sorted_levels]
          levels = levels[:-1]
          levels = [format(x, '.4f') for x in levels]
          #dates = [str(d) for d in utest.sorted_dates]
          dates = np.array(r.variables["UU"].sorted_dates)

          #t2m = r.get_first_record_for_name("TT", label="PAN_ERAI_DEF")        
                  
        # Converting to m/s
        uv = np.sqrt(np.power(uu, 2) + np.power(vv, 2))/1.944
        uv_10 = uv[:,-1,:,:]
        uv = uv[:,:-1,:,:]

        # Virtual Temperature
        tv = tt*(1 + 0.61*(hu/(1-hu)))
        tv_0 = t2m*(1 + 0.61*(hu_0/(1-hu_0)))    

        mslp_a = np.zeros_like(tt)
        for t in range(mslp.shape[0]):
            mslp_a[:,t] += mslp[t]

        p = mslp_a/(np.exp(gz_tt*g/(Rd*tv)))

        # estimating the air density
        pho = p/(Rd*tv)
        pho_0 = mslp/(Rd*tv_0)
              
        lats = []
        lons = []
        stnames = []

        stations = open('DatFiles/stations.txt', 'r')
        for line in stations:
          aa = line.replace("\n", '').split(';')
          if (aa[0] != "#"):      
            lats.append(float(aa[3]))
            lons.append(float(aa[5]))
            stnames.append(aa[1].replace(',',"_"))

        # looping throught all the stations
        for lat, lon, name in zip(lats, lons, stnames):
          os.system('mkdir -p {1}/CSV_{4}/{3}/{0}{2:02d}'.format(year, folder, month, name, exp))
          # Extract the info from the grid 
          i, j = geo_idx([lat, lon], np.array([lats2d, lons2d]))

          # Separating the negative and positive values, to apply to the wind
          neg_shf = np.less_equal(shf[:, i, j], 0)

          neg_pho = pho[neg_shf, :, i, j]
          pos_pho = pho[~neg_shf, :, i, j]

          neg_pho_0 = pho_0[neg_shf, i, j]
          pos_pho_0 = pho_0[~neg_shf, i, j]
          

          neg_p = p[neg_shf, :, i, j]
          pos_p = p[~neg_shf, :, i, j]

          neg_gz_tt = gz_tt[neg_shf, :, i, j]
          pos_gz_tt = gz_tt[~neg_shf, :, i, j]

          neg_gz_uu = gz_uu[neg_shf, :, i, j]
          pos_gz_uu = gz_uu[~neg_shf, :, i, j]

          neg_wind = uv_10[neg_shf, i, j]
          pos_wind = uv_10[~neg_shf, i, j]

          neg_t2m = t2m[neg_shf, i, j]
          pos_t2m = t2m[~neg_shf, i, j]

          neg_stemp = tskin[neg_shf, i, j]
          pos_stemp = tskin[~neg_shf, i, j]

          neg_wind_model = uv[neg_shf, :, i, j]
          pos_wind_model = uv[~neg_shf, :, i, j]

          neg_tt_model = tt[neg_shf, :, i, j]
          pos_tt_model = tt[~neg_shf, :, i, j]

          neg_dates = dates[neg_shf]
          pos_dates = dates[~neg_shf]

          # temps
          saveDataframe(folder, name, year, month, m, levels, neg_dates, pos_dates, neg_tt_model, pos_tt_model, 'temp', None, [neg_t2m, pos_t2m], [neg_stemp, pos_stemp])

          # Wind
          saveDataframe(folder, name, year, month, m, levels, neg_dates, pos_dates, neg_wind_model, pos_wind_model, 'wind', [neg_wind, pos_wind])

          # GZ Wind
          saveDataframe(folder, name, year, month, m, levels, neg_dates, pos_dates, neg_gz_uu, pos_gz_uu, 'gz_wind')

          # GZ Temp
          saveDataframe(folder, name, year, month, m, levels, neg_dates, pos_dates, neg_gz_tt, pos_gz_tt, 'gz_temp')

          # Density
          saveDataframe(folder, name, year, month, m, levels, neg_dates, pos_dates, neg_pho, pos_pho, 'density')

          # Pressure
          saveDataframe(folder, name, year, month, m, levels, neg_dates, pos_dates, neg_p, pos_p, 'pressure', None, None, None, [neg_pho_0, pos_pho_0])

          
""" 
          df1 = pd.DataFrame(data=neg_wind_model, columns=levels)
          df2 = pd.DataFrame(data=pos_wind_model, columns=levels)

          df1 = df1.assign(Dates=neg_dates)
          df2 = df2.assign(Dates=pos_dates)

          df1.to_csv("{0}/CSV/{1}/{2}{3:02d}/{1}_{2}{3:02d}{4:02d}_windpress_neg.csv".format(folder, name, year, month, m))
          df2.to_csv("{0}/CSV/{1}/{2}{3:02d}/{1}_{2}{3:02d}{4:02d}_windpress_pos.csv".format(folder, name, year, month, m))

          df1 = pd.DataFrame(data=neg_tt_model, columns=levels)
          df2 = pd.DataFrame(data=pos_tt_model, columns=levels)

          df1 = df1.assign(SurfTemp=neg_stemp)
          df1 = df1.assign(T2M=neg_t2m)
          df1 = df1.assign(UV=neg_wind)

          df2 = df2.assign(SurfTemp=pos_stemp)
          df2 = df2.assign(T2M=pos_t2m)
          df2 = df2.assign(UV=pos_wind)

          df1 = df1.assign(Dates=neg_dates)
          df2 = df2.assign(Dates=pos_dates)

          df1.to_csv("{0}/CSV/{1}/{2}{3:02d}/{1}_{2}{3:02d}{4:02d}_neg.csv".format(folder, name, year, month, m))
          df2.to_csv("{0}/CSV/{1}/{2}{3:02d}/{1}_{2}{3:02d}{4:02d}_pos.csv".format(folder, name, year, month, m)) """

def saveDataframe(folder, name, year, month, m, levels, neg_dates, pos_dates, data_neg, data_pos, df_name, UV=None, T2M=None, Skin=None, pho=None):

  df1 = pd.DataFrame(data=data_neg, columns=levels)
  df2 = pd.DataFrame(data=data_pos, columns=levels)

  df1 = df1.assign(Dates=neg_dates)
  df2 = df2.assign(Dates=pos_dates)

  if T2M is not None:
    df1 = df1.assign(T2M=T2M[0])
    df2 = df2.assign(T2M=T2M[1])

  if UV is not None:
    df1 = df1.assign(UV10=UV[0])
    df2 = df2.assign(UV10=UV[1])

  if Skin is not None:
    df1 = df1.assign(Tskin=Skin[0])
    df2 = df2.assign(Tskin=Skin[1])

  if pho is not None:
    df1 = df1.assign(Tskin=pho[0])
    df2 = df2.assign(Tskin=pho[1])

  df1.to_csv("{0}/CSV_{6}/{1}/{2}{3:02d}/{1}_{2}{3:02d}{4:02d}_{5}_neg.csv".format(folder, name, year, month, m, df_name, exp))
  df2.to_csv("{0}/CSV_{6}/{1}/{2}{3:02d}/{1}_{2}{3:02d}{4:02d}_{5}_pos.csv".format(folder, name, year, month, m, df_name, exp))        

def geo_idx(dd, dd_array, type="lat"):
  '''
    search for nearest decimal degree in an array of decimal degrees and return the index.
    np.argmin returns the indices of minium value along an axis.
    so subtract dd from all values in dd_array, take absolute value and find index of minimum.
    
    Differentiate between 2-D and 1-D lat/lon arrays.
    for 2-D arrays, should receive values in this format: dd=[lat, lon], dd_array=[lats2d,lons2d]
  '''
  if type == "lon" and len(dd_array.shape) == 1:
    dd_array = np.where(dd_array <= 180, dd_array, dd_array - 360)

  if (len(dd_array.shape) < 2):
    geo_idx = (np.abs(dd_array - dd)).argmin()
  else:
    if (dd_array[1] < 0).any():
      dd_array[1] = np.where(dd_array[1] <= 180, dd_array[1], dd_array[1] - 360)

    a = abs( dd_array[0]-dd[0] ) + abs(  np.where(dd_array[1] <= 180, dd_array[1], dd_array[1] - 360) - dd[1] )
    i,j = np.unravel_index(a.argmin(), a.shape)
    geo_idx = [i,j]

  return geo_idx


if __name__ == "__main__":
  main()
