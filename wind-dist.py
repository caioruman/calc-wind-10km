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
args=parser.parse_args()

datai = args.anoi
dataf = args.anof


def main():
  
  exp = "cPanCan_011deg_ERA5_80lvl_rerun"
      
  folder = "/home/cruman/Documents/Scripts/calc-wind-10km"
#  Cedar
#  main_folder = "/home/cruman/projects/rrg-sushama-ab/teufel/{0}".format(exp)
#  Beluga
  main_folder = "/home/cruman/projects/rrg-sushama-ab/cruman/storage_model/Output/{0}".format(exp)

  # to be put in a loop later. 
  for year in range(datai, dataf+1):
    os.system('mkdir -p {1}/CSV/{0}'.format(year, folder))

    for month in range(1,13):  

      # Sample point for testing. Try Ile St Madeleine later: 47.391348; -61.850658
      #sp_lat = 58.107914
      #sp_lon = -68.421492
      #     
      name = "71925__Cambridge_Bay__NT_YCB"
      if os.path.exists("{0}/CSV/{4}/{1}_{2}{3:02d}_windpress_neg.csv".format(folder, name, year, month, year)):
        print("Month already calculated. skipping.")
        continue

      arq_dm_month = glob("{0}/Samples/{1}_{2}{3:02d}/dm*".format(main_folder, exp, year, month)).sort()
      arq_pm_month = glob("{0}/Samples/{1}_{2}{3:02d}/pm*".format(main_folder, exp, year, month)).sort()

      print(arq_dm_month)
      sys.exit()

      for arq_dm, arq_pm in zip(arq_dm_month, arq_pm_month):
      # Reading SHF. File shape: (time, soil type, lat, lon)
        with RPN(arq_pm) as r:
          print("Opening file {0}".format(arq_pm))
          shf = np.squeeze(r.variables["AHF"][:])
          # I want the grid average, getting the field number 5
          shf = shf[:,4,:,:]
                  
          tskin = np.squeeze(r.variables["TSKN"][:]) - 273.15
          tskin = tskin[:,4,:,:]
          
          if 'lons2d' not in locals():
            lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()  
              

        # Reading Wind and Temperature (time, level, lat, lon)
        with RPN(arq_dm) as r:
          print("Opening file {0}".format(arq_dm))
          uu = np.squeeze(r.variables["UU"][:])
          vv = np.squeeze(r.variables["VV"][:])                        

          tt = np.squeeze(r.variables["TT"][:])
          tt_10 = tt[:,-1,:,:]  
                  
          #utest = r.variables["UU"]
          #ttest = r.variables["TT"]
          #print([lev for lev in utest.sorted_levels])
          #print([lev for lev in ttest.sorted_levels])

          #t2m = r.get_first_record_for_name("TT", label="PAN_ERAI_DEF")        
                  
        uv = np.sqrt(np.power(uu, 2) + np.power(vv, 2))
        uv_10 = uv[:,-1,:,:]
        
        sys.exit()
              
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

          # Extract the info from the grid 
          i, j = geo_idx([lat, lon], np.array([lats2d, lons2d]))

          # Separating the negative and positive values, to apply to the wind
          neg_shf = np.less_equal(shf[:, i, j], 0)

          neg_wind = uv[neg_shf, i, j]
          pos_wind = uv[~neg_shf, i, j]

          neg_t2m = t2m[neg_shf, i, j]
          pos_t2m = t2m[~neg_shf, i, j]

          neg_stemp = surf_temp[neg_shf, i, j]
          pos_stemp = surf_temp[~neg_shf, i, j]

          neg_wind_press = uv_pressure[neg_shf, 10:, i, j]
          pos_wind_press = uv_pressure[~neg_shf, 10:, i, j]

          neg_tt_press = tt_press[neg_shf, 10:, i, j]
          pos_tt_press = tt_press[~neg_shf, 10:, i, j]

          neg_dates_d = dates_d[neg_shf]
          pos_dates_d = dates_d[~neg_shf]

          df1 = pd.DataFrame(data=neg_wind_press, columns=levels[10:])
          df2 = pd.DataFrame(data=pos_wind_press, columns=levels[10:])

          df1 = df1.assign(Dates=neg_dates_d)
          df2 = df2.assign(Dates=pos_dates_d)

          df1.to_csv("{0}/CSV_RCP/{4}/{1}_{2}{3:02d}_windpress_neg.csv".format(folder, name, year, month, year))
          df2.to_csv("{0}/CSV_RCP/{4}/{1}_{2}{3:02d}_windpress_pos.csv".format(folder, name, year, month, year))

          df1 = pd.DataFrame(data=neg_tt_press, columns=levels[10:])
          df2 = pd.DataFrame(data=pos_tt_press, columns=levels[10:])

          df1 = df1.assign(SurfTemp=neg_stemp)
          df1 = df1.assign(T2M=neg_t2m)
          df1 = df1.assign(UV=neg_wind)

          df2 = df2.assign(SurfTemp=pos_stemp)
          df2 = df2.assign(T2M=pos_t2m)
          df2 = df2.assign(UV=pos_wind)

          df1 = df1.assign(Dates=neg_dates_d)
          df2 = df2.assign(Dates=pos_dates_d)

          df1.to_csv("{0}/CSV_RCP/{4}/{1}_{2}{3:02d}_neg.csv".format(folder, name, year, month, year))
          df2.to_csv("{0}/CSV_RCP/{4}/{1}_{2}{3:02d}_pos.csv".format(folder, name, year, month, year))
  #      sys.exit()

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
