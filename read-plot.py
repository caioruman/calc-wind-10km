import numpy as np
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt
import os
import argparse

from datetime import date, datetime, timedelta
from calendar import monthrange

from glob import glob
from rpn.rpn import RPN
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn import level_kinds

from netCDF4 import Dataset
import time

'''
  - 
'''

parser=argparse.ArgumentParser(description='Separates the wind profiles based on SHF', formatter_class=argparse.RawTextHelpFormatter)
#parser.add_argument("-op", "--opt-arg", type=str, dest='opcional', help="Algum argumento opcional no programa", default=False)
parser.add_argument("anoi", type=int, help="Ano", default=0)
parser.add_argument("anof", type=int, help="Anof", default=0)
parser.add_argument("exp", type=str, help="exp", default=0)
args=parser.parse_args()

datai = args.anoi
dataf = args.anof
exp = args.exp

def main():

  # CSV data folder: /pixel/project01/cruman/ModelData/cPanCan_011deg_ERA5_90lvl_rerun/CSV_cPanCan_011deg_ERA5_90lvl_rerun
  folder = "/pixel/project01/cruman/ModelData/{0}/CSV_{0}".format(exp)  

  season = [['DJF', (12, 1, 2)], ['JJA', (6, 7, 8)]]  

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
  for i, name in enumerate(stnames):

    for sname, smonths in season:      

      for year in range(datai, dataf+1):
        
        # open CSV
        aux_path = "{0}/{1}".format(folder, sname)

          #days = monthrange(year, month)

          #for day in range(days[1]):
          #00000_Iqaluit_NU_19790206_pressure_pos.csv
        wind_pos, wind_neg = readDataCSV(aux_path, name, year, smonths, 'wind')
        temp_pos, temp_neg = readDataCSV(aux_path, name, year, smonths, 'temp')
        pho_pos, pho_neg = readDataCSV(aux_path, name, year, smonths, 'density')
        press_pos, press_neg = readDataCSV(aux_path, name, year, smonths, 'pressure')
        gzwind_pos, gzwind_neg = readDataCSV(aux_path, name, year, smonths, 'gz_wind')
        gztemp_pos, gztemp_neg = readDataCSV(aux_path, name, year, smonths, 'gz_temp')

        # All the fields are read, now do stuff

        # Option 1: Calculate the mean profile of the variables and save to a txt to plot later

        # Option 2: Detect the inversions using the Khan Paper Method. Save deltaZ, deltaT and %. Also find the level of Max wind value. Also save the Date the time of the inversion.        


def readDataCSV(aux_path, name, year, smonths, var):

  aux_p = []
  aux_n = []

  for month in smonths:
    file_path_pos = "{0}/{2}{3:02d}/{1}_{2}{3:02d}*_{4}_pos.csv".format(aux_path, name, year, month, var)
    file_path_neg = "{0}/{2}{3:02d}/{1}_{2}{3:02d}*_{4}_neg.csv".format(aux_path, name, year, month, var)
    
    aux_p.extend(glob(file_path_pos))
    aux_n.extend(glob(file_path_neg))
    
  df_p = pd.concat((pd.read_csv(f, index_col=0) for f in np.sort(aux_p)), ignore_index=True)
  df_n = pd.concat((pd.read_csv(f, index_col=0) for f in np.sort(aux_n)), ignore_index=True)

  return df_p, df_n



if __name__ == "__main__":
  main()
