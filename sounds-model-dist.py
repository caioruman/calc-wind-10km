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

import time

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

import cmocean

'''
  - Read the soundings data 
  - Read both model data
  - PBL clustering separation with the 90lvl data
  - Using the labels from above, apply to the 80lvl and soundings

  - Or
  - PBL clustering separation on the soundings data
  - Using the labels of that, apply to the model data 
'''

parser=argparse.ArgumentParser(description='Separates the wind profiles based on SHF', formatter_class=argparse.RawTextHelpFormatter)
#parser.add_argument("-op", "--opt-arg", type=str, dest='opcional', help="Algum argumento opcional no programa", default=False)
parser.add_argument("anoi", type=int, help="Ano", default=0)
parser.add_argument("anof", type=int, help="Anof", default=0)
args=parser.parse_args()

datai = args.anoi
dataf = args.anof


def main():  

  sfolder = '/pixel/project01/cruman/Data/Soundings'
  mfolder = '/pixel/project01/cruman/ModelData'
  
  exp90 = 'cPanCan_011deg_ERA5_90lvl_rerun'
  exp80 = 'cPanCan_011deg_ERA5_80lvl_rerun'

  folder_90 = "{0}/{1}/CSV_{1}".format(mfolder, exp90)
  folder_80 = "{0}/{1}/CSV_{1}".format(mfolder, exp80)
  
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

  print(stnames)

if __name__ == "__main__":
  main()
