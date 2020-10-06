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
  stnames_comma = []
  stations = open('DatFiles/stations.txt', 'r')
  for line in stations:
    aa = line.replace("\n", '').split(';')
    if (aa[0] != "#"):     
      lats.append(float(aa[3]))
      lons.append(float(aa[5]))
      stnames_comma.append(aa[1])
      stnames.append(aa[1].replace(',',"_"))      

  for name_s, name in zip(stnames_comma, stnames):
    for sname, smonths in season:

      wind_pos, wind_neg = readDataSoundings(sfolder, name_s, smonths, datai, dataf)


def readDataSoundings(folder, name, months, datai, dataf):
  
  dt = datetime(datai, 1, 1, 0, 0)
  date_f = datetime(dataf, 12, 31, 12, 0)

  ff = np.sorted(glob.glob('{0}/{1}/soundings_*_????.csv'.format(folder, name)))

  for f in ff:
    df = pd.read_csv(f, index_col=0)

    # Loop throught the soundings
    while dt <= date_f:

      df_aux = df.query("Year == {0} and Month == {1} and Day == {2} and Hour == {3}".format(dt.year, dt.month, dt.day, dt.hour))

      if not df_aux.empty:

        # subtracting the first height level from the other levels
        df_aux['HGHT'] = df['HGHT'] - df['HGHT'][0]
        # removing indices where height > 600
        ind = df_aux[df_aux.HGHT > 600].index
        df_aux = df_aux.drop(ind)

        print(df_aux)
        sys.exit()

        


  return wind_pos, wind_neg

def create_lists_preplot(centroids_n, centroids_p, histo_n, histo_p, perc_n, perc_p, numb_n, numb_p):
  cent = []
  histo = []
  perc = []
  shf = []
  numb = []

  if (perc_n[0] > perc_n[1]):
    k = 0
    j = 1
  else:
    k = 1
    j = 0

  cent.append(centroids_n[k])
  cent.append(centroids_n[j])

  histo.append(histo_n[k])
  histo.append(histo_n[j])

  perc.append(perc_n[k])
  perc.append(perc_n[j])

  shf.append('SHF-')
  shf.append('SHF-')      

  numb.append(numb_n[k])
  numb.append(numb_n[j])

  if (perc_p[0] > perc_p[1]):
    k = 0
    j = 1
  else:
    k = 1
    j = 0

  cent.append(centroids_p[k])
  cent.append(centroids_p[j])

  histo.append(histo_p[k])
  histo.append(histo_p[j])

  perc.append(perc_p[k])
  perc.append(perc_p[j])

  shf.append('SHF+')
  shf.append('SHF+')

  numb.append(numb_p[k])
  numb.append(numb_p[j])

  return cent, histo, perc, shf, numb

def readDataCSV(aux_path, name, smonths, var, UV=False, T2M=False, pho=False):

  aux_p = []
  aux_n = []

  for year in range(datai, dataf+1):
    for month in smonths:
      file_path_pos = "{0}/{1}/{2}{3:02d}/{1}_{2}{3:02d}*_{4}_pos.csv".format(aux_path, name, year, month, var)
      file_path_neg = "{0}/{1}/{2}{3:02d}/{1}_{2}{3:02d}*_{4}_neg.csv".format(aux_path, name, year, month, var)

      aux_p.extend(glob(file_path_pos))
      aux_n.extend(glob(file_path_neg))
    
  df_p = pd.concat((pd.read_csv(f, index_col=0) for f in np.sort(aux_p)), ignore_index=True)
  df_n = pd.concat((pd.read_csv(f, index_col=0) for f in np.sort(aux_n)), ignore_index=True)

  df_n = df_n.drop(columns=['Dates'])
  df_p = df_p.drop(columns=['Dates'])

  if T2M:
    df_n = df_n.drop(columns=['T2M'])
    df_p = df_p.drop(columns=['T2M'])

    df_n = df_n.drop(columns=['Tskin'])
    df_p = df_p.drop(columns=['Tskin'])

  if UV:
    df_n = df_n.drop(columns=['UV10'])
    df_p = df_p.drop(columns=['UV10'])  

  if pho:
    df_n = df_n.drop(columns=['Pho'])
    df_p = df_p.drop(columns=['Pho'])

  return df_p, df_n

def kmeans_probability(df, df_tmp):
  '''
    For now fixed at 2 clusters
    returns: Array of the centroids, the two histograms and % of each group
  '''
  kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
        
  # Getting the location of each group.
  pred = kmeans.predict(df)
  labels = np.equal(pred, 0)
  centroids = kmeans.cluster_centers_

  # Converting to numpy array
  df_a = np.array(df)
  df_tmp = np.array(df_tmp)

  # Dividing between the 2 clusters
  df_0 = df_a[labels,:]
  df_1 = df_a[~labels,:]

  df_tmp_0 = df_tmp[labels,:]
  df_tmp_1 = df_tmp[~labels,:]

  profileT_0 = np.mean(df_tmp_0, axis=0)
  profileT_1 = np.mean(df_tmp_1, axis=0)

  aux_grid = np.linspace(223.15,293.15,80)
  #print()
  #histT_0 = calc_histogram(df_tmp_0, 223.15, 293.15)
  #histT_1 = calc_histogram(df_tmp_1, 223.15, 293.15)

  histT_0 = calc_kerneldensity(df_tmp_0, aux_grid)
  histT_1 = calc_kerneldensity(df_tmp_1, aux_grid) 

  # Getting the probability distribution. Bins of 0.5 m/s
 # hist_0 = calc_histogram(df_0)
 # hist_1 = calc_histogram(df_1)

  # Getting the probability distribution. Kernel Density  
  aux_grid = np.linspace(0,40,80)
  hist_0 = calc_kerneldensity(df_0, aux_grid)
  hist_1 = calc_kerneldensity(df_1, aux_grid)  

  #plot_stuff(hist_0, centroids[0])
  #plot_stuff(hist_1, centroids[1])
  #sys.exit()

  #centroids = kmeans.cluster_centers_, [hist_0, hist_1], [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]]

  perc = [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]]
  numb = [df_0.shape[0], df_1.shape[0]]

  return df_tmp_0, df_0, df_tmp_1, df_1, centroids, [profileT_0, profileT_1], [histT_0, histT_1], [hist_0, hist_1], perc, numb

def calc_kerneldensity(df, aux_grid):
  hist_aux = []
  for i in range(0,df.shape[1]):
    kde_skl = KernelDensity(bandwidth=0.4)
    #aux = np.array(df_n['1000.0'])
    aux = np.copy(df[:,i])    
    kde_skl.fit(aux[:, np.newaxis])
    log_pdf = kde_skl.score_samples(aux_grid[:, np.newaxis])
    hist_aux.append(np.exp(log_pdf)*100)
    
    

  return hist_aux

def calc_histogram(df, irange=0, frange=40.25):

  hist_l = []
  bins = np.arange(irange,frange+1,1)  
  for i in range(0, df.shape[1]):    
    hist, bins = np.histogram(df[:,i], bins=bins)
    hist_l.append(hist*100/sum(hist))
    print(sum(hist*100/sum(hist)))

  return np.asarray(hist_l)

def plot_wind_seasonal(levels, centroids, histo, perc, shf, datai, dataf, name, period, numb, wind=False):

  y = levels
  #x = np.arange(0,40,1)  
  if wind:
    x = np.linspace(0,40,80)
    vmin=0
    vmax=40
    var = 'wind'
    lvl = np.arange(0,22,3)
    
  else:
    # for temperature
    #x = np.arange(223.15,293.15,1)
    x = np.linspace(223.15,293.15,80)
    print(x.shape)
    vmin=223
    vmax=293
    var = 'tmp'  
    lvl = np.arange(0,13,2)

  X, Y= np.meshgrid(x, y)
   
  v = np.arange(vmin, vmax+1, 2) 
  fig = plt.figure(figsize=[28,16])

  for k, letter in zip(range(0,4), ['a', 'b', 'c', 'd']):
    subplt = '22{0}'.format(k+1)
    plt.subplot(subplt)
    
    #print(histo[k])
    #print(histo[k])
    CS = plt.contourf(X, Y, histo[k], cmap='cmo.haline', extend='max', levels=lvl)
    #CS.set_clim(vmin, vmax)
    #plt.gca().invert_yaxis()
    #print(centroids[k])
    plt.plot(centroids[k], y, color='white', marker='o', lw=4, markersize=10, markeredgecolor='k')
    if (k % 2):
      CB = plt.colorbar(CS, extend='both')
      CB.ax.tick_params(labelsize=20)
    
    if wind:
      plt.xlim(0,25)
      plt.xticks(np.arange(0,25,5), fontsize=20)
    else:
      plt.xticks(np.arange(225,291,5), fontsize=20)
    
    plt.ylim(0,280)
    
    plt.yticks(np.arange(0,280,10), fontsize=20)    
    plt.title('({0}) {1:2.2f} % {2} | #: {3}'.format(letter, perc[k], shf[k], numb[k]), fontsize='20')
  plt.tight_layout()
  plt.savefig('Images/{0}_{1}{2}_{3}_{4}.png'.format(name, datai, dataf, period, var), pad_inches=0.0, bbox_inches='tight')
  plt.close()
  

  return None       

if __name__ == "__main__":
  main()
