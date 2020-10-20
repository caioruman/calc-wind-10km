import numpy as np
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt
import os
import argparse

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from calendar import monthrange

from glob import glob

import time

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

import cmocean
from scipy import interpolate

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

      df_wind, df_tmp = readDataSoundings(sfolder, name_s, smonths, datai, dataf)

      df_wind_inv = df_wind.query("deltaT > 0")
      df_wind_noInv = df_wind.query("deltaT < 0")

      df_tmp_inv = df_tmp.query("deltaT > 0")
      df_tmp_noInv = df_tmp.query("deltaT < 0")

      df_dates_inv = pd.DataFrame(df_tmp_inv['Dates'].copy())
      df_dates_noInv = pd.DataFrame(df_tmp_noInv['Dates'].copy())

      df_dates_noInv['Dates'] = pd.to_datetime(df_dates_noInv['Dates'])#, format='%Y%m%d %H:%M', errors='ignore')
      df_dates_inv['Dates'] = pd.to_datetime(df_dates_inv['Dates'])#, format='%Y%m%d %H:%M', errors='ignore')
      #df_dates_inv.rename(columns={"Dates": "Dates1"})
      #df_dates_noInv.rename(columns={"Dates": "Dates1"})

      print(df_dates_inv)

      df_wind_inv = df_wind_inv.drop(columns=['deltaT', 'Dates'])
      df_wind_noInv = df_wind_noInv.drop(columns=['deltaT', 'Dates'])

      df_tmp_inv = df_tmp_inv.drop(columns=['deltaT', 'Dates'])
      df_tmp_noInv = df_tmp_noInv.drop(columns=['deltaT', 'Dates'])      

      # Clustering analysis on the above data. Divide it by the deltaT column      
      # Columns of the dataframe: [300,275,250,225,200,175,150,125,100,75,50,25,10, deltaT, Dates]      
      df_tmp_0_noInv, df_wind_0_noInv, df_tmp_1_noInv, df_wind_1_noInv, centroids_NoInv, profileT_NoInv, histT_NoInv, hist_NoInv, perc_NoInv, numb_NoInv, labels_NoInv, deltaT_NoInv, hist_deltaT_NoInv = kmeans_probability(df_wind_noInv, df_tmp_noInv)
      df_tmp_0_inv, df_wind_0_inv, df_tmp_1_inv, df_wind_1_inv, centroids_inv, profileT_inv, histT_inv, hist_inv, perc_inv, numb_inv, labels_inv, deltaT_inv, hist_deltaT_inv = kmeans_probability(df_wind_inv, df_tmp_inv)

      # plot just to see the results

      levels = [300,275,250,225,200,175,150,125,100,75,50,25,10]
      levels = [500, 450, 400, 350, 325, 300, 280, 260, 240, 220, 189, 162, 139, 119, 102, 88, 76, 66, 57, 49, 42, 36, 31, 26, 22, 18, 14, 10]

      # Change the create_lists to sort the profile in another way.
      # 1st possibility: compare the wind between the surface and at higher levels. High wind = Shear driven or WSBL
      cent, histo, perc, inv, numb = create_lists_preplot(centroids_NoInv, centroids_inv, hist_NoInv, hist_inv, perc_NoInv, perc_inv, numb_NoInv, numb_inv, centroids_NoInv, centroids_inv)
      
      plot_wind_seasonal(levels, cent, histo, perc, inv, datai, dataf, name, sname, numb, True)
      
      cent, histo, perc, inv, numb = create_lists_preplot(profileT_NoInv, profileT_inv, histT_NoInv, histT_inv, perc_NoInv, perc_inv, numb_NoInv, numb_inv, centroids_NoInv, centroids_inv)

      plot_wind_seasonal(levels, cent, histo, perc, inv, datai, dataf, name, sname, numb)

      # Plotting deltaT
      cent, histo, perc, inv, numb = create_lists_preplot(deltaT_NoInv, deltaT_inv, hist_deltaT_NoInv, hist_deltaT_inv, perc_NoInv, perc_inv, numb_NoInv, numb_inv, centroids_NoInv, centroids_inv)

      plot_wind_seasonal(levels, cent, histo, perc, inv, datai, dataf, name, sname, numb, False, True)

      # Read the model data
      # Return only the data that match the soundings      
      wind_inv_90, wind_noInv_90 = readDataCSV(folder_90, name, smonths, 'wind', df_dates_inv, df_dates_noInv, True)
      temp_inv_90, temp_noInv_90 = readDataCSV(folder_90, name, smonths, 'temp', df_dates_inv, df_dates_noInv, False, True)

      wind_inv_80, wind_noInv_80 = readDataCSV(folder_80, name, smonths, 'wind', df_dates_inv, df_dates_noInv, True)
      temp_inv_80, temp_noInv_80 = readDataCSV(folder_80, name, smonths, 'temp', df_dates_inv, df_dates_noInv, False, True)

      # Plot the model data

      df_tmp_0_noInv90, df_wind_0_noInv90, df_tmp_1_noInv90, df_wind_1_noInv90, centroids_NoInv90, profileT_NoInv90, histT_NoInv90, hist_NoInv90, perc_NoInv90, numb_NoInv90, labels_NoInv90, deltaT_NoInv90, hist_deltaT_NoInv90 = kmeans_probability(wind_noInv_90, temp_noInv_90)
      df_tmp_0_inv90, df_wind_0_inv90, df_tmp_1_inv90, df_wind_1_inv90, centroids_inv90, profileT_inv90, histT_inv90, hist_inv90, perc_inv90, numb_inv90, labels_inv90, deltaT_inv90, hist_deltaT_inv90 = kmeans_probability(wind_inv_90, temp_inv_90)

      df_tmp_0_noInv80, df_wind_0_noInv80, df_tmp_1_noInv80, df_wind_1_noInv80, centroids_NoInv80, profileT_NoInv80, histT_NoInv80, hist_NoInv80, perc_NoInv80, numb_NoInv80, labels_NoInv80, deltaT_NoInv80, hist_deltaT_NoInv80 = kmeans_probability(wind_noInv_80, temp_noInv_80)
      df_tmp_0_inv80, df_wind_0_inv80, df_tmp_1_inv80, df_wind_1_inv80, centroids_inv80, profileT_inv80, histT_inv80, hist_inv80, perc_inv80, numb_inv80, labels_inv80, deltaT_inv80, hist_deltaT_inv80 = kmeans_probability(wind_inv_80, temp_inv_80)

      levels = wind_inv_90.columns
      cent, histo, perc, inv, numb = create_lists_preplot(centroids_NoInv90, centroids_inv90, hist_NoInv90, hist_inv90, perc_NoInv90, perc_inv90, numb_NoInv90, numb_inv90, centroids_NoInv90, centroids_inv90)
      
      plot_wind_seasonal(levels, cent, histo, perc, inv, datai, dataf, name, sname, numb, True, False, 'model_90')
      
      cent, histo, perc, inv, numb = create_lists_preplot(profileT_NoInv90, profileT_inv90, histT_NoInv90, histT_inv90, perc_NoInv90, perc_inv90, numb_NoInv90, numb_inv90, centroids_NoInv90, centroids_inv90)

      plot_wind_seasonal(levels, cent, histo, perc, inv, datai, dataf, name, sname, numb, False, False, 'model_90')

      # apply the labels from the soundings to the model data
      # 
      

      
      # 

def readDataCSV(aux_path, name, smonths, var, df_dates_inv, df_dates_noInv, UV=False, T2M=False, pho=False):

  aux_inv = []
  aux_noInv = []

  for year in range(datai, dataf+1):
    for month in smonths:
      file_path_pos = "{0}/{1}/{2}{3:02d}/{1}_{2}{3:02d}*_{4}_pos.csv".format(aux_path, name, year, month, var)
      file_path_neg = "{0}/{1}/{2}{3:02d}/{1}_{2}{3:02d}*_{4}_neg.csv".format(aux_path, name, year, month, var)

      aux_inv.extend(glob(file_path_pos))
      aux_noInv.extend(glob(file_path_neg))
    
  df_inv = pd.concat((pd.read_csv(f, index_col=0) for f in np.sort(aux_inv)), ignore_index=True)
  df_noInv = pd.concat((pd.read_csv(f, index_col=0) for f in np.sort(aux_noInv)), ignore_index=True)

  #Merging the two
  df_full = pd.concat([df_inv, df_noInv])

  #print(df_inv.head())
  # Getting rid of the microseconds part.
  df_full['Dates'] = df_full['Dates'].astype('datetime64[s]')
#  df_noInv['Dates'] = df_noInv['Dates'].astype('datetime64[s]')
  
  # Getting the model values that correspond to the soundings
  df_inv = pd.merge(df_full, df_dates_inv, on=['Dates'], how='inner')
  df_noInv = pd.merge(df_full, df_dates_noInv, on=['Dates'], how='inner')
  # Fix the merbe above. The Dates are slightly different, so it might not be working because of that: 1980-01-02 23:00:00.000006 vs 1980-01-02 23:00:00
  
  if T2M:
    df_noInv = df_noInv.drop(columns=['T2M'])
    df_inv = df_inv.drop(columns=['T2M'])

    df_noInv = df_noInv.drop(columns=['Tskin'])
    df_inv = df_inv.drop(columns=['Tskin'])

  if UV:
    df_noInv = df_noInv.drop(columns=['UV10'])
    df_inv = df_inv.drop(columns=['UV10'])  

  if pho:
    df_noInv = df_noInv.drop(columns=['Pho'])
    df_inv = df_inv.drop(columns=['Pho'])


  df_noInv = df_noInv.drop(columns=['Dates'])
  df_inv = df_inv.drop(columns=['Dates'])

  return df_inv, df_noInv

def readDataSoundings(folder, name, months, datai, dataf):
  
  year_i = datai  

  #ff = np.sort(glob('{0}/{1}/soundings_*_????.csv'.format(folder, name)))
  #ff = [glob('{0}/{1}/soundings_*_{2}.csv'.format(folder, name, x)) for x in range(datai,dataf+1)]

  # model levels plus extra levels higher up
  levels = [500, 450, 400, 350, 325, 300, 280, 260, 240, 220, 189, 162, 139, 119, 102, 88, 76, 66, 57, 49, 42, 36, 31, 26, 22, 18, 14, 10]

  # sounding levels. The vertical resolution isnt good. The linear interpolation will be strange
  #levels = [300,275,250,225,200,175,150,125,100,75,50,25,10]

  df_wind = pd.DataFrame(columns=levels + ['deltaT'] + ['Dates'])
  df_tmp = pd.DataFrame(columns=levels + ['deltaT'] + ['Dates'])
  
  i = 0
  e = 0  
  for y in range(datai, dataf+1):
    f = glob('{0}/{1}/soundings_*_{2}.csv'.format(folder, name, y))    
    df = pd.read_csv(f[0], index_col=0)
    print(y, f)
    for m in months:
      
      dt = datetime(y, m, 1, 0, 0)
      date_f = dt + relativedelta(months=+1)      

      # Loop throught the soundings
      while dt < date_f:
        #print(dt)

        #print(i)
        df_aux = df.query("Year == {0} and Month == {1} and Day == {2} and Hour == {3}".format(dt.year, dt.month, dt.day, dt.hour))

        if not df_aux.empty:

          #if np.isnan(df_aux['TEMP']).any():
          #  print(df_aux['HGHT'])
          #  print(df_aux['TEMP'])
          #  print(df_aux['HGHT'].values[0])
          #  print(df['HGHT'] - df['HGHT'].values[0] + 10)
          # subtracting the first height level from the other levels
          # removing indices where height > 600
          ind = df_aux[df_aux.HGHT > 1000].index
          df_aux = df_aux.drop(ind)

          ind = df_aux[df_aux.HGHT <= 0].index
          df_aux = df_aux.drop(ind)

          df_aux = df_aux.dropna()

          if (df_aux.empty):
            dt = dt + timedelta(hours=12)
            print('emtpy df')
            e += 1
            continue
            

          #print(df_aux['HGHT'].values)
          new_height = df_aux['HGHT'] - df_aux['HGHT'].values[0] + 10
          #print(new_height.values)

          #print(df_aux)
          #if np.isnan(df_aux['TEMP'].values[:10]).any() or np.isnan(new_height).any():
            #print(df_aux['TEMP'].values[:10])
            #print(new_height)
            #print(dt)
          #  dt = dt + timedelta(hours=12)
            #sys.exit()
         #   print('NaN found')
          #  print(df_aux['TEMP'].values[:10])
           # print(new_height.values)
            #continue

          if len(df_aux['TEMP']) < 4:
            dt = dt + timedelta(hours=12)
            print('Less than 4 items')
            e += 1
            continue

          try:
            aux_tmp = interpolateData(df_aux['TEMP'], levels, new_height.values) + 273.15
            aux_wind = interpolateData(df_aux['SKNT'], levels, new_height.values)/1.944
          except ValueError as err:
            dt = dt + timedelta(hours=12)
            print(err)
            e += 1
            #print(df_aux['SKNT'])
            #print(df_aux['HGHT'])            
            continue

          #aux_inv = df_aux['TEMP'].values[1] - df_aux['TEMP'].values[0]
#          print(aux_inv)
          aux_inv = aux_tmp[-14] - aux_tmp[-1] # Around ~90m
          #print(levels)
          #print(aux_tmp)
          #print(df_aux['HGHT'])
          #print(df_aux['TEMP']+273.15)
          #sys.exit()
          #aux_inv = aux_tmp[19] - aux_tmp[0] # Around ~200m          
        
          df_wind.loc[i] = aux_wind.tolist() + [aux_inv] + [dt.strftime('%Y-%m-%d %H:%M:%S')]
          #df2 = pd.DataFrame(aux_wind.tolist() + [aux_inv] + [dt], columns=levels + ['deltaT'] + ['Dates'])
          #print(aux_wind.tolist() + [aux_inv] + [dt])
          #pd.concat([df_wind, df2])
          df_tmp.loc[i] = aux_tmp.tolist() + [aux_inv] + [dt.strftime('%Y-%m-%d %H:%M:%S')]
          # next steps:
          # Do a try() catch() statement to catch errors and jump to the next date

        dt = dt + timedelta(hours=12)
        i += 1     

  print("{0} soundings discarded due to being empty or not having enough data, from a total of {1}".format(e, i))

  return df_wind, df_tmp

def interpolateData(data, new_levels, height):

  #data_interp = np.zeros([len(new_levels)])

  # for each date in the array
  #for i in range(height.shape[0]):
    #height[i,:] -= height[i,-1]+1
  f = interpolate.interp1d(height, data, kind='linear')
  data_interp = f(new_levels)

  return data_interp

def create_lists_preplot(centroids_noInv, centroids_inv, histo_noInv, histo_inv, perc_noInv, perc_inv, numb_noInv, numb_inv, windp_noInv, windp_inv):
  #profileT_NoInv, profileT_inv, histT_NoInv, histT_inv, perc_NoInv, perc_inv, numb_NoInv, numb_inv, centroids_NoInv, centroids_inv)
  cent = []
  histo = []
  perc = []
  shf = []
  numb = []

  aux_noInv_case1 = windp_noInv[0][-5] - windp_noInv[0][0]
  aux_noInv_case2 = windp_noInv[1][-5] - windp_noInv[1][0]
  aux_Inv_case1 = windp_inv[0][-5] - windp_inv[0][0]
  aux_Inv_case2 = windp_inv[1][-5] - windp_inv[1][0]

  # The no inversion case will always come first
  # I'm assuming that the wind shear is greater for the WSBL and Shear driven PBL
  if (aux_noInv_case1 > aux_noInv_case2):
    k = 0
    j = 1
  else:
    k = 1
    j = 0

  cent.append(centroids_noInv[k])
  cent.append(centroids_noInv[j])

  histo.append(histo_noInv[k])
  histo.append(histo_noInv[j])

  perc.append(perc_noInv[k])
  perc.append(perc_noInv[j])

  shf.append('No Inversion')
  shf.append('No Inversion')      

  numb.append(numb_noInv[k])
  numb.append(numb_noInv[j])

  if (aux_Inv_case1 > aux_Inv_case2):
    k = 0
    j = 1
  else:
    k = 1
    j = 0

  cent.append(centroids_inv[k])
  cent.append(centroids_inv[j])

  histo.append(histo_inv[k])
  histo.append(histo_inv[j])

  perc.append(perc_inv[k])
  perc.append(perc_inv[j])

  shf.append('With Inversions')
  shf.append('With Inversions')

  numb.append(numb_inv[k])
  numb.append(numb_inv[j])

  return cent, histo, perc, shf, numb

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

  df_deltat_0 = df_tmp_0.copy()
  df_deltat_1 = df_tmp_1.copy()
  
  for i in range(df_tmp_0.shape[1]-1, -1, -1):
    df_deltat_0[:,i] = df_tmp_0[:,i] - df_tmp_0[:,0]

  for i in range(df_tmp_0.shape[1]-1, -1, -1):
    df_deltat_1[:,i] = df_tmp_1[:,i] - df_tmp_1[:,0]  

  profileT_0 = np.mean(df_tmp_0, axis=0)
  profileT_1 = np.mean(df_tmp_1, axis=0)

  profileDeltaT_0 = np.mean(df_deltat_0, axis=0)
  profileDeltaT_1 = np.mean(df_deltat_1, axis=0)

  aux_grid = np.linspace(223.15,293.15,80)
  #print()
  #histT_0 = calc_histogram(df_tmp_0, 223.15, 293.15)
  #histT_1 = calc_histogram(df_tmp_1, 223.15, 293.15)

  histT_0 = calc_kerneldensity(df_tmp_0, aux_grid)
  histT_1 = calc_kerneldensity(df_tmp_1, aux_grid) 

  aux_grid = np.linspace(-15,15,80)

  histDeltaT_0 = calc_kerneldensity(df_deltat_0, aux_grid)
  histDeltaT_1 = calc_kerneldensity(df_deltat_1, aux_grid) 

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

  return df_tmp_0, df_0, df_tmp_1, df_1, centroids, [profileT_0, profileT_1], [histT_0, histT_1], [hist_0, hist_1], perc, numb, labels, [profileDeltaT_0, profileDeltaT_1], [histDeltaT_0, histDeltaT_1]

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

def plot_wind_seasonal(levels, centroids, histo, perc, shf, datai, dataf, name, period, numb, wind=False, deltaT=False, cname=''):

  y = levels
  #x = np.arange(0,40,1)  
  if wind:
    x = np.linspace(0,40,80)
    vmin=0
    vmax=40
    var = 'wind'
    lvl = np.arange(0,22,3)
  elif deltaT:
    x = np.linspace(-15,15,80)
    vmin=-15
    vmax=15
    var = 'deltaT'
    lvl = np.arange(0,22,3)
    print(np.max(histo[0]))
  else:
    # for temperature
    #x = np.arange(223.15,293.15,1)
    x = np.linspace(223.15,293.15,80)
    print(x.shape)
    vmin=223
    vmax=293
    var = 'tmp'  
    lvl = np.arange(0,22,3)

  X, Y= np.meshgrid(x, y)
   
  #v = np.arange(vmin, vmax+1, 2) 
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
    elif deltaT:
      plt.xlim(-15,10)
      plt.xticks(np.arange(-10,11,2), fontsize=20)
    else:
      plt.xticks(np.arange(225,291,5), fontsize=20)
    
    plt.ylim(0,400)
    
    plt.yticks(np.arange(0,280,10), fontsize=20)    
    plt.title('({0}) {1:2.2f} % {2} | #: {3}'.format(letter, perc[k], shf[k], numb[k]), fontsize='20')
  plt.tight_layout()
  plt.savefig('Images/{0}_{1}{2}_{3}_{4}_{5}.png'.format(name, datai, dataf, period, var, cname), pad_inches=0.0, bbox_inches='tight')
  plt.close()
  

  return None       

if __name__ == "__main__":
  main()
