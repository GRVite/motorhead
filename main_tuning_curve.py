#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:51:47 2018

This script will help you to compute the tuning curve of all the the data from head direction cells of one animal,
and it will save the data in a .hdf file located in ./data_output 

@author: grvite
"""
import pandas as pd
from functions_mt import *
import numpy as np


"""
0. Determine parameters
"""
nbins = 60
#number of columns for subplots
col= 5
#indicate if it is the first time that you run this script
first_run = False

"""
1. Load and accomodate data
"""

#Select directory where you have all the folders of your animals data
data_directory = './data_read_t/'
#get information about the animals and the sessions
dic = getcodes(data_directory)

if first_run == True:
    #Select directory where you have all the .ang and .pos files of your animals data
    pos_dir = './data_read_t/positions'
    #data handling -temporary- 
    files_managment(dic, data_directory, pos_dir)

#Create a dataframe with the information of the tuning curves of all the neurons
abins = np.linspace(0, 2*np.pi, nbins)
df_tuning = pd.DataFrame(index = abins[0:-1])

#spikes, shank, hd_spikes, wake_ep, sws_ep, rem_ep = data_hand ( './data_read_t/Mouse17/Mouse17-130129/', 'Mouse17-130129')     #del
        
for mouse in dic.keys():
    for ID in dic[mouse]:
        path = data_directory + mouse + '/' + ID
        print('the path is', path)
        try:
            spikes, shank, hd_spikes = data_hand (path, ID)
        except KeyError:
            print('problem with spikes for {0}'.format(ID))
            break
        wake_ep = loadEpoch(path, 'wake')
        
        """
        3. Compute the tuning curve for all the HD neurons
        """
        #Make a list of your neurons numbers
        indx = list(hd_spikes.keys())
        print(indx)
        keys = list(map(str, indx))
        name_cols = list(map(lambda x: ID + '-' + x, keys))
        # read angular data
        ang = np.genfromtxt(path + '/' + ID + '_ang.txt')
        # transform it to Tsd
        ang = nts.TsdFrame(d = ang[:,1], t = ang[:,0], time_units = 's')
        for i, n in zip(indx, name_cols):    
            print (i,n)
            tuning = tuneit (hd_spikes, ang, wake_ep, i, nbins)
            df_tuning[n] = pd.Series(index = abins[0:-1], data = np.squeeze(tuning.values)) 

#Sort values by columns
df_tuning = df_tuning.sort_index(axis=1)

"""
4. Plot Results
"""
#smooth data to make easy to cumpute the width of the tuning curve
df_smooth = df_tuning.rolling(window =15, win_type='gaussian', center=True, min_periods = 1).mean(std = 5.0)

#Determine the number of raws
raws = int(np.ceil(len(df_smooth.columns)/col))

#Get columns names
name_cols = df_tuning.columns

fig = plt.figure(figsize=(12,48))
for c,num in zip(name_cols, range(1,len(df_smooth.columns)+1)):
    ax = fig.add_subplot(raws,col,num)
    ax.plot(df_smooth[c], color ='darkorange')
    #ax.set_xlabel('radians')
    ax.set_title(c)
plt.tight_layout()
plt.savefig('./plots/' + 'tuning_plot_' + '.pdf')


#Polar plots

#smooth data, we change the parameters just for presentation since the smoothing does not allow to close the loop for polar plots. Consider that no measure is taken from polar plo
df_polar = df_tuning.rolling(window = 5, win_type='bartlett', center=True, min_periods = 1).mean(std = 8.0)

phase = df_polar.index.values
fig = plt.figure(figsize=(25,160))
for c,num in zip(name_cols, range(1,len(df_polar.columns)+1)):
    ax = fig.add_subplot(raws, col, num, projection='polar')
    ax.plot(phase, df_polar[c], color ='darkorange')
    #ax.set_xlabel('radians')
    ax.set_title(c)
plt.tight_layout(h_pad=0.1)
plt.savefig('./plots/' + 'tuning_polar_' + '.pdf')



"""
5. Compute widths
"""
df_tun_widths = pd.DataFrame(index= name_cols, columns = ['width'])
for i in name_cols:
    array = df_smooth[i].values 
    df_tun_widths.loc[i, 'width'] = width_gaussian (nbins, array)

#Save data
df_tuning.to_hdf('./data_output/df_tuning.hdf', 'tuning')
df_tun_widths.to_hdf('./data_output/df_tuning_widths.hdf', 'widhts_t')
    
