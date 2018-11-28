#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:51:47 2018

This script will help you to compute the tuning curve of all the the data from head direction cells of one animal,
and it will save the data in a .hdf file located in ./data_output 
You should run this script before main_autocorrelation,py and main.py

@author: grvite
"""
import pandas as pd
from functions_mt import *
import numpy as np

#define number of bins for the tuning curve computation 
nbins = 60
#number of columns for subplots
col= 15
#indicate if it is the first time that you run this script
first_run = False
#Select directory where you have all the folders of your animals data
data_directory = '../data_read_t/'
# Define output directory for the plots
output = '../plots'
# Define directory to save the .hdf data
hdf_dir = '../data_output'

"""
1. Load and accomodate data
"""

#get information about the animals and the sessions
dic = getcodes(data_directory)

if first_run == True:
    #Select directory where you have all the .ang and .pos files of your animals data
    pos_dir = './data_read_t/positions'
    #data handling -temporary- 
    files_managment(dic, data_directory, pos_dir)

"""
2. Compute the tuning curve for all the HD neurons
"""

#Create a dataframe with the information of the tuning curves of all the neurons
abins = np.linspace(0, 2*np.pi, nbins)
df_tuning = pd.DataFrame(index = abins[0:-1])

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
3. Compute widths
"""
#smooth data to make easy to cumpute the width of the tuning curve
df_smooth = df_tuning.rolling(window =5, win_type='gaussian', center=True, min_periods = 1).mean(std = 5.0)

df_tun_widths = pd.DataFrame(index= name_cols, columns = ['width'])
for i in name_cols:
    print(i)
    array = df_smooth[i].values 
    df_tun_widths.loc[i, 'width'] = width_gaussian (nbins, array)
    
"""
4. Plot Results
"""

#A. Example
##Tuning curve
label= 'width = ' + str( df_tun_widths.loc[neuron].values[0])
neuron = 'Mouse12-120806-23'
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='linen', edgecolor = 'linen', alpha=0.5)
fig = df_smooth[neuron].plot(color='chocolate', figsize=(11,8))
fig.set_ylabel("Firing rate (spk/s)", fontsize=14); 
fig.set_xlabel("Direction (radians)", fontsize=14)
fig.axhline(df_smooth[neuron].max()/2, color='red')
fig.set_title(neuron, fontsize=14)
fig.arrow(2.3, 11.57, 1.5, 14, ls="--")
fig.annotate("", xy=(2.3, 11.57), xytext=(1.5, 14), arrowprops=dict(arrowstyle="->"))
fig.text(0.42, 0.87, label, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox = props)

plt.savefig(output + '/tun_plot_' + neuron + '.pdf', bbox_inches = 'tight')


##Polar plot
phase = df_smooth.index.values
figp = plt.figure(figsize=(8,8))
ax = figp.add_subplot(111, projection='polar')
ax.plot(phase, df_smooth[neuron], color ='darkorange')
ax.set_title(neuron)
plt.savefig(output + '/tun_polar_' + neuron + '.pdf', bbox_inches = 'tight')
    
plt.plot(phase, df_smooth[neuron], color ='darkorange', projection = 'polar')

#B. General

#Determine the number of raws
raws = int(np.ceil(len(df_smooth.columns)/col))
#Get columns names
name_cols = df_tuning.columns
fig = plt.figure(figsize=(12,160))
for c,num in zip(name_cols, range(1,len(df_smooth.columns)+1)):
    ax = fig.add_subplot(raws,col,num)
    ax.plot(df_smooth[c], color ='darkorange')
    #ax.set_xlabel('radians')
    ax.set_title(c)
plt.tight_layout()
plt.savefig( output + '/tuning_plot_' + '.pdf')


#Polar plots
#smooth data, we change the parameters just for presentation since the smoothing does not allow to close the loop for polar plots. Consider that no measure is taken from polar plo
df_polar = df_tuning.rolling(window = 5, win_type='bartlett', center=True, min_periods = 1).mean(std = 8.0)
phase = df_polar.index.values
fig = plt.figure(figsize=(40,250))
for c,num in zip(name_cols, range(1,len(df_polar.columns)+1)):
    ax = fig.add_subplot(raws, col, num, projection='polar')
    ax.plot(phase, df_polar[c], color ='darkorange')
    #ax.set_xlabel('radians')
    ax.set_title(c)
#plt.tight_layout(h_pad=0.1)
plt.savefig(output + '/tuning_polar_' + '.pdf')




"""
5. Save data in .hdf format
"""
df_tuning.to_hdf(hdf_dir + '/df_tuning.hdf', 'tuning')
df_tun_widths.to_hdf(hdf_dir + '/df_tuning_widths.hdf', 'widhts_t')
    
