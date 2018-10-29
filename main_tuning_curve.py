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

"""
0. Determine parameters
"""
nabins = 60
#number of columns for subplots
col= 5

"""
1. Load and accomodate data
"""
ID= 'Mouse12-120806'
data_directory = './data_read/'
spikes, shank, hd_spikes, wake_ep, sws_ep, rem_ep = data_hand (data_directory, ID)

"""
2. Determine the position in the arena
"""
mouse_position = det_pos (data_directory, 'orange', 'a')

"""
3. Compute the tuning curve for all the HD neurons
"""
#Make a list of your neurons numbers
indx = list(hd_spikes.keys())
keys = list(map(str, indx))
name_cols = list(map(lambda x: ID + '_n_' + x, keys))

#Create a dataframe with the information of the tuning curves of all the neurons
abins = np.linspace(0, 2*np.pi, nabins+1)
df_tuning = pd.DataFrame(columns = name_cols)
for i, n in zip(indx, name_cols):    
    print (i,n)
    my_data, tuning_curve = tuneit (hd_spikes, wake_ep, mouse_position, i, nabins, 'a')
    df_tuning[n] = pd.Series(index = abins[0:-1], data = tuning_curve.flatten())

#Interpolate in case of a nan value
df_tuning = df_tuning.interpolate(method = 'from_derivatives')

#branch 2
df_tuning.loc[2*np.pi]=np.nan
df_tuning.iloc[-1] = df_tuning.iloc[0]
df_smooth = df_tuning.rolling(window = 15, win_type='gaussian', center=True, min_periods = 1, closed ='both').mean(std = 5.0)

#branch1
df_tmp = df_tuning.iloc[::-1]
df_tmp = df_tmp.drop(df_tmp.index.values[0], axis=0)
frames = [df_tuning, df_tmp]
df_tmp =pd.concat(frames) 
df_tmp = df_tmp.drop(df_tmp.index.values[-1], axis=0)
frames = [df_tmp, df_tuning]
df_tmp =pd.concat(frames) 
df_tmp=df_tmp.rolling(window = 15, win_type='bartlett', center=True, min_periods = 1).mean(std = 5.0)
df_tmp=df_tmp.iloc[59:120]


#flip values
arrayf = np.flipud (df_tuning.values)
#delete last value to not have a values repeated when you append it 
array_=np.delete(arrayf,59, axis=0)
array =  np.append(array_, df_tuning.values, axis=0)
array_=np.delete(arrayf,0, axis=0)
array =  np.append(array, array_, axis=0)
df_tun_smooth=pd.DataFrame(data=array, columns=name_cols)
#Smooth it
df_tun_smooth = df_tun_smooth.rolling(window = 15, win_type='gaussian', center=True, min_periods = 1).mean(std = 5.0) #We need to smooth the data to make the computation of the width easier
df_tun_smooth=df_tun_smooth.iloc[60:120,:]
df_tun_smooth.set_index(df_tuning.index)



"""
#Polar trick 
"""
df_tmp = df_smooth
#Determine the number of raws of your plot
raws = int(np.ceil(len(df_tmp.columns)/col))
phase = df_tmp.index.values
fig = plt.figure(figsize=(24,18))
for c,num in zip(name_cols, range(1,len(df_tmp.columns)+1)):
    ax = fig.add_subplot(raws, col, num, projection='polar')
    ax.plot(phase, df_tmp[c], color ='darkorange')
    #ax.set_xlabel('radians')
    ax.set_title(c)
plt.tight_layout()




"""
4. Plot Results
"""

#Determine the number of raws
raws = int(np.ceil(len(df_tun_smooth.columns)/col))

fig = plt.figure(figsize=(12,8))
for c,num in zip(name_cols, range(1,len(df_tun_smooth.columns)+1)):
    ax = fig.add_subplot(raws,col,num)
    ax.plot(df_tun_smooth[c], color ='darkorange')
    #ax.set_xlabel('radians')
    ax.set_title(c)
plt.tight_layout()
plt.savefig('./plots/' + 'tuning_plot_' + '.pdf')


#Polar plots
#add extra row for clossing the loop
df_tun_smooth = df_tun_smooth.append(df_tun_smooth.loc[0,:], ignore_index=True)
#Define phase values
phase = np.linspace(0, 2*np.pi, nabins)
#add extra value for clossing the loop
phase = np.append(phase, 0)
fig = plt.figure(figsize=(24,18))
for c,num in zip(name_cols, range(1,len(df_tun_smooth.columns)+1)):
    ax = fig.add_subplot(raws, col, num, projection='polar')
    ax.plot(phase, df_tun_smooth[c], color ='darkorange')
    #ax.set_xlabel('radians')
    ax.set_title(c)
plt.tight_layout()
plt.savefig('./plots/' + 'tuning_polar_' + '.pdf')



"""
5. Compute widths
"""
df_tun_widths = pd.DataFrame(index= name_cols, columns = ['width'])
for i in name_cols:
    array = df_tun_smooth[i].values 
    df_tun_widths.loc[i, 'width'] = width_gaussian (60, array)

#Save data
df_tuning.to_hdf('./data_output/df_tuning.hdf', 'tuning')
df_tun_widths.to_hdf('./data_output/df_tuning_widths.hdf', 'widhts_t')
    
