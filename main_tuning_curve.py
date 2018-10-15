#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:51:47 2018

@author: grvite
"""
import pandas as pd
from functions_mt import *


#1. Load and accomodate data
ID= 'Mouse12-120806'
data_directory = './data_read/'
spikes, shank, hd_spikes, wake_ep, sws_ep, rem_ep = data_hand (data_directory, ID)

#2. Determine the position in the arena
mouse_position = det_pos (data_directory, 'orange', 'a')

#3. Compute the tuning curve for all the HD neurons

#Make a list of your neurons numbers
index = list(hd_spikes.keys())
keys = list(map(str, index))
name_cols = list(map(lambda x: ID + '_n_' + x, keys))

#Create a dataframe with the information of the tuning curves of all the neurons
df_tuning = pd.DataFrame(columns = name_cols)
df_meanfiring = pd.DataFrame(index=[0], columns = name_cols)
nabins = 60
for i, n in zip(index, name_cols):    
    mean_firing_rate, my_data, tuning_curve, abins = tuneit (data_directory, hd_spikes, wake_ep, mouse_position, i, nabins, 'a')
    df_tuning[n] = pd.Series(index = abins[0:-1], data = tuning_curve.flatten())
    df_meanfiring[n].loc[0] = mean_firing_rate

#Interpolate in case of a nan value
df_tuning = df_tuning.interpolate(method = 'from_derivatives')

#Smooth it
df_tun_smooth = df_tuning.rolling(window = 15, win_type='gaussian', center=True, min_periods = 1).mean(std = 5.0) #We need to smooth the data to make the computation of the width easier

#compute widths
df_tun_widths = pd.DataFrame(index=[0], columns = name_cols)
for i in name_cols:
    array = df_tun_smooth[i].values 
    df_tun_widths[i].loc[0] = width_gaussian (60, array)

#Save data
df_tuning.to_hdf('./data_output/df_tuning.hdf', 'tuning')
df_meanfiring.to_csv('./data_output/df_mean_firing_rate.csv')
df_tun_widths.to_csv('./data_output/df_tuning_widths.csv')
    
