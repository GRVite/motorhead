#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:53:34 2018
@author: grvite
"""

import pandas as pd
from functions_mt import *

#1. Load and accomodate data
ID= 'Mouse12-120806'
spikes, shank, hd_spikes, wake_ep, sws_ep, rem_ep = data_hand(data_directory, ID)

#Lists of epochs
epochs = [wake_ep, sws_ep, rem_ep]
eplist = ['wake', 'sws', 'rem']

#Make a list of your neurons numbers
index = list(hd_spikes.keys())
keys = list(map(str, index))
name_cols = list(map(lambda x: ID + '_n_' + x, keys))

# Create individual dataframes to store the autocorrelation data from your different epochs
df_wake = pd.DataFrame(columns = name_cols)
df_sws = pd.DataFrame(columns = name_cols)
df_rem = pd.DataFrame(columns = name_cols)
lista_df = [df_wake, df_sws, df_rem]

#Dataframe for widths
df_widths = pd.DataFrame(index = eplist, columns = name_cols)

#Determine the bin size and the number of bins for the autocorrelation
bin_size=1
numberofbins=50

#Read mean firing data
df_meanfiring = pd.read_csv('../data_read/df_meanfiring.csv')

#Create dataframes for all epochs with the information of the autocorrelation and all neurons
for ep, epl, df in zip (epochs, eplist , lista_df):   
    for i, n in zip (index, name_cols):  
        meanfiring = df_meanfiring[n].iloc[0]
        aucor, width_auto = plotautco (hd_spikes, i, meanfiring, ep, epl, bin_size, numberofbins, 'a')
        df[n] = pd.Series(data = aucor.flatten())
        df_widths[n][epl]= width_auto

#Save data
df_wake.to_hdf('../data_output/df_autocorrelation_wake.hdf', 'df_autocorrelation_wake')
df_sws.to_hdf('../data_output/df_autocorrelation.hdf', 'df_autocorrelation_sws')
df_rem.to_hdf('/data_output/df_autocorrelation.hdf', 'df_autocorrelatio_remn')
df_widths.to_csv('/data_output/df_autocorrelation_widths.csv')