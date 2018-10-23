#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will help you to compute the autocorrelation of all the data 
from head direction cells of one animal, and it will save the data in a .hdf file 
located in ./data_output 

Created on Fri Oct  5 10:53:34 2018
@author: grvite
"""

import pandas as pd
from functions_mt import *

"""
0. Determine parameters
"""
#Determine the bin size and the number of bins for the autocorrelation
bin_size = 10
nb_bins = 100


"""
1. Load and accomodate data
"""
#Determine mouse ID
ID= 'Mouse12-120806'
spikes, shank, hd_spikes, wake_ep, sws_ep, rem_ep = data_hand(data_directory, ID)
#. Make a list of your neurons numbers
index = list(hd_spikes.keys())
keys = list(map(str, index))
name_id = list(map(lambda x: ID + '_n_' + x, keys))
#Determine the real times for the bin size and number of bins selected
times= np.arange(0, bin_size*(nb_bins+1), bin_size) - (nb_bins*bin_size)/2
# Create a pd to store the autocorrelation data from your different epochs
df = pd.DataFrame(index = times, columns = pd.MultiIndex.from_product([name_cols, eplist]))
#make lists of epochs for the loops and the index of your pandas dataframe 
eplist = ['wake', 'sws', 'rem']
epochs = [wake_ep, sws_ep, rem_ep]
#Dataframe for storing widths
df_widths = pd.DataFrame(index = name_id, columns = eplist)


"""
2. Compute results
"""
for ep, epl in zip (epochs, eplist):   
    for i, n in zip (index, name_ind):
        #compute width
        meanfiring = meanfiring_f (hd_spikes, i, ep)
        #compute autocorrelation
        aucor, width_auto = plotautco (hd_spikes, i, meanfiring, ep, epl, bin_size, numberofbins, 'a')
        #store the values of the autocorrelation
        df[n, epl]= aucor
        #store the width
df_widths[epl][n]= width_auto


"""
3. Make plots
"""
from matplotlib.pyplot import hlines as hlines
def a_subplots(epoch, name_id, col= 5):
    #Determine the number of raws
    raws = int(np.ceil(len(name_id)/col))
    fig = plt.figure(figsize=(12,8))
    for i, num in zip(name_id, range(1,len(name_id)+1)):
        ax = fig.add_subplot(raws,col,num)
        ax.plot(df[i, epoch], color ='brown')
        ax.set_title(i)
    plt.tight_layout()
    plt.savefig('./plots/' + 'autocorrelogram_plot_' + epoch + '.pdf')
    plt.show()

for i in eplist: a_subplots(i, name_id)

"""
4. Save data in .hdf format
"""

df_wake.to_hdf('./data_output/df_autocorrelation_wake.hdf', 'df_autocorrelation_wake')
df_sws.to_hdf('./data_output/df_autocorrelation.hdf', 'df_autocorrelation_sws')
df_rem.to_hdf('./data_output/df_autocorrelation.hdf', 'df_autocorrelatio_remn')
df_widths.to_hdf('./data_output/df_autocorrelation_widths.hdf', 'widhts_a')