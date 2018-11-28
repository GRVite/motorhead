#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will help you to compute the autocorrelation of all the data 
from head direction cells of one animal, and it will save the data in a .hdf file 
located in ./data_output 

You are supposed to run main_tuning_curve first.

df_widhts[df_widths.index.str.contains('Mouse12')]


Created on Fri Oct  5 10:53:34 2018
@author: grvite
"""

import pandas as pd
from functions_mt import *

"""
0. Determine parameters
"""
# Determine the bin size and the number of bins for the autocorrelation
bin_size = 20
nb_bins = 200

# Select directory where you have all the .ang and .pos files of your animals data
pos_dir = './data_read_t/positions'

#Select directory where you have all the folders of your animals data
data_directory = '../data_read_t/'
# Define output directory for the plots
output = '../plots'
# Define directory to save the .hdf data
hdf_dir = '../data_output'

"""
1. Load and accomodate data
"""

# Get codes of the animals, sessions
dic = getcodes (data_directory)

#import codes of neurons from df_tuning.hdf file
neurons = pd.read_hdf(hdf_dir + '/df_tuning.hdf', 'tuning', start = 0, stop = 0).columns

#Determine the real times for the bin size and number of bins selected
times= np.arange(0, bin_size*(nb_bins+1), bin_size) - (nb_bins*bin_size)/2

# Make lists of epochs for the loops and the index of your pandas dataframe 
eplist = ['wake', 'sws', 'rem', 'rem_pre', 'rem_post']

"""
2. Compute autocorrelation
"""

# Create a dataframe to store the autocorrelation data from your different epochs
df = pd.DataFrame(index = times, columns = pd.MultiIndex.from_product([neurons, eplist]))

# Create a dataframe for storing smoothed data
df_smooth =  pd.DataFrame(index = times, columns = pd.MultiIndex.from_product([neurons, eplist]))

for mouse in dic.keys():
    for session in dic[mouse]:
        path = data_directory + mouse + '/' + session
        print('the path is', path)
        try:
            spikes, shank, hd_spikes = data_hand (path, session)
        except KeyError:
            print('problem with spikes for {0}'.format(session))
            break
        #load epochs data
        wake_ep 		= loadEpoch(path, 'wake')
        sleep_ep 		= loadEpoch(path, 'sleep')
        sws_ep 			= loadEpoch(path, 'sws')
        rem_ep 			= loadEpoch(path, 'rem')
        sws_ep = sleep_ep.intersect(sws_ep)
        rem_ep = sleep_ep.intersect(rem_ep)
        rem_pre = loadEpoch(path,'rem_pre')
        rem_post = loadEpoch(path,'rem_post')
        epochs = [wake_ep, sws_ep, rem_ep, rem_pre, rem_post]
        for neuron in list(hd_spikes.keys()):
            for ep, epl in zip (epochs, eplist):  
                #determine the mean firing rate
                meanfiring = meanfiring_f (hd_spikes, neuron, ep)
                #compute autocorrelation
                aucorr = corr_calc(hd_spikes, neuron, ep, bin_size, nb_bins)
                #store it
                df[session + '-' + str(neuron), epl] = pd.Series(index = times, data = aucorr.flatten()) 
                smooth = smooth_corr(aucorr, nbins, bin_size, meanfiring, window = 7, stdv = 5.0, plot = False)            
                #smooth data and store it
                df_smooth[session + '-' + str(neuron), epl] = pd.Series(index = times, data = smooth.flatten())

"""
3. Calculate widths
"""
# Create a dataframe for storing the widths              
df_widths = pd.DataFrame(index= neurons, columns = eplist)
for n in neurons:
    for i in eplist: 
        width = calc_width (df_smooth[n, i], nb_bins, binsize)
        print(n)
        df_widths.loc [n, i] = width
        
           
"""
4. Make plots
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
    plt.savefig(ouput + 'autocorrelogram_plot_' + epoch + '.pdf')
    plt.show()

for i in eplist: a_subplots(i, name_id)

"""
5. Save data in .hdf format
"""
df.to_hdf(hdf_dir + '/df_autocorrelation.hdf', 'df_autocorrelation')
df_smooth.to_hdf(hdf_dir + '/df_smooth_autocorrelation.hdf', 'df_smooth_autocorrelation')
df_widths.to_hdf(hdf_dir + '/df_autocorrelation_widths.hdf', 'widhts_a')
df_var.to_hdf(hdf_dir + '/df_autocorrelation_var.hdf', 'var_a')