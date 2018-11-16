#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will help you to compute the autocorrelation of all the data 
from head direction cells of one animal, and it will save the data in a .hdf file 
located in ./data_output 

You are supposed to run main_tuninc_curve.py first, since this script takes data from it

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
#Select directory where you have all the folders of your animals data
data_directory = './data_read_t/'
#Make a list of your animals
#Sometimes, you have a .DS_Store, and you do not want it in your list.
main = [i for i in os.listdir(data_directory) if i != '.DS_Store']

#Make a dict of all your sessions
dic = {}
for n, m in zip(range(len(main)), main):
    dire = data_directory + m
    dire = {i for i in os.listdir(dire) if i != '.DS_Store'}
    dic [n] = dire

#Move files one level up from Analysis folder
for i in dic.keys():
    print(i)
    for j in dic[i]:
        path = data_directory + main[i] + '/' + j + '/'
        print('the path is', path)
        for file in os.listdir(path + '/Analysis'):
            os.rename (path + '/Analysis/' + file, path + file)

#import codes of neurons from df_tuning.hdf file
name_cols = pd.read_hdf('./data_output/df_tuning.hdf', 'tuning', start = 0, stop = 0).columns

#Determine the real times for the bin size and number of bins selected
times= np.arange(0, bin_size*(nb_bins+1), bin_size) - (nb_bins*bin_size)/2
#make lists of epochs for the loops and the index of your pandas dataframe 
eplist = ['wake', 'sws', 'rem']
# Create a pd to store the autocorrelation data from your different epochs
df = pd.DataFrame(index = times, columns = pd.MultiIndex.from_product([name_cols, eplist]))
#Dataframe for storing widths
df_widths = pd.DataFrame(columns = eplist)
df_var = pd.DataFrame(columns = eplist)

for mouse in dic.keys():
    for ID in dic[mouse]:
        path = data_directory + main[mouse] + '/' + ID 
        print('the path is', path)
        spikes, shank, hd_spikes, wake_ep, sws_ep, rem_ep = data_hand (path, ID)
        # Make a list of your neurons numbers
        index = list(hd_spikes.keys())
        epochs = [wake_ep, sws_ep, rem_ep]
        for i in index:
            for ep, epl in zip (epochs, eplist):  
                #compute width
                meanfiring = meanfiring_f (hd_spikes, i, ep)
                corr = corr_calc(hd_spikes, i, ep, bin_size, nb_bins)
                loci = ID + '_n_' + str(i)
                df[loci, epl] = pd.Series(index = times, data = corr.flatten()) 
                df_widths.loc [loci, epl] = width_corr (corr, nb_bins, bin_size, meanfiring, window = 7, stdv = 5.0)
                df_var.loc [loci, epl] = np.var(corr)

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

df.to_hdf('./data_output/df_autocorrelation.hdf', 'df_autocorrelation')
df_widths.to_hdf('./data_output/df_autocorrelation_widths.hdf', 'widhts_a')
df_var.to_hdf('./data_output/df_autocorrelation_var.hdf', 'var_a')