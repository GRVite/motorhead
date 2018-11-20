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
#Select directory where you have all the folders of your animals data
data_directory = './data_read_t/'
#Select directory where you have all the .ang and .pos files of your animals data
pos_dir = './data_read_t/positions'

"""
1. Load and accomodate data
"""


#get codes of animals, sessions and neurons

main, dic, neuronas = redata(data_directory, pos_dir)




#Determine the real times for the bin size and number of bins selected
times= np.arange(0, bin_size*(nb_bins+1), bin_size) - (nb_bins*bin_size)/2
#make lists of epochs for the loops and the index of your pandas dataframe 
eplist = ['wake', 'sws', 'rem', 'rem_pre', 'rem_post']
# Create a pd to store the autocorrelation data from your different epochs
df = pd.DataFrame(index = times, columns = pd.MultiIndex.from_product([name_cols, eplist]))
#Dataframe for storing widths

index = pd.MultiIndex(leve)

df_widths = pd.DataFrame(columns = eplist, index = pd.MultiIndex.from_product([main,sessions,neurons]))

df_var = pd.DataFrame(columns = eplist, index = pd.MultiIndex.from_product([main,sessions,neurons]))

arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))

dic_ocur = {}
for animal in main:
    dic_ocur[animal] = sum(animal in s for s in sessions)

#enumarate animals
labels_a=[]
mapping = {}
for i, v in enumerate(main):
    mapping[v] = str(i)* dic_ocur[v]
    for i in mapping[v]:
        labels_a.append(str(i)* dic_ocur[v])

multiplica dic_ocur por mapping

for i in main:
    for i in 
indi = pd.MultiIndex(levels=[main, sessions],
                     labels=[labels_a, labels_a], #Integers for each level designating which label at each location
       names=['animal', 'session'])
df= pd.DataFrame(index = indi)

for i in ID.keys():
    for session in ID[i]:
        path = data_directory + i + '/' + session
        print('the path is', path)
        spikes, shank, hd_spikes= data_hand (path, session)
        wake_ep 		= loadEpoch(path, 'wake')
        sleep_ep 		= loadEpoch(path, 'sleep')
        sws_ep 			= loadEpoch(path, 'sws')
        rem_ep 			= loadEpoch(path, 'rem')
        sws_ep = sleep_ep.intersect(sws_ep)
        rem_ep = sleep_ep.intersect(rem_ep)
        rem_pre = loadEpoch(path,'rem_pre')
        rem_post = loadEpoch(path,'rem_post')
        # Make a list of your neurons numbers
        index = list(hd_spikes.keys())
        for ind in index:
            for ep, epl in zip (epochs, eplist):  
                #compute width
                meanfiring = meanfiring_f (hd_spikes, ind, ep)
                corr = corr_calc(hd_spikes, ind, ep, bin_size, nb_bins)
                print(i, session, 'n_' + str(ind))
                df_widths.loc [(i, session, 'n_' + str(ind)), epl] = width_corr (corr, nb_bins, bin_size, meanfiring, window = 7, stdv = 5.0)
                
for mouse, codem in zip(dic.keys(),range(len(main))):
    print(mouse[5:])
    s = list(dic['Mouse17'])
    s.sort
    print(s)
    for session in s:
        path = data_directory + main[codem] + '/' + mouse + '-' + session
        print('the path is', path)
        spikes, shank, hd_spikes= data_hand (path, mouse + '-' + session)
        wake_ep 		= loadEpoch(path, 'wake')
        sleep_ep 		= loadEpoch(path, 'sleep')
        sws_ep 			= loadEpoch(path, 'sws')
        rem_ep 			= loadEpoch(path, 'rem')
        sws_ep = sleep_ep.intersect(sws_ep)
        rem_ep = sleep_ep.intersect(rem_ep)
        rem_pre = loadEpoch(path,'rem_pre')
        rem_post = loadEpoch(path,'rem_post')
        # Make a list of your neurons numbers
        index = list(hd_spikes.keys())
        index.sort
        print(index)
        epochs = [wake_ep, sws_ep, rem_ep, rem_pre, rem_post]
        for i, neur in zip (index, neurons):
            print(i,neur)
            for ep, epl in zip (epochs, eplist):  
                #compute width
                meanfiring = meanfiring_f (hd_spikes, i, ep)
                corr = corr_calc(hd_spikes, i, ep, bin_size, nb_bins)
                loci = ID + '_n_' + str(i)
                df[loci, epl] = pd.Series(index = times, data = corr.flatten()) 
                df_widths.loc [(mouse, session, neur), epl] = width_corr (corr, nb_bins, bin_size, meanfiring, window = 7, stdv = 5.0)
                print(df_widths)
                df_var.loc [(mouse, session, neur), epl] [loci, epl] = np.var(corr)
           
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