#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:14:25 2018

This script computes the speed of the head direction signal. It reads the data
from main_tuning_curve and main_autocorrelation, calculates the ratio and plots the results. 

@author: grvite
"""

import pandas as pd
import numpy as np
from functions_mt import *

#Select directory where you have all the folders of your animals data
data_directory = './data_read_t/'

#get codes of animals and sesssions
dic = getcodes(data_directory)

#Read widths of the tuning curves
tun = pd.read_hdf('./data_output/df_tuning_widths.hdf')
lista = []
for i in tun.index.values:
    lista.append(((i.split('-')[0], i.split('-')[1], i.split('-')[2])))
index = pd.MultiIndex.from_tuples(lista, names=['animal', 'session', 'neuron'])
tun.set_index(index, inplace = True)

#Read widths of the autocorrelograms
aut = pd.read_hdf('./data_output/df_autocorrelation_widths.hdf')
aut.set_index(index, inplace = True)

#Create dataframe to store the ratio data
ratio = pd.DataFrame(index = index, columns = aut.columns)
eplist = eplist = ['wake', 'sws', 'rem', 'rem_pre', 'rem_post']

#Calculate the ratio
for ep in eplist: ratio[ep] = tun['width'].div(aut[ep])

#Sort values by wake epoch 
ratio = ratio.sort_values('wake')

"""
    General
"""
from matplotlib.pyplot import *

#Sort values by wake epoch result = result.sort_values('wake')

# Plot general results
eplist = ['wake', 'sws', 'rem']

fig = figure(figsize=(14,10))
for i in eplist:
    plot(ratio[i].values, marker='o', label=i)
    #xticks(range(len(aut.index)+1), [i[-2:] for i in  aut.index])
    xlabel('neuron')
    ylabel('speed (radians/s)')
    legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    title('Speed of the needle')
dir2s = './plots/' + 'speedftneedle' + 'general'
plt.savefig(dir2s, bbox_inches = 'tight')

# Plot results per animal
#select epochs
eplist = ['wake', 'sws', 'rem']
for ID in dic:
    fig = figure(figsize=(14,10))
    for i in eplist:
        plot(ratio.loc[ID, i].values, marker='o', label=i)
        #xticks(range(len(aut.index)+1), [i[-2:] for i in  aut.index])
        xlabel('neuron')
        ylabel('speed (radians/s)')
        legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        title('Speed of the needle')
    dir2s = './plots/' + 'speedftneedle' + 'general' + ID
    plt.savefig(dir2s, bbox_inches = 'tight')

"""
    REM and nREM vs Wake
"""
eplist = ['rem_pre', 'wake', 'rem_post']

fig = figure(figsize=(10,8))
for ID in dic:
    fig = figure(figsize=(14,10))
    for i in eplist:
        plot(ratio.loc[ID, i].values, marker='o', label=i)
        #xticks(range(len(aut.index)+1), [i[-2:] for i in  aut.index])
        xlabel('neuron')
        ylabel('speed (radians/s)')
        legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        title('Speed of the needle')
    dir2s = './plots/' + 'speedftneedle' + 'REMvsWAKE' + ID
    plt.savefig(dir2s, bbox_inches = 'tight')

#barplot

#Mean per group
data = ratio.groupby(level=[['animal']]).sum()/ratio.groupby(level=['animal']).count()
#General
data.plot.bar()
#Bar plot 'wake' vs 'rem'
data[['wake', 'rem']].plot.bar()
#BAr plot rempre vs rempost
data[['rem_pre', 'rem_post']].plot.bar()

eplist = ['wake', 'rem']

fig = figure(figsize=(14,10))'
for ID in dic:
    for i in eplist:
        bar(ratio[i].mean, label=i)
        #xticks(range(len(aut.index)+1), [i[-2:] for i in  aut.index])
        xlabel('neuron')
        ylabel('speed (radians/s)')
        legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        title('Speed of the needle')
dir2s = './plots/' + 'speedftneedle' + 'general'
plt.savefig(dir2s, bbox_inches = 'tight')

#Plot results
for i in eplist: 
        errors = result.loc[i].std()
        result.loc[i].plot.bar(title = "Speed of the needle")
        xticks(np.arange(len(names)), names)

