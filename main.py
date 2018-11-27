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


# Define directory where you have all the folders of your animals data
data_directory = './data_read_t'

# Define directory where are the .hdf derived from the computation of the tuning curve and the autocorrelation
dir_read = '../data_output'

# Define output directory
output = '../plots'


"""
1. Read Data
"""

#get codes of animals and sesssions
dic = getcodes(data_directory)

#Read widths of the tuning curves
tun = pd.read_hdf(dir_read + '/df_tuning_widths.hdf')
lista = []
for i in tun.index.values:
    lista.append(((i.split('-')[0], i.split('-')[1], i.split('-')[2])))
index = pd.MultiIndex.from_tuples(lista, names=['animal', 'session', 'neuron'])
tun.set_index(index, inplace = True)

#Read widths of the autocorrelograms
aut = pd.read_hdf(dir_read + '/df_autocorrelation_widths.hdf')
aut.set_index(index, inplace = True)


"""
2. Calculation of the speed of the needle
"""

#Create dataframe to store the ratio data
ratio = pd.DataFrame(index = index, columns = aut.columns)
eplist = eplist = ['wake', 'sws', 'rem', 'rem_pre', 'rem_post']

#Calculate the ratio
for ep in eplist: ratio[ep] = tun['width'].div(aut[ep])


"""
3. Plots
"""
from matplotlib.pyplot import *

#Sort values by wake epoch just for visualization
ratio = ratio.sort_values('wake')


"""
3.1 Plot general results
"""

#select epochs
eplist = ['wake', 'sws', 'rem']

fig = figure(figsize=(14,10))
for i in eplist:
    plot(ratio[i].values, marker='o', label=i)
    #xticks(range(len(aut.index)+1), [i[-2:] for i in  aut.index])
    xlabel('neuron')
    ylabel('speed (radians/s)')
    legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
dir2s = output + '/speedftneedle' + 'general'
plt.savefig(dir2s, bbox_inches = 'tight')

# Plot results per animal
for ID in dic:
    fig = figure(figsize=(14,10))
    for i in eplist:
        plot(ratio.loc[ID, i].values, marker='o', label=i)
        #xticks(range(len(aut.index)+1), [i[-2:] for i in  aut.index])
        xlabel('neuron')
        ylabel('speed (radians/s)')
        legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    dir2s = output + '/speedftneedle' + 'general' + ID
    plt.savefig(dir2s, bbox_inches = 'tight')

"""
3.2 REM and nREM vs Wake
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
    dir2s = output + '/speedftneedle' + 'REMvsWAKE' + ID
    plt.savefig(dir2s, bbox_inches = 'tight')

"""
3.3. Barplots
"""

#Mean per group
data = ratio.groupby(level=[['animal']]).sum()/ratio.groupby(level=['animal']).count()

#General
fig = data[['wake', 'sws','rem']].plot.bar(figsize=(12,8))
fig.set_ylabel("Speed (rad/s)"); 
plt.savefig(output + '/bar_general', bbox_inches = 'tight')
#Bar plot 'wake' vs 'rem'
fig = data[['wake', 'rem']].plot.bar(figsize=(12,8))
fig.set_ylabel("Speed (rad/s)"); 
plt.savefig(output + '/rem_vs_wake', bbox_inches = 'tight')
#Bar plot rempre vs rempost
fig = data[['rem_pre', 'rem_post']].plot.bar(figsize=(12,8))
fig.set_ylabel("Speed (rad/s)"); 
plt.savefig(output + '/bar_rpre_vs_rempost', bbox_inches = 'tight')


fig = figure(figsize=(14,10))
for ID in dic:
    for i in eplist:
        bar(ratio[i].mean, label=i)
        #xticks(range(len(aut.index)+1), [i[-2:] for i in  aut.index])
        xlabel('neuron')
        ylabel('speed (radians/s)')
        legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        title('Speed of the needle')
dir2s = output + '/speedftneedle' + 'general'
plt.savefig(dir2s, bbox_inches = 'tight')

#Plot results
for i in eplist: 
        errors = result.loc[i].std()
        result.loc[i].plot.bar(title = "Speed of the needle")
        xticks(np.arange(len(names)), names)

a=complexity
a.plot()
del a['Type']
mu, sigma = "This", 15
plt.figure(figsize=(8,8)); plt.axhline(0, color='green'); plt.ylim([0,1]), plt.ylabel("complexity value"),
plt.text(1, 0.5, r'$\mu=100,\ \sigma=15$'), plt.title('Level of complexity'); 
parallel_coordinates(a.iloc[0:5][['Right Nerve', 'Left Nerve',  'Chiasm',  'ID']], "ID", color=['y', 'b', 'm', 'y', 'm'])
parallel_coordinates(a.iloc[6:11][['Right Nerve', 'Left Nerve', 'Chiasm', 'ID']], "ID", color=['#556270', '556271', '556272', '556273', '556271'])
