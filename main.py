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

#Read widths of the tuning curves
tun = pd.read_hdf('./data_output/df_tuning_widths.hdf')
#Read widths of the autocorrelograms
aut = pd.read_hdf('./data_output/df_autocorrelation_widths.hdf')

#Create dataframe to store the ratio data
result = pd.DataFrame(index = aut.index, columns = aut.columns)
eplist = ['wake', 'sws', 'rem']

#Calculate the ratio
for ep in eplist: result[ep] = tun['width'].div(aut[ep])

#Sort values by wake epoch
result = result.sort_values('wake')

#plot results
from matplotlib.pyplot import *
fig = figure()
for i in eplist:
    plot(result[i].values, marker='o', label=i)
    xticks(range(len(aut.index)+1), [i[-2:] for i in  aut.index])
    xlabel('neuron')
    ylabel('speed (radians/s)')
    legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    title(aut.index[0][0:-5])
speed = './plots/' + 'speedftneedle'
plt.savefig(speed, bbox_inches = 'tight')
    
    



#Plot results
for i in eplist: 
        errors = result.loc[i].std()
        result.loc[i].plot.bar(title = "Speed of the needle")
        xticks(np.arange(len(names)), names)

