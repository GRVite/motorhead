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
for i in eplist: 
    result.loc[i] = tun.values/aut.loc[i].values

names  = [x[-2:] for x in aut.columns]

from pylab import *
#Plot results
for i in eplist: 
        errors = result.loc[i].std()
        result.loc[i].plot.bar(title = "Speed of the needle")
        xticks(np.arange(len(names)), names)
        speed = './plots/' + 'barplot_speed' + i
        plt.savefig(speed)
