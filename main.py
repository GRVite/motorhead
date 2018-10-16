#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:14:25 2018

This script computes the speed of the head direction signal. It reads the data
from main_tuning_curve and main_autocorrelation, calculates the ratio and plots the results. 

@author: grvite
"""

import pandas as pd

#Read widths of the tuning curves
tun = pd.read_csv('./data/df_tuning_widths.csv')
tun.drop('Unnamed: 0', axis = 1, inplace = True) #check this

#Read widths of the autocorrelograms
aut = pd.read_csv('./data_output/df_autocorrelation_widths.csv', index_col = 0)


#Check units!

#Create dataframe to store the ratio data
result = pd.DataFrame(index = aut.index, columns = aut.columns)
eplist = ['wake', 'sws', 'rem']

for i in eplist: 
    result.loc[i] = tun.values/aut.loc[i].values

#Plot results
    result.loc['wake'] = tun.values/aut.loc['wake'].values