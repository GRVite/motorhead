#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:06:39 2018

This script provide functions that help you to 
a) load and accomodate your .mat data, 
b) determine the position in the arena,
c) compute and plot the tuning curve and
d) the autocorrelation of one neuron
(From a to c, the code has been taken mainly from https://github.com/PeyracheLab/StarterPack/blob/master/python/functions.py)
@author: Gilberto Rojas-Vite. Peyrache Lab.

"""

import numpy as np
import pandas as pd
import neuroseries as nts
import os
from functions import *
from functions_mt import *
import scipy.io
import matplotlib.pyplot as plt
data_directory = './data_read/'
routea= './plots/'
routeb= r'cd /home/grvite/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/DreamSpeed - Gilberto/figs/'

"""
0. General Purpose Functions
"""


"""
A. Load and accommodate data
"""
def data_hand(data_directory, ID):
    files 			= os.listdir(data_directory) 
    generalinfo 	= scipy.io.loadmat(data_directory + ID + '_GeneralInfo.mat')
    shankStructure 	= loadShankStructure(generalinfo)
    spikes,shank 	= loadSpikeData(data_directory +  ID + '_SpikeData.mat', shankStructure['thalamus'])
    my_thalamus_neuron_index = list(spikes.keys())
    hd_neuron_index = loadHDCellInfo(data_directory+  ID + '_HDCells.mat', my_thalamus_neuron_index)
    hd_spikes = {}
    for neuron in hd_neuron_index:
        hd_spikes[neuron] = spikes[neuron]
    wake_ep 		= loadEpoch(data_directory, 'wake')
    sleep_ep 		= loadEpoch(data_directory, 'sleep')
    sws_ep 			= loadEpoch(data_directory, 'sws')
    rem_ep 			= loadEpoch(data_directory, 'rem')
    sws_ep = sleep_ep.intersect(sws_ep)
    rem_ep = sleep_ep.intersect(rem_ep)
    return (spikes, shank, hd_spikes, wake_ep, sws_ep, rem_ep)


"""
B. Position in the arena
"""
def det_pos(data_directory, color, path2save):
    #load the angular value at each time steps and make a frame of it
    data = np.genfromtxt(data_directory+'Mouse12-120806_PosHD.txt')
    mouse_position = nts.TsdFrame(d = data[:,[1,2,3]], t = data[:,0], time_units = 's')
    # But TsdFrame is a wrapper of pandas and you can change name of columns in pandas
    # So let's change the columns name
    mouse_position.columns = ['x', 'y', 'ang']
   
    #Plot position
    plt.figure()
    plt.plot(mouse_position['x'].values, mouse_position['y'].values, color)
    plt.xlabel("x position (cm)")
    plt.ylabel("y position (cm)")
    plt.title("Position of the mouse in the arena")
    if path2save == 'a': plot_curve = './plots/' + 'position_' + '.pdf'
    elif  path2save == 'b': plot_curve = r'cd /home/grvite/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/DreamSpeed - Gilberto/figs/' + 'position_' + '.pdf'
    plt.savefig
    plt.show()  
    
    return (mouse_position)    

"""
C. Firing rate
"""
def firetdisco(hd_spikes, neuro_num, epoch):
    """
    # epoch means we need to use the object nts.IntervalSet
    # The interval are contained in the file : Mouse12-120806_ArenaEpoch.txt
    new_data = np.genfromtxt(data_directory+'Mouse12-120806_ArenaEpoch.txt')
    # We can integrate it in nts:
    exploration = nts.IntervalSet(start = new_data[:,0], end = new_data[:,1], time_units = 's')
    """
    # Next step is to compute an average firing rate for one neuron
    # Let's take neuron 
    my_neuron = hd_spikes[neuro_num]
    # To speed up computation, we can restrict the time of spikes
    my_neuron = my_neuron.restrict(epoch)
    first_spike = my_neuron.index[0]
    last_spike = my_neuron.index[-1]
    #Determine bin size in us
    bin_size=1000000 # = 1s
     # Observe the -1 for the value at the end of an array
    duration = last_spike - first_spike
    # it's the time of the last spike
    # with a bin size of 1 second, the number of points is 
    nb_points = duration/bin_size  
    nb_points = int(nb_points)
    #Determine the bins of your data and apply digitize to get a classification index
    bins = np.arange(first_spike, last_spike, bin_size)
    index = np.digitize(my_neuron.index.values, bins, right=False)
  
    #Create a pd
    df_n = pd.DataFrame(index = index)
    df_n['firing_time']=my_neuron.index.values
    #count the number of spikes per bin
    df_n_grouped=df_n.groupby(df_n.index).size().reset_index(name='counts')
    df_n_grouped.set_index('index', inplace=True)
    #generate the real index
    df_comp = pd.DataFrame(index = range(1, np.unique(index)[-1]+1))
    #put that index in your df
    df_cn = df_comp.combine_first(df_n_grouped)
    df_cn.set_index(bins, inplace=True)
    df_cn.fillna(0, inplace=True)
    spike_count = nts.Tsd(t = bins+(bin_size/2.), d = df_cn['counts'].values) #delete, just test
    firing_rate = nts.Tsd(t = bins+(bin_size/2.), d = spike_count.values/(bin_size/1000./1000.))
    meanfiring = firing_rate.sum()/(duration/1000000)
    
    return first_spike, last_spike, bin_size, bins, firing_rate, meanfiring
    
"""
D. Tuning curve
"""


def tuneit(data_directory, hd_spikes, wake_ep, mouse_position, neuro_num, nabins, path2save):
    
    """ Firing rate """
    first_spike, last_spike, bin_size, bins, firing_rate, meanfiring = firetdisco (hd_spikes, neuro_num, wake_ep)
    
    """ Angular direction """
    # Next step is to compute the average angular direction with the same time bin
    # Steps are the same
    head_direction = np.zeros(len(bins))
    for i in range(len(bins)):	
        start = i*bin_size + first_spike
        end = start + bin_size
        head_direction_in_interval = mouse_position['ang'].loc[start:end]
        average_head_direction = np.mean(head_direction_in_interval)
        head_direction[i] = average_head_direction

    head_direction = nts.Tsd(t = np.arange(first_spike, last_spike, bin_size), d = head_direction, time_units = 'us')
    head_direction = head_direction.as_units('s')
    # we can put the two array together
    my_data = np.vstack([firing_rate.values, head_direction.values]).transpose()
    
    # and put them in pandas
    my_data = nts.TsdFrame(t = np.arange(first_spike, last_spike, bin_size), d = my_data)
    # and give name to column
    my_data.columns = ['firing', 'angle']
    # Check you variable and observe the NaN value in the angle column
    # We want to get rid of them so you can call the isnull() function of pandas
    # Observe the False and True value
    # We want the time position that is the inverse of False when calling isnull
    # So you call the angle column
    # And you invert the boolean with ~
    # now we can downsample my_data with index that are only True in the previous line
    my_data = my_data[~my_data.isnull()['angle'].values]
    # Check your variable my_data to see that the NaN have disappeared
    #Compute the mean_firing_rate
    """This part gets rid of some neurons!!!!!!!!!!!"""
        
    """ Tuning curve computation """

    tuning_curve = np.zeros(nabins)
    # the tuning curve is the mean firing rate per angular bins
    # First step is to define the angular bins
    angular_bins = np.linspace(0, 2*np.pi, nabins+1)
    #transform it to values
    val = my_data['angle'].values

    # Now we can loop
    for i in range(nabins):
        left_border = angular_bins[i]
        right_border = angular_bins[i+1]
        index = np.logical_and(val>left_border, val<=right_border)
        tuning_curve[i] = np.mean(my_data[index]['firing'])


    #Define phase values
    phase = np.linspace(0, 2*np.pi, nabins)
    
    """ Plots """
    from matplotlib.pyplot import hlines as hlines
    
    plt.figure(figsize = (8,6)) 
    plt.plot(phase, tuning_curve, color='darkorange')
    hlines (tuning_curve.max()/2, phase.min(),  phase.max(), 'r')
    #vlines (phase.max()/2, tuning_curve.min(),  tuning_curve.max(), 'r')    
    plt.xlabel("Head-direction (rad)")
    plt.ylabel("Firing rate")
    plt.title("ADn neuron")
    plt.grid()
    if path2save == 'a': plot_curve = './plots/' + 'tuning_curve_' + str(neuro_num) + '.pdf'
    elif  path2save == 'b': plot_curve = r'cd /home/grvite/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/DreamSpeed - Gilberto/figs/' + 'tuning_curve_' + str(neuro_num) + '.pdf'
    plt.savefig(plot_curve)
    plt.show()
    tun_std=np.std(tuning_curve)
    
    # Polar plot
    plt.figure(figsize=(12,8))
    plt.subplot(111, projection='polar')
    plt.plot(phase, tuning_curve, color = 'chocolate', markerfacecoloralt = 'darkorange')
    plt.xlabel("Head-direction (rad)")
    #plt.ylabel("Firing rate")
    plt.title("ADn neuron")
    plt.grid()
    if path2save == 'a': plot_polar = './plots/' + 'tuning_polar_' + str(neuro_num) + '.pdf'
    elif  path2save == 'b': plot_polar = r'cd /home/grvite/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/DreamSpeed - Gilberto/figs/' + 'tuning_polar_' + str(neuro_num) + '.pdf'
    plt.savefig(plot_polar)
    plt.show()

    return (my_data, tuning_curve, angular_bins)


#Function for Width calculation for the tuning curve
    

#Tuning curve computation for plots with gaussian shape
def width_gaussian(nabins, array):
    phase = np.linspace(0, 2*np.pi, nabins)
    dic = dict(zip(array, list(range(0,len(array))))) 
    max_tmp = array.max()
    pos_max = dic [max_tmp]
    lista=[]
    for i in array:
        if i>=array.max()/2:
            lista.append(i)
    nums = np.array(lista)
    lo = nums.min()
    pos_min = dic[lo]
    width_auto = abs((pos_max-pos_min)*((2*np.pi)/(len(phase)-1)))*2
    print("the width is", width_auto)
    return  width_auto

"""
D. Autocorrelation
"""

def plotautco(hd_spikes, neuro_num, meanfiring, epoch, epochstr, binsize, nbins, path2save):        
        
    # Let's take neuron 
    my_neuron = hd_spikes[neuro_num]
    # To speed up computation, we can restrict the time of spikes to epoch of wake
    my_neuron = my_neuron.restrict(epoch)
    #change units to ms
    mi_neurona = my_neuron.as_units('ms') 
    first_spike = mi_neurona.index [0]
    last_spike = mi_neurona.index[-1]
    duration = last_spike - first_spike
    
    #compute autcorrelation
    aucorr = crossCorr(mi_neurona.index, mi_neurona.index, binsize, nbins)
    aucorr [int(nbins/2)] = 0.0
    intervalo = mi_neurona.index.max() - mi_neurona.index.min()
    #aucorr = aucorr/1000/meanfiring #normalize by the meanfiring rate
    
    
    #Smooth the data for an easier calculation of the width
    dfa = aucorr [0:int(nbins/2)]
    dfa = pd.DataFrame(dfa).rolling(window = 7, win_type='gaussian', center=True, min_periods = 1).mean(std = 5.0)
    dfb = np.flipud(aucorr [int(nbins/2)+1::])
    dfb = pd.DataFrame(dfb).rolling(window = 7, win_type='gaussian', center=True, min_periods = 1).mean(std = 5.0)
    array = np.append((dfa.values),0)
    arrayt = np.append(np.append((dfa.values),0), np.flipud(dfb.values))

    #calculating width
    dic = dict(zip(array, list(range(0,nbins+1)))) 
    lista=[]
    meanfiring = meanfiring
    half_mfr2max= ((array.max() - meanfiring)/2) +meanfiring
    print(half_mfr2max, array.max())
    for i in array:
        if i>=half_mfr2max:
            lista.append(i)
    nums = np.array(lista)
    index_min = dic[nums.min()]
    index_max = dic[nums.max()]
    width_auto = (abs(index_max-index_min)) *2 +1 #get the distance in bins
    width_auto = width_auto*binsize/1000
    print("for neuron {0} the width is".format(hd_spikes[neuro_num]), width_auto, ' s')
       
    
    """Plot autocorrelogram"""
    from matplotlib.pyplot import hlines as hlines
    plt.figure(figsize=(12,8))
    #plt.plot(aucorr) # Plot the raw version
    plt.plot(arrayt) # Plot the smoothed version
    plt.title("Autocorrelogram")
    plt.xlabel("time")
    #middle horizontal line
    hlines (meanfiring, 0,  nbins, 'g', label = 'mean firing rate')
    hlines (half_mfr2max, 0,  nbins, 'r', label = 'half point')
    if path2save == 'a': autocorrelogram = './plots/' + 'autocorrelogram_' + str(neuro_num) + '_' + epochstr + '.pdf'
    elif  path2save == 'b': autocorrelogram = r'cd /home/grvite/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/DreamSpeed - Gilberto/figs/' + 'autocorrelogram_' + str(neuro_num) + '_' + epochstr +'.pdf'
    plt.savefig(autocorrelogram)
    
    return aucorr, width_auto

#end

first_spike, last_spike, bin_size, bins, firing_rate, meanfiring = firetdisco (hd_spikes, 19, sws_ep)
aucor, width_auto = plotautco (hd_spikes, 19, meanfiring, sws_ep, 'w', 50, 200, 'a')

