#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:06:39 2018

This script provide functions that help you to 
a) load and accomodate your .mat data, 
b) determine the position in the arena,
c) compute and plot the tuning curve and
d) the autocorrelation of one neuron
(This code uses some functions and scripts from https://github.com/PeyracheLab/StarterPack)
@author: Gilberto Rojas Vite. Peyrache Lab.

"""

import numpy as np
import pandas as pd
import neuroseries as nts
import os
from functions import *
import scipy.io
import matplotlib.pyplot as plt
routea= './plots/'
routeb= r'cd /home/grvite/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/DreamSpeed - Gilberto/figs/'

"""
0. General Purpose Functions
"""


"""
A. Load and accommodate data
"""

def getcodes(data_directory):
    import os
    #Make a list of your animals
    #Sometimes, you have a .DS_Store, and you do not want it in your list.
    main = [i for i in os.listdir(data_directory) if i != '.DS_Store' and i != 'PositionFiles.tar' and i != 'positions']
    main.sort()
    
    #Make a dict of all your sessions
    dic = {}
    sessions = []
    for m in main:
        dire = data_directory + m
        dire = {i for i in os.listdir(dire) if i != '.DS_Store'}   
        dic [m] = dire
        for s in dic[m]:
            sessions.append(s)
    return dic


def gen_labels(data_directory):
    import os
    #Make a list of your animals
    #Sometimes, you have a .DS_Store, and you do not want it in your list.
    main = [i for i in os.listdir(data_directory) if i != '.DS_Store' and i != 'PositionFiles.tar' and i != 'positions']
    main.sort()
    
    #Make a dict of all your sessions
    dic = {}
    sessions = []
    for m in main:
        dire = data_directory + m
        dire = {i for i in os.listdir(dire) if i != '.DS_Store'}   
        dic [m] = dire
        for s in dic[m]:
            sessions.append(s)
    
    #Make a dictionary with sessions as keys and neurons as values of the keys
    neuronas={}
    for mouse in list(dic.keys()): #mouse = 'Mouse12', 'Mouse17' and so on
        for s in dic[mouse]: #s = session 
            path = data_directory + mouse + '/' + s
            spikes, shank, hd_spikes = data_hand (path, s)
            neuronas[s] = list(hd_spikes.keys())[0]
            lista=[]
            for i in list(hd_spikes.keys()): #we iterate in the list that contains the number of neurons per session
                lista.append(i)
                neuronas[s] = lista
                
    #Make labels for animals, sessions and neurons
    c0 = 0
    c1 = 0
    c2 = 0
    labels_a = []
    labels_s=[]
    labels_n = []
    neurons = []
    for a in dic:
        for s in dic[a]:
            print(s)
            for n in neuronas[s]:
                neurons.append(s + '-' + str(n))
                labels_n.append(c0)
                labels_s.append(c1)
                labels_a.append(c2)
                c0+=1
            c1+=1
        c2+=1
    return [labels_a,labels_s, labels_n]

def files_managment(dic, data_directory, pos_dir):
    import os       
    #Move files one level up from Analysis folder
    for i in dic.keys():
        print(i)
        for j in dic[i]:
            path = data_directory + i + '/' + j + '/'
            print('the path is', path)
            for file in os.listdir(path + '/Analysis'):
                os.rename (path + '/Analysis/' + file, path + file) #let you move a file from one dir to another

    #function for copying files from a folder called 'positions' with files of mouse positions and angles to the folders for each session
    from shutil import copyfile 
    for i in dic.keys():
        print(i)
        for j in dic[i]:
            path = data_directory + i + '/' + j + '/' + j
            try: 
                print('the path is', pos_dir + '/' + i + '/' + j + '/' + j + '.pos.txt')
                copyfile (pos_dir + '/' + i + '/' + j + '/' + j + '.pos', path + '_pos.txt')
                copyfile (pos_dir + '/' + i + '/' + j + '/' + j + '.ang', path + '_ang.txt')
            except FileNotFoundError: 
                print("exemption for ", i)
                pass
    

def find_pos():
    #Search ---run just if PosHd.txt is not in the files 
    import os.path
    miss = []   
    pos_dir= './data_read_t/positions'   
    for i in dic.keys():
        print(i)
        for j in dic[i]:
            path = data_directory + main[i] + '/' + j + '/' + j
            print('the path is', path)
            if os.path.exists(path + '_PosHD.txt') == False: 
                miss.append(j) 
                os.rename(pos_dir + '/' + main[i] + '/' + j + '/' + j + '.pos.txt', path + '_PosHD.txt')
                
def data_hand(data_directory, ID):
    files 			= os.listdir(data_directory) 
    generalinfo 	= scipy.io.loadmat(data_directory + '/GeneralInfo.mat')
    shankStructure 	= loadShankStructure(generalinfo)
    spikes,shank 	= loadSpikeData(data_directory + '/SpikeData.mat', shankStructure['thalamus'])
    my_thalamus_neuron_index = np.array(list(spikes.keys()))
    hd_neuron_index = loadHDCellInfo(data_directory + '/HDCells.mat', my_thalamus_neuron_index)
    if data_directory == './data_read_t/Mouse25/Mouse25-140130': 
        print('hey', spikes.keys(), hd_neuron_index)
    hd_spikes = {}
    for neuron in my_thalamus_neuron_index[hd_neuron_index]:
        hd_spikes[neuron] = spikes[neuron]
    return (spikes, shank, hd_spikes)


"""
B. Position in the arena
"""

def det_pos(data_directory, ID, color, path2save):
    #load the angular value at each time steps and make a nts frame of it
    data = np.genfromtxt(data_directory + ID + '_PosHD.txt')
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
    #fill with 0 for the na values
    df_cn.fillna(0, inplace=True)
    #generate a Tsd with the data
    spike_count = nts.Tsd(t = bins+(bin_size/2.), d = df_cn['counts'].values) 
    #change units to spikes per second = firing rate
    firing_rate = nts.Tsd(t = bins+(bin_size/2.), d = spike_count.values/(bin_size/1000./1000.))
    return first_spike, last_spike, bin_size, bins, firing_rate

#Determine the mean firing rate for any epoch
def meanfiring_f(hd_spikes, neuro_num, epoch):
    #select one neuron
    my_neuron = hd_spikes[neuro_num]
    #restrict the activity to one epoch
    my_neuron = my_neuron.restrict(epoch)
    #change units to seconds
    my_neuron.as_units('s')
    #count the number of spikes
    count = my_neuron.index.shape[0]
    #calculate mean firing rate
    meanf = count/epoch.tot_length('s')
    return (meanf)


"""
D. Tuning curve
"""

    
#This function allows you to calculate the tuning curve for one neuron
def tuneit (hd_spikes, ang, wake_ep, neuro_num, nbins):

    # restrict to wake_ep
    ang = ang.restrict(wake_ep)
    
    #load neuron information
    my_neuron = hd_spikes[neuro_num]
    # restrict to wake_ep
    my_neuron = my_neuron.restrict(wake_ep)
    
    #realign data
    ang_spk = ang.realign(my_neuron)
    
    #define angular bins
    phase = np.linspace(0, 2*np.pi, nbins)
    
    #a = spike count, b = bins limits
    a, b = np.histogram(ang_spk.values, phase)
    
    #c = angular count, d = bins limits
    c, d = np.histogram(ang.values, phase)
    
    #tuning curve
    tuning = pd.DataFrame (data = a/c/np.median(np.diff(ang.as_units('s').index)), index = phase[0:-1])
    
    return tuning

#Function for width  computation for plots with gaussian shape
def width_gaussian(nabins, array):
    phase = np.linspace(0, 2*np.pi, nabins)
    dic = dict(zip(array, list(range(len(array))))) 
    max_a = array.max()
    pos_max = dic [max_a]
    #find the position in the array of the middle point 
    x = int(len(array)/2)
    
    if pos_max > x: array = np.append (array[x:], array[:x])
    else: array = np.append (array[x:], array)
    
    lista=[]
    for i in array:
        if i>=max_a/2:
            lista.append(i)
    nums = np.array(lista)
    lo = nums.min()
    dic = dict(zip(array, list(range(len(array))))) 
    pos_min = dic[lo]
    max_a = array.max()
    pos_max = dic [max_a]
    width_auto = abs((pos_max-pos_min)*((2*np.pi)/(len(phase)-1)))*2
    print("the width is", width_auto)
    return  width_auto

#This function calculate the tuning curve for all the neurons of one session
"""
D. Autocorrelation
"""

def corr_calc(hd_spikes, neuro_num, epoch, binsize, nbins):    
    # Let's take neuron 
    my_neuron = hd_spikes[neuro_num]
    # To speed up computation, we can restrict the time of spikes to epoch of wake
    my_neuron = my_neuron.restrict(epoch)
    #change units to ms
    mi_neurona = my_neuron.as_units('ms') 
    #compute autcorrelation
    aucorr = crossCorr(mi_neurona.index, mi_neurona.index, binsize, nbins)
    aucorr [int(nbins/2)] = 0.0
    #aucorr = aucorr/1000/meanfiring #normalize by the meanfiring rate
    return aucorr
"""   
def width_corr(aucorr, nbins, binsize, meanfiring, window = 7, stdv = 5.0, plot):
    #Smooth the data for an easier calculation of the width
    dfa = aucorr [0:int(nbins/2)]
    dfa = pd.DataFrame(dfa).rolling(window = window, win_type='gaussian', center=True, min_periods = 1).mean(std = stdv)
    dfb = np.flipud(aucorr [int(nbins/2)+1::])
    dfb = pd.DataFrame(dfb).rolling(window = window, win_type='gaussian', center=True, min_periods = 1).mean(std = stdv)
    array = np.append((dfa.values),0)
    arrayt = np.append(np.append((dfa.values),0), np.flipud(dfb.values))
    #Make a Tsd
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
    ndf = nts.Tsd(t = times, d = arrayt/meanfiring)
    if plot = True: ndf.plot()
    #calculating width
    dic = dict(zip(array, list(range(0,nbins+1)))) 
    lista=[]
    #half_mfr2max= ((array.max() - meanfiring)/2) + meanfiring
    
    for i in array:
        if i>=half_mfr2max:
            lista.append(i)
    nums = np.array(lista)
    index_min = dic[nums.min()]
    index_max = dic[nums.max()]
    width_auto = (abs(index_max-index_min)) *2 +1 #get the distance in bins
    width_auto = width_auto*binsize/1000
    return width_auto    

def width_corr2(aucorr, nbins, binsize, meanfiring, window = 7, stdv = 5.0, plot):
    #Smooth the data for an easier calculation of the width
    dfa = aucorr [0:int(nbins/2)]
    dfa = pd.DataFrame(dfa).rolling(window = window, win_type='gaussian', center=True, min_periods = 1).mean(std = stdv)
    dfb = np.flipud(aucorr [int(nbins/2)+1::])
    dfb = pd.DataFrame(dfb).rolling(window = window, win_type='gaussian', center=True, min_periods = 1).mean(std = stdv)
    array = np.append((dfa.values),0)
    arrayt = np.append(np.append((dfa.values),0), np.flipud(dfb.values))
    #Make a Tsd
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
    ndf = nts.Tsd(t = times, d = arrayt/meanfiring)
    
    if plot = True: ndf.plot()
    
    array = array/meanfiring  
    #calculating width
    dic = dict(zip(array, list(range(0,nbins+1)))) 
    lista=[]
    half= array.max()/2
    for i in array:
        if i>=half:
            lista.append(i)
    nums = np.array(lista)
    index_min = dic[nums.min()]
    index_max = dic[nums.max()]
    width_auto = (abs(index_max-index_min)) *2 +1 #get the distance in bins
    width_auto = width_auto*binsize/1000
    return width_auto  
"""
def smooth_corr(aucorr, nbins, binsize, meanfiring, window = 7, stdv = 5.0, plot = False):
    aucorr= aucorr-meanfiring
    dfa = aucorr [0:int(nbins/2)]
    dfa = pd.DataFrame(dfa).rolling(window = window, win_type='gaussian', center=True, min_periods = 1).mean(std = stdv)
    dfb = np.flipud(aucorr [int(nbins/2)+1::])
    dfb = pd.DataFrame(dfb).rolling(window = window, win_type='gaussian', center=True, min_periods = 1).mean(std = stdv)
    #array = np.append((dfa.values),0)
    arrayt = np.append(np.append((dfa.values),0), np.flipud(dfb.values))
    if plot == True: 
        #Make a Tsd
        times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
        ndf = nts.Tsd(t = times, d = arrayt/meanfiring)
        ndf.plot()
    return arrayt
    
def calc_width(array, nbins, binsize):
    array = array[:int(nbins/2)]
    #calculating width
    dic = dict(zip(array, list(range(0,nbins+1)))) 
    lista=[]
    half= array.max()/2
    for i in array:
        if i>=half:
            lista.append(i)
    nums = np.array(lista)
    index_min = dic[nums.min()]
    index_max = dic[nums.max()]
    width_auto = (abs(index_max-index_min)) *2 +1 #get the distance in bins
    width_auto = width_auto*binsize/1000
    return width_auto  

def aucorr_plot(data, nbins, binsize, epochstr, path2save): 
    """Plot autocorrelogram"""
    from matplotlib.pyplot import hlines as hlines
    plt.figure(figsize=(12,8))
    #plt.plot(aucorr) # Plot the raw version
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
    data = nts.Tsd(t = times, d = data, time_units = 'ms')
    plt.plot(data.as_units('ms')) # Plot the smoothed version
    plt.title("Autocorrelogram")
    plt.xlabel("time (ms)")
    #middle horizontal line
    #hlines (meanfiring, 0,  nbins, 'g', label = 'mean firing rate')
    hlines (data.max()/2, 0-nbins*binsize/2,  nbins*binsize/2, 'r', label = 'half point')
    if path2save == 'a': autocorrelogram = './plots/' + 'autocorrelogram_' + str(neuro_num) + '_' + epochstr + '.pdf'
    elif  path2save == 'b': autocorrelogram = r'cd /home/grvite/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/DreamSpeed - Gilberto/figs/' + 'autocorrelogram_' + str(neuro_num) + '_' + epochstr +'.pdf'
    plt.savefig(autocorrelogram)

#end


