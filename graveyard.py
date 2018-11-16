#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:54:39 2018

@author: vite
"""

#Date: 14Nov2018
#This function allows you to calculate the tuning curve for one neuron
def tuneit(hd_spikes, wake_ep, mouse_position, neuro_num, nabins, path2save):
    
    """ Firing rate """
    #Calculate firing rate
    first_spike, last_spike, bin_size, bins, firing_rate = firetdisco (hd_spikes, neuro_num, wake_ep)
    
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
    
    """Plots
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
    plt.show()"""

    return (my_data, tuning_curve)