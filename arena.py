#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:22:34 2018

@author: vite
"""

data_directory = './data_read_t/Mouse17/Mouse17-130125/'
ID = 'Mouse17-130125'
data_directory = './data_read/'
ID = 'Mouse12-120806'
data = np.genfromtxt(data_directory + ID + '_pos.txt')


mouse_position = nts.TsdFrame(d = data[:,[1,2]], t = data[:,0], time_units = 's')

nbins = 60






#way 1
my_neuron = hd_spikes[34]
my_neuron = my_neuron.restrict(wake_ep)
ang_spk = ang.realign(my_neuron)

ang = np.genfromtxt(data_directory + ID + '_ang.txt')
ang = nts.TsdFrame(d = ang[:,1], t = ang[:,0], time_units = 's')
ang = ang.restrict(wake_ep)

com = ang_spk.start_time()
fin = ang_spk.end_time()
duration = fin - com
bin_size=duration/nbins
bins = np.arange(com,fin, bin_size)
indice = np.digitize(ang_spk.index.values, bins, right=False)

ang = pd.DataFrame(data = ang_spk.values, index = indice)
ang = df.groupby(df.index).mean()
ang = ang.values

spk = pd.DataFrame(data = my_neuron.index.values, index = indice)
spk = spk.groupby(spk.index).size().reset_index(name='counts')
spk = spk['counts'].values

df = pd.DataFrame(data = spk, columns =['spk'])
df['angle'] = ang
df.sort_values(by='angle', inplace = True)
plt.plot(df['angle'], df['spk'])


#way 2

ang = np.genfromtxt(data_directory + ID + '_ang.txt')
ang = nts.TsdFrame(d = ang[:,1], t = ang[:,0], time_units = 's')
ang = ang.restrict(wake_ep)

my_neuron = hd_spikes[34]
my_neuron = my_neuron.restrict(wake_ep)

ang_spk = ang.realign(my_neuron)
ang_spk = pd.DataFrame(data=ang_spk.values, columns=['angle'], index = my_neuron.index.values)


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


#third
ang = np.genfromtxt(data_directory + ID + '_ang.txt')
ang = nts.TsdFrame(d = ang[:,1], t = ang[:,0], time_units = 's')
ang = ang.restrict(wake_ep)

my_neuron = hd_spikes[34]
my_neuron = my_neuron.restrict(wake_ep)
ang_spk = ang.realign(my_neuron)




        
plt.plot()


tuning = = nts.TsdFrame(

  ############################           ############## ###############  

nbins = 60      
neuro_num = 34 
  
#Fourth
def tuneit (hd_spikes, wake_ep, neuro_num, nbins)  
    # read angular data
    ang = np.genfromtxt(data_directory + ID + '_ang.txt')
    # transform it to Tsd
    ang = nts.TsdFrame(d = ang[:,1], t = ang[:,0], time_units = 's')
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
    
    #a = spike count, b and d the bonds of the bins
    a, b = np.histogram(ang_spk.values, phase)
    
    #c = angular count
    c, d = np.histogram(ang.values, phase)
    
    #tuning curve
    tuning = pd.DataFrame (data = a/c, index = phase[0:-1])


#define bin_size
bin_size=1000000 # = 1s

com = my_neuron.start_time()
fin = my_neuron.end_time()
bins = np.arange(com,fin, bin_size)
duration = fin - com

# generate an array of labels for our data corresponding to the bins defined 
indice = np.digitize(my_neuron.index.values, bins, right=False)

# generate a pandas dataframe for the angular values with the defined index
df_ang = pd.DataFrame(data = ang_spk.values, index = indice)
# take the mean of the angular data per bin
ang = df_ang.groupby(df_ang.index).mean()

#generate a pandas dataframe with data of the spikes per time bin with the defined index
spk = df_ang.groupby(df_ang.index).size().reset_index(name='counts')['counts']

#Make a df with the 
my_data = pd.DataFrame (data = spk.values, columns = ['firing'])


tuning_curve = np.zeros(nabins)
# the tuning curve is the mean firing rate per angular bins
# First step is to define the angular bins
angular_bins = np.linspace(0, 2*np.pi, nabins+1)
#transform it to values
val = ang.values
# Now we can loop
for i in range(nabins):
    left_border = angular_bins[i]
    right_border = angular_bins[i+1]
    index = np.logical_and(val>left_border, val<=right_border)
    tuning_curve[i] = np.mean(my_data[index]['firing'])
    


#Five
# read angular data
ang = np.genfromtxt(data_directory + ID + '_ang.txt')
# transform it to Tsd
ang = nts.TsdFrame(d = ang[:,1], t = ang[:,0], time_units = 's')
# restrict to wake_ep
ang = ang.restrict(wake_ep)

#load neuron information
my_neuron = hd_spikes[34]
# restrict to wake_ep
my_neuron = my_neuron.restrict(wake_ep)

#realign data
ang_spk = ang.realign(my_neuron)
ang_spk.to
my_data = pd.DataFrame (data = ang_spk.values, columns = ['firing'])






data = ang_spk.values
df_spk = pd.DataFrame(data = data, index = indice)
df_spk = pd.DataFrame(data = data,


data = np.genfromtxt(data_directory + ID + '_pos.txt')


mouse_position = nts.TsdFrame(d = data[:,[1,2]], t = data[:,0], time_units = 's')
mouse_position['ang'] = ang[:,1]
mouse_position.columns = ['x', 'y', 'ang']
mouse_position = mouse_position.restrict(wake_ep)

bin_size=1000000 # = 1s
com = mouse_position.start_time()
fin = mouse_position.end_time()

duration = fin - com
bins = np.arange(com,fin, bin_size)
indice = np.digitize(mouse_position.index.values, bins, right=False)
data = mouse_position['ang'].values
df_pos = pd.DataFrame(data = data, index = indice)
data_pos = df_pos.groupby(df_pos.index).mean().values
df_pos = pd.DataFrame(data = data, index = bins)
df_pos['firing'] = firing_rate.values
values = df_pos.iloc[:].values
values= np.squeeze(values)
nts_pos = nts.Tsd(t=bins, d= values)

nts_pos[]
test2 = nts_pos.realign(firing_rate)
test2 = test2.to_frame
test2['firing'] = firing_rate.values



#concat
pd.concat(nts_pos, firing_rate)
pd.concat(nts_pos.as_series, firing_rate.as_series)

#changing it to series
nts_pos.as_series['firing_rate']=firing_rate.values

#normal
test = pd.DataFrame()
test['firing'] = firing_rate.values

#check data 
data = np.array[[1,2],[2,4]]

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