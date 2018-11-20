#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:47:07 2018

@author: vite
"""
import pandas as pd

#Select directory where you have all the folders of your animals data
data_directory = './data_read_t/'
#Make a list of your animals
#Sometimes, you have a .DS_Store, and you do not want it in your list.
main = [i for i in os.listdir(data_directory) if i != '.DS_Store']

#Make a dict of all your sessions
dic = {}
for n, m in zip(range(len(main)), main):
    dire = data_directory + m
    dire = {i for i in os.listdir(dire) if i != '.DS_Store'}
    dic [m] = dire

#sessions
sessions = []
for i in dic.values(): 
    for j in i: sessions.append(j[8:])

codes = pd.DataFrame(columns=['neurons'], index = pd.MultiIndex.from_product((main, sessions)))

neur_dic
neuronas= []
#Get the number of neurons
for mouse in dic.keys():
    print(mouse)
    for ID, s in zip (dic[mouse], sessions):
        path = data_directory + mouse + '/' + ID
        print(path)
        spikes, shank, hd_spikes = data_hand (path, ID)
        for i in hd_spikes.keys():
            print(ID, s)
            neuronas.append(ID[8:]+'-' + str(i))


        codes.loc[(mouse, ID), 'neurons']=list(hd_spikes.keys())

#drop na values
codes.dropna(inplace = True)

#save in .hdf format
codes.to_hdf('./data_output/codes.hdf', 'df')

return main,codes