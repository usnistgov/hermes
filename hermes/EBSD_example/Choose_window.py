# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:58:30 2021

@author: asm6
"""

import numpy as np

def choose_window(x_seed, y_seed, input_locs, window_size = 51):
    '''
    input_locs is the locations of all the data
    x in column 0, y in column 1, z in column 2
    
    We want data that is a square window of size window_size^2 entries
    '''
    #only include the locations where the x location is in renge definde by the index from x_seed to x_seed+window_size
    x_input_locs = input_locs[np.isin(input_locs[:,0], 
                                      np.unique(input_locs[:,0])[x_seed:window_size+x_seed])]
    
    mask = np.isin(input_locs[:,0], np.unique(input_locs[:,0])[x_seed:window_size+x_seed])
    
    
    #Down select for the values of y defined by the similar range. 
    y_input_locs = x_input_locs[np.isin(x_input_locs[:,1],
                                        np.unique(x_input_locs[:,1])[y_seed:window_size+y_seed])]
    
    mask = mask * np.isin(input_locs[:,1], np.unique(input_locs[:,1])[y_seed:window_size+y_seed])
    
    
    return y_input_locs, mask