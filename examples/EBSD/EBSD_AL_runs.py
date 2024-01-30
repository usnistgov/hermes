# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:52:06 2021

@author: asm6
"""

import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

import warnings
warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

import tqdm
import multiprocessing
import sys; sys.path.insert(0,'C:/Users/asm6/Documents/Projects/EBSD/EBSD analysis/EBSD 2D active learning loops/')

import orix
import orix.io

from EBSD_graph_cluster import cluster
from Train_Predict_Plot_GPC import train_HSGPC_classifier, train_GPC_classifier, predict_class, plot_fit
from Choose_window import choose_window
from EBSD_batch_acquistion import acquire_next_batch
from Euler_to_quat_noise import Euler_to_quat_noise
from EBSD_DataDaemon import DataDaemon


'''
Read in all the input data both for the Ground truth and for the raw measureemtns
'''
#Read in the Grain mapping results from using the Burn Algorithm on the whole data set from Sukbin.
Ground_Truth = pd.read_fwf('C:/Users/asm6/Documents/Projects/EBSD/zirconia_sukbin_Oct2020/zirconia_sukbin_Oct2020/voxel_info_no_preamble.txt', names=['Grain Label', 'X', 'Y', 'Z', 'Q1', 'Q2', 'Q3', 'Q4'])

'''Need to keep the -3 label as a grain label so that none of the windows have missing points
'''
#Throw away the voxels with the '-3' dummy label 
# Ground_Truth = Ground_Truth[Ground_Truth['Grain Label'] != -3]

L = np.array(Ground_Truth['Grain Label'])

#Dimensional limits of the input space
x_min = np.min(Ground_Truth['X'])
x_max = np.max(Ground_Truth['X'])
y_min = np.min(Ground_Truth['Y'])
y_max = np.max(Ground_Truth['Y'])
z_min = np.min(Ground_Truth['Z'])
z_max = np.max(Ground_Truth['Z'])

#Read .ang file
cm = orix.io.loadang('C:/Users/asm6/Documents/Projects/EBSD/zirconia_sukbin_Oct2020/zirconia_sukbin_Oct2020/Registered_zirconia_1.ang')
### data = cm.conj
# data = cm.data


# Read each column from the .ang file
euler1, euler2, euler3, x, y, iq, ci, phase_id, di, fit = np.loadtxt('C:/Users/asm6/Documents/Projects/EBSD/zirconia_sukbin_Oct2020/zirconia_sukbin_Oct2020/Registered_zirconia_1.ang', unpack=True)
data = np.concatenate((euler1.reshape(-1,1), euler2.reshape(-1,1), euler3.reshape(-1,1)), 1)

#List the xyz locations
'''Assumes z=-0.4'''
input_loc = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis = 1)
input_loc = np.concatenate((input_loc, -0.4*np.ones_like(x.reshape(-1,1))), axis = 1)
#Find the Ground truth for those locations
Layer_ground_truth = Ground_Truth[Ground_Truth['Z'] == -0.4]
# Layer_ground_truth = Layer_ground_truth[Layer_ground_truth['X'] >= 0-10e-3]
# Layer_ground_truth = Layer_ground_truth[Layer_ground_truth['Y'] >= 0-10e-3]
# Layer_ground_truth = Layer_ground_truth[Layer_ground_truth['X'] <= np.max(input_loc[:,0])+10e-3]
# Layer_ground_truth = Layer_ground_truth[Layer_ground_truth['Y'] <= np.max(input_loc[:,1])+10e-3]
Layer_ground_truth = Layer_ground_truth[np.isin(Layer_ground_truth['X'],input_loc[:,0])]
Layer_ground_truth = Layer_ground_truth[np.isin(Layer_ground_truth['Y'],input_loc[:,1])]

Ground_truth_labels = np.array(Layer_ground_truth['Grain Label'])

    
def Run_an_AL_campaign(params):
    model_type = params[0]
    random_seed = params[1]
    queue = params[2]
    

    np.random.seed(random_seed)
    
    window = 75 #pixels in the square window to use
    x_lim = np.unique(input_loc[:,0]).shape[0]
    y_lim = np.unique(input_loc[:,1]).shape[0]
    assert x_lim - window > 0, 'Window too large for x input size'
    assert y_lim - window > 0, 'Window too large for y input size'
    
    x_seed = np.random.randint(0,x_lim - window + 1)
    y_seed = np.random.randint(0,y_lim - window + 1)
    
    sub_input_loc, mask = choose_window(x_seed, y_seed, input_loc, window_size = window)
    
    local_ground_truth = Ground_truth_labels[mask]

    #Take noisy observations!
    #with ## uncertainty in the Euler angles
    local_euler = data[mask]
    local_data = Euler_to_quat_noise(local_euler, 2*np.pi/180, 2*np.pi/180, 2*np.pi/180)
    local_data = orix.quaternion.Quaternion(data=local_data)
    # local_data = data[mask]


    # Number of points measured to start with
    start_measurements = 10
    measured_index = np.random.permutation(sub_input_loc.shape[0])
    measured_index = measured_index[np.arange(0,start_measurements)]
    
    #Start a container for the results tabel
    al_results_table_values = []  
    
    Maximum_loops = 30
    for i in range(Maximum_loops):
        #Get the Active training locations from the index:
        active_train_locations = sub_input_loc[measured_index]
        
        #Take measurements at the active training sites:
        active_train_measurements = local_data[measured_index]
        
        #Cluster
        active_labels, active_probabilities, active_C, active_Graph = cluster(active_train_locations, active_train_measurements)
    
        # Train
        if model_type == 'Heteroscedastic':
            active_model = train_HSGPC_classifier(active_train_locations, active_labels, active_probabilities)
        elif model_type == 'Homoscedastic':
            active_model = train_GPC_classifier(active_train_locations, active_labels, len(active_probabilities[0,:]))
        else:
            print('Error NOT valid model type')
        #Predict
        active_classes, active_total_var, active_mean, active_Var = predict_class(active_model, sub_input_loc)
        #Caclulare Adjusted Rand Score to the ground truth
        ARS = adjusted_rand_score(local_ground_truth, active_classes)
    
    
        #Acquire next points
        points = acquire_next_batch(active_model, sub_input_loc, measured_index, batch_size=4)
        
        loop_results = [i, measured_index, points, active_classes, active_total_var, ARS]
        al_results_table_values.append(loop_results)
        
        measured_index = np.concatenate((measured_index, points.flatten()))
        
        #Test for convergence
        if i > 3:
            #find the active_classes of previous result
            back_1_map = al_results_table_values[-2][3]
            back_2_map = al_results_table_values[-3][3]
            back_3_map = al_results_table_values[-4][3]
            
            ARS_back = np.array([adjusted_rand_score(active_classes, back_1_map),
                                 adjusted_rand_score(active_classes, back_2_map),
                                 adjusted_rand_score(active_classes, back_3_map)])
            
            #If all the ARS scores to the previous 4 loops are above a value, escape!
            converged = ARS_back > 0.85
            if all(converged):
                break
            
            
        
    model_tabel_values = [model_type, random_seed, local_ground_truth, x_seed, y_seed, al_results_table_values]
    
    
    queue.put(model_tabel_values)
    return model_tabel_values

# test = Run_an_AL_campaign(param_list[0])
    
if __name__ == '__main__':
    data_deamon = DataDaemon(db_name = 'results_13.db', overwrite=False)
    
    queue = data_deamon.start(chunksize=10)
    '''
    Generate all the parameters to sweep over
    '''
    param_list = []
    
    i = 0
    while i < 10:
        param_list.append(['Homoscedastic', i, queue])
        # i +=1
        param_list.append(['Heteroscedastic', i, queue])
        i +=1
    
    

    
    p = multiprocessing.Pool(8)
    for _ in tqdm.tqdm(p.imap_unordered(Run_an_AL_campaign, param_list), total = len(param_list[0])):
        pass
    
    data_deamon.stop()


# x_seed = 110
# y_seed = 110
# sub_input_loc = choose_window(x_seed, y_seed, input_loc, window_size = 51)