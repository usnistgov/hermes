# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:48:28 2021

@author: asm6
"""

import sqlite3
import pandas as pd


import numpy as np
from matplotlib import pyplot as plt


# con = sqlite3.connect("C:/Users/asm6/results_9.db") 
db_number = 17
con = sqlite3.connect(f"C:/Users/asm6/Documents/Projects/EBSD/EBSD analysis/EBSD 2D active learning loops/From NISABA/results_{db_number}.db") 

db = pd.read_sql('SELECT * FROM model_table', con)

db.info()

'''Read and prind ground truth array'''
# print(np.frombuffer(db['ground_truth'][0], dtype = np.int64))

db_2 = pd.read_sql('SELECT * FROM al_results_table', con)

'''Read and print items from the al_results_table'''
# print(np.frombuffer(db_2['measured_index'][0], dtype=np.int32))
# print(np.frombuffer(db_2['next_points_index'][0], dtype=np.int32))
# print(np.frombuffer(db_2['label_map'][0], dtype=np.float64))
# print(np.frombuffer(db_2['uncertainty_map'][0], dtype=np.float64))


db_3 = pd.read_sql('SELECT * FROM model_table JOIN al_results_table ON model_table.id = al_results_table.model_id', con)
#count how many models of a certain type made it to a certain loop:
# print(db_3[(db_3['model']=='Heteroscedastic') & (db_3['loop_index'] == 12)])

'''Average ARS for all each type of model at each loop iteration'''
mean_no = db_3[db_3['model'] == 'Homoscedastic'].groupby('loop_index')['adjusted_rand_score'].mean()
std_no = db_3[db_3['model'] == 'Homoscedastic'].groupby('loop_index')['adjusted_rand_score'].std()

mean_w_ucp = db_3[db_3['model'] == 'Heteroscedastic'].groupby('loop_index')['adjusted_rand_score'].mean()
std_w_ucp = db_3[db_3['model'] == 'Heteroscedastic'].groupby('loop_index')['adjusted_rand_score'].std()

'''Plot the averge learning curves'''
loops = np.arange(db_3['loop_index'].max()+1)
plt.figure(figsize = (10,10))
plt.fill_between(loops[0:mean_no.shape[0]], mean_no - 2*std_no, mean_no + 2*std_no,alpha=0.5, label ='Homoscedastic CI')
plt.fill_between(loops[0:mean_w_ucp.shape[0]], mean_w_ucp - 2*std_w_ucp, mean_w_ucp + 2*std_w_ucp, alpha=0.5, label='Heteroscedastic CI')
plt.plot(loops[0:mean_no.shape[0]], mean_no, label = 'Homoscedastic')
plt.plot(loops[0:mean_w_ucp.shape[0]], mean_w_ucp, label = 'Heteroscedastic')
plt.legend()
plt.xlabel('AL Loops')
plt.ylabel('ARS') 
plt.show()

'''Plot each of the learning curves'''
no_noise_random_seed = np.array(db_3[db_3['model'] == 'Homoscedastic']['random_seed']).reshape(-1,1)
no_noise_loop_index = np.array(db_3[db_3['model'] == 'Homoscedastic']['loop_index']).reshape(-1,1)
no_noise_ars = np.array(db_3[db_3['model'] == 'Homoscedastic']['adjusted_rand_score']).reshape(-1,1)
all_loops_no = np.concatenate((no_noise_random_seed, no_noise_loop_index, no_noise_ars), axis=1)

ucp_random_seed = np.array(db_3[db_3['model'] == 'Heteroscedastic']['random_seed']).reshape(-1,1)
ucp_loop_index = np.array(db_3[db_3['model'] == 'Heteroscedastic']['loop_index']).reshape(-1,1)
ucp_ars = np.array(db_3[db_3['model'] == 'Heteroscedastic']['adjusted_rand_score']).reshape(-1,1)
all_loops_ucp = np.concatenate((ucp_random_seed, ucp_loop_index, ucp_ars), axis=1)


plt.figure(figsize = (10,10))
# for seed in np.unique(no_noise_random_seed):
#     plt.plot(all_loops_no[all_loops_no[:,0]==seed][:,1], 
#              all_loops_no[all_loops_no[:,0]==seed][:,2], 'o-')#r-')

# for seed in np.unique(ucp_random_seed):
#     plt.plot(all_loops_ucp[all_loops_ucp[:,0]==seed][:,1], 
#              all_loops_ucp[all_loops_ucp[:,0]==seed][:,2], 'o--', mfc='none')#b*')
for seed in np.unique(no_noise_random_seed):
    plt.plot(all_loops_no[all_loops_no[:,0]==seed][:,1], 
             all_loops_no[all_loops_no[:,0]==seed][:,2], 'o-')#r-')
    plt.plot(all_loops_ucp[all_loops_ucp[:,0]==seed][:,1], 
             all_loops_ucp[all_loops_ucp[:,0]==seed][:,2], 'o--', mfc='none')#b*')
    
    plt.show()


plt.legend()
plt.xlabel('AL Loops')
plt.ylabel('ARS') 
plt.show()




'''Find the best Homoscedastic model, 
then plot the label and uncertainty maps for that, 
and the coorisponding Heteroscedasic model'''
arg = db_3[db_3['model']=='Homoscedastic']['adjusted_rand_score'].argmax()

label_map_no = db_3['label_map'][arg]
u_map_no = db_3['uncertainty_map'][arg]

rand_key = db_3['random_seed'][arg]

new_arg = db_3[(db_3['random_seed']==rand_key) 
               & (db_3['model']=='Heteroscedastic')]['loop_index'].idxmax()

label_map_w_ucp = db_3['label_map'][new_arg]
u_map_w_ucp = db_3['uncertainty_map'][new_arg]


"""!!!!!! You are here!!!!!"""
'''Plot the maps'''
# plt.plot()


con.close()