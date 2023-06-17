# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:15:14 2021

@author: asm6

array storing functions from https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
"""
import numpy as np
import io
import sqlite3
from contextlib import closing
import multiprocessing 
import copy

compressor = 'zlib'  # zlib, bz2

def adapt_array(arr):
    return arr.tobytes()
    # """
    # http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    # """
    # # zlib uses similar disk size that Matlab v5 .mat files
    # # bz2 compress 4 times zlib, but storing process is 20 times slower.
    # out = io.BytesIO()
    # np.save(out, arr)
    # out.seek(0)
    # return sqlite3.Binary(out.read().encode(compressor))  # zlib, bz2

def convert_array(text):
    return np.frombuffer(text)
    # out = io.BytesIO(text)
    # out.seek(0)
    # out = io.BytesIO(out.read().decode(compressor))
    # return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

class DataDaemon:
    def __init__(self,db_name='db/results.db',overwrite=False):
        self.db_name = db_name
        self.daemon = None
        self.manager = None
        self.queue = None
        
        with closing(sqlite3.connect(db_name)) as connection:
            with closing(connection.cursor()) as cursor:
                
                if overwrite:
                    cursor.execute('DROP TABLE IF EXISTS model_table')
                    cursor.execute('DROP TABLE IF EXISTS al_results_tabel')
                    connection.commit()
                    
                cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="model_table"')
                if cursor.fetchone() is None: # table exists
                    # cursor.execute(
                    #     "CREATE TABLE model_table"
                    #     "("
                    #     "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    #     "model TEXT, "
                    #     "random_seed INTEGER, "
                    #     "ground_truth ARRAY, "
                    #     "x_seed INTEGER, "
                    #     "y_seed INTEGER, "
                    #     ")"
                    # )
                    cursor.execute("CREATE TABLE model_table" 
                                   "("
                                   "id INTEGER PRIMARY KEY AUTOINCREMENT," 
                                   "model VARCHAR(255), "
                                   "random_seed INTEGER, "
                                   "ground_truth ARRAY, "
                                   "x_seed INTEGER," 
                                   "y_seed INTEGER"
                                   ")"
                                   )
                    
                    cursor.execute(
                        "CREATE TABLE al_results_table"
                        "("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "model_id INTEGER, "
                        "loop_index INTEGER, "
                        "measured_index ARRAY, "
                        "next_points_index ARRAY, "
                        "label_map ARRAY, "
                        "uncertainty_map ARRAY, "
                        "adjusted_rand_score REAL"
                        ")"
                    )
                    
        
                
    def start(self,chunksize=1):
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue()
        
        if chunksize>1:
            self.daemon = multiprocessing.Process(target=DataDaemon._runloop_chunked, args=(self.queue,self.db_name,chunksize))
        else:
            self.daemon = multiprocessing.Process(target=DataDaemon._runloop, args=(self.queue,self.db_name))
        self.daemon.daemon = True
        self.daemon.start()        
        return self.queue
    
    @staticmethod
    def _runloop_chunked(queue,db_name,chunksize):
        # #create ground_truth_mapping
        # with closing(sqlite3.connect(db_name)) as connection:
        #     with closing(connection.cursor()) as cursor:
        #         cursor.execute(f"SELECT (id,npts) FROM ground_truth_plots")
        #         mapping = {i[1]:i[0] for k in cursor.fetchall()}
        
        chunked = []
        finished = False
        while (not finished):
            q_item = queue.get()
            if q_item is None:
                finished = True
            else:
                chunked.append(q_item)
                
            if (len(chunked)>=chunksize) or (finished and len(chunked)>0):
                with closing(sqlite3.connect(db_name)) as connection:
                    with closing(connection.cursor()) as cursor:
                        for model_table_values in chunked:
                            cursor.execute("INSERT INTO model_table(model, random_seed, ground_truth, x_seed, y_seed) VALUES (?,?,?,?,?)",
                                           (model_table_values[0], model_table_values[1], model_table_values[2], model_table_values[3], model_table_values[4]))
                            result_id = copy.copy(cursor.lastrowid)
                            
                            # print('model_table_values:', model_table_values)
                            # print('al_results_table_values', model_table_values[5])
                            id_list = []
                            for i in range(len(model_table_values[5])):
                                id_list.append(result_id)
                            for i, t in enumerate(model_table_values[5]):
                                t.insert(0, id_list[i])
                            print('al_results_table_values', model_table_values[5])    
                            cursor.executemany("INSERT INTO al_results_table(model_id, loop_index, measured_index, next_points_index, label_map, uncertainty_map, adjusted_rand_score) VALUES(?,?,?,?,?,?,?)",
                                                   model_table_values[5])
                            # #get ground_truth id based on
                            # cursor.execute(f"SELECT id FROM ground_truth_plots WHERE npts=={params[0]}")
                            # ground_truth_id = cursor.fetchone()[0]
                            
                            # params_plots = list(params)
                            # params_plots.append(result_id)
                            # params_plots.append(ground_truth_id)
                            
                            # qmarks = '?,'*len(params_plots)
                            # cursor.execute("INSERT INTO params(npts, noise, method, affinity, co_affinity, gamma, degree, c0, co_gamma, fms, result_plot_id, ground_truth_plot_id) VALUES ("+qmarks[:-1]+")",params_plots)
                    
                    connection.commit()
                chunked = []
                
    @staticmethod
    def _runloop(queue,db_name):
        while True:
            q_item = queue.get()
            if q_item is None:
                break
            model_table_values = q_item
                
            with closing(sqlite3.connect(db_name)) as connection:
                with closing(connection.cursor()) as cursor:
                    cursor.execute("INSERT INTO model_table(model, random_seed, ground_truth, x_seed, y_seed) VALUES (?,?,?,?,?)",
                                   (model_table_values[0], model_table_values[1], model_table_values[2], model_table_values[3], model_table_values[4]))
                    result_id = copy.copy(cursor.lastrowid)
                    
                    # print('model_table_values:', model_table_values)
                    # print('al_results_table_values', model_table_values[5])
                    id_list = []
                    for i in range(len(model_table_values[5])):
                        id_list.append(result_id)
                    for i, t in enumerate(model_table_values[5]):
                        t.insert(0, id_list[i])
                    print('al_results_table_values', model_table_values[5])    
                    cursor.executemany("INSERT INTO al_results_table(model_id, loop_index, measured_index, next_points_index, label_map, uncertainty_map, adjusted_rand_score) VALUES(?,?,?,?,?,?,?)",
                                           model_table_values[5])
                    # #get ground_truth id based on
                    # cursor.execute(f"SELECT id FROM ground_truth_plots WHERE npts=={params[0]}")
                    # ground_truth_id = cursor.fetchone()[0]
                    
                    # params_plots = list(params)
                    # params_plots.append(result_id)
                    # params_plots.append(ground_truth_id)
                    
                    # qmarks = '?,'*len(params_plots)
                    # cursor.execute("INSERT INTO params(npts, noise, method, affinity, co_affinity, gamma, degree, c0, co_gamma, fms, result_plot_id, ground_truth_plot_id) VALUES ("+qmarks[:-1]+")",params_plots)
                    
                    
                connection.commit()
                
    def stop(self):
        self.queue.put(None)#write_daemon will stop at None
        self.daemon.join()#wait for write_daemon to hit None