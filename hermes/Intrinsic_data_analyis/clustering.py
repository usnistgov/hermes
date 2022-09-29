# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:57:27 2022

@author: Austin McDannald
"""

from . import Intrinsic_data_analysis

import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
# class Distance_measures(Intrinsic_data_analysis):

# class Similarity_measures(Intrinsic_data_analysis):
#     sim = exp(-distance^2)
    
#     sim = 1/(dist+epsilon)
    

class Clustering(Intrinsic_data_analysis):
    """Class for clustering algorithms."""
    def __init__(self, locations, measurement_similarity):
        pass
    
    def Get_global_membership_prob(cluster_labels, measurement_similarity, v=1, exclude_self=False):
        """Get the probability of each measurement beloning to each cluster."""
        
        #cluster_labels is an array of the labels for each measurement.
        #v is a parameter that adjusts the strentgh of the partitioning with the similarities
        
        #Find the clusters
        clusters, counts = np.unique(cluster_labels, return_counts = True)
        #One hot encoding of the labels
        one_hot = cluster_labels.reshape(-1,1) == clusters.reshape(1,-1)
        #n x n x clusters stack of one hot encoding        
        stack = np.ones_like(measurement_similarity[:,:,np.newaxis])@one_hot[:,np.newaxis,:]
        if exclude_self == True:
            #Identity matrix
            I = np.eye(cluster_labels.shape[0])
            #Block identity tensor
            I_s = I[:,:,np.newaxis]@one_hot[:,np.newaxis,:]
            #stack excluding self 
            stack -=  I_s
        
        #n x n x clusters similarity tensor stack
        sims = measurement_similarity[:,:,np.newaxis]@np.ones((1,1,len(clusters)))
        #Block similarity tensor
        block_sim = stack*sims
        #count the number of members in each cluster to consider
        cluster_counts = np.sum(stack, axis=0)
        #Sum the similarities of each measurement to each member of each cluster
        cluster_sim_sums = np.sum(block_sim, axis=0)
        #Average similarity of each measurement to each cluster
        ave_cluster_sim = cluster_sim_sums/cluster_counts
        #Sum of similarities of each measurement to all clusters
        sum_cluster_sim = np.sum(ave_cluster_sim**v, axis=1).reshape(-1,1)
        #Convert cluster similarities to probaiblities
        probabilities = ave_cluster_sim**v/sum_cluster_sim
        return probabilities
        

class Contiguous_Clustering(Clustering):
    """Use this algorthim to cluster data in domains with a contigious constraint.
    Example domains where this applies: Phase regions in a phase diagram, 
    grains in a micrograph, etc.
    
    Locations of measurements are used to form a graph.
    The similarities of those measureements are used as wieghts for the edges of that graph.
    The graph is partitioned to form the clusters."""
    
    
    def __init__(self, locations, measurement_similarity):
        pass
    
    
    def Form_graph(locations, measurement_distance, measurement_similarity):
        """Forms a graph based on the measurement locations
        using a Delauny Triangulation. This type of graph will preserve the 
        contiguous constraint when cut.
        Assigns the measurement distance and similarity as edge attributes.
        Returns a networkx graph object."""
        
        
        #Create the Adjacency Matrix to fill from the Delauny Triangulation
        adj_matrix = np.zeros((locations[:,0].size,locations[:,0].size))
        #Check if the data is all on the same layer:
        Is_2d = np.std(locations[:,2]) < 10e-6
   
        if Is_2d:
            Tri = Delaunay(locations[:,0:2], qhull_options='i QJ')

            for i in range(np.shape(Tri.simplices)[0]):
               adj_matrix[Tri.simplices[i,0], Tri.simplices[i,1]] = 1
               adj_matrix[Tri.simplices[i,1], Tri.simplices[i,0]] = 1

               adj_matrix[Tri.simplices[i,0], Tri.simplices[i,2]] = 1
               adj_matrix[Tri.simplices[i,2], Tri.simplices[i,0]] = 1

               adj_matrix[Tri.simplices[i,1], Tri.simplices[i,2]] = 1
               adj_matrix[Tri.simplices[i,2], Tri.simplices[i,1]] = 1

        else:
           Tri = Delaunay(locations, qhull_options='i QJ')
     
           for i in range(np.shape(Tri.simplices)[0]):
               adj_matrix[Tri.simplices[i,0], Tri.simplices[i,1]] = 1
               adj_matrix[Tri.simplices[i,1], Tri.simplices[i,0]] = 1
               
               adj_matrix[Tri.simplices[i,0], Tri.simplices[i,2]] = 1
               adj_matrix[Tri.simplices[i,2], Tri.simplices[i,0]] = 1
   
               adj_matrix[Tri.simplices[i,0], Tri.simplices[i,3]] = 1
               adj_matrix[Tri.simplices[i,3], Tri.simplices[i,0]] = 1
   
               adj_matrix[Tri.simplices[i,1], Tri.simplices[i,2]] = 1
               adj_matrix[Tri.simplices[i,2], Tri.simplices[i,1]] = 1
   
               adj_matrix[Tri.simplices[i,1], Tri.simplices[i,3]] = 1
               adj_matrix[Tri.simplices[i,3], Tri.simplices[i,1]] = 1
   
               adj_matrix[Tri.simplices[i,2], Tri.simplices[i,3]] = 1
               adj_matrix[Tri.simplices[i,3], Tri.simplices[i,2]] = 1

 
        #Convert the Ajacency Matrix into a list of edges between nodes
        rows, columns = np.where(adj_matrix ==  1)

        edges = np.concatenate((rows.reshape(-1,1), columns.reshape(-1,1)), axis = 1)
        edges = list(map(tuple, edges))

        #Create a list of nodes from the positions
        positions = list(map(tuple, locations))

        #Create a graph from the list of nodes and edges
        Graph = nx.Graph()
        for i in range(locations[:,0].size):
           Graph.add_node(i, pos = positions[i])
        Graph.add_edges_from(edges)

        ##### APPLY DISTANCE & SIMILARITY MEASURE #########
        for i in range(np.array(Graph.edges).shape[0]):
           j = np.array(Graph.edges)[i,0]
           k = np.array(Graph.edges)[i,1]
           nx.set_edge_attributes(Graph, {(j,k): measurement_distance[j,k]}, name= 'Distance' )
           nx.set_edge_attributes(Graph, {(j,k): measurement_similarity[j,k]}, name= 'Weight' )
        
        return Graph
        
    def Get_local_membership_prob(graph, clusters, scale):
        """Get the membership proabilities of each measurement beloning to each cluster
        considering the structure of the graph."""
        
        # Do a thing
        return graph, labels, P_ik
    
class Contiguous_Fixed_K_Clustering(Contiguous_Clustering):
    """Use these algorithms when the number of clusters is known."""
    def __init__(self, locations, measurement_similarity):
        self.graph = self.Form_graph(locations, measurement_similarity)
        pass
    
    @classmethod
    def Spectral(locactions, graph):
        # do spectral clustering using the weighted graph as the affinity. 
        return labels
    
    
class Contigous_Community_Discovery(Contiguous_Clustering):
    """Use these algorithms when the number of clusters is not known."""
    
    def __init__(self, locations, measurement_similarity):
        pass
    
    @classmethod
    def RB_pots(locations, graph):
        return lables
    
    @classmethod
    def gl_expansion():
        return lables
    
    
    @classmethod
    def iteriative_fixed_k(locations, graph):
        """Call a fixed k clustering method iteratively 
        using the Gap Statisic method to choose K."""
        
        

    
    
class test(Clustering):
    """Testing"""
    def __init__(self):
        #self
        pass
    
    @classmethod
    def test(cls):
        print("Success!")