# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:57:27 2022

@author: Austin McDannald
"""

# from . import Intrinsic_data_analysis
# import Intrinsic_data_analysis
from .clustering.Base import Intrinsic_data_analysis

import numpy as np

import networkx as nx
from cdlib import algorithms

from scipy.spatial import Delaunay
from sklearn.cluster import SpectralClustering
   

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
    
    @classmethod    
    def Get_local_membership_prob(graph, cluster_labels, v = 1):
        """Get the membership proabilities of each measurement beloning to each cluster
        considering the structure of the graph.
        #v is a parameter that adjusts the strentgh of the partitioning with the similarities
        Get membership probabilites:
        Create a containers for the distances, connections, and similarities 
        of each node for each label.
        Each label will be that row number, each node will be that column number. 
        """
        
        
        max_labels = np.max(cluster_labels)+1
        max_nodes = len(cluster_labels)
        cumulative_distance_maxtrix = np.zeros((max_labels, max_nodes))
        cumulative_similarity_maxtrix = np.zeros((max_labels, max_nodes))
        connection_matrix = np.zeros((max_labels, max_nodes))


        for node in graph.nodes(data=True):
            node_L = node[1]['Labels']
            #Add the self-connection of this node to its label
            connection_matrix[node_L,node[0]] += 1
            #Find all the nieghbors and iterate over them.
            edges = graph.edges(node[0], data = True)
            for edge in edges:
                connected_node_L = graph.nodes.data(data='Labels')[edge[1]]
                #Add the connection of this node to the neighboring label
                connection_matrix[connected_node_L,node[0]] += 1
                #Add the distance and wieght of this node to the nieghboring label
                cumulative_distance_maxtrix[connected_node_L, node[0]] += edge[2]["Distance"]
                cumulative_similarity_maxtrix[connected_node_L, node[0]] += edge[2]["Weight"]

        #Find the average distance and similarity 
        #of each node to its nieghboring labels:
        average_distance_matrix = cumulative_distance_maxtrix/connection_matrix
        average_similarity_matrix = cumulative_similarity_maxtrix/connection_matrix
        
        #Convert the average similarities to probabilities 
        node_sum  = np.nansum(average_similarity_matrix**v,axis=0) #sum of similarities across the clusters for each node.
        probability_matrix = np.nan_to_num(average_similarity_matrix**v/node_sum, 0)
        probabilities = probability_matrix.T
        return probabilities
    
class Contiguous_Fixed_K_Clustering(Contiguous_Clustering):
    """Use these algorithms when the number of clusters is known."""
    def __init__(self, locations, measurement_similarity):
        self.graph = self.Form_graph(locations, measurement_similarity)
        pass
    
    @classmethod
    def Spectral(locactions, graph,
                 n_clusters, eigen_solver=None, n_components=None, 
                 random_state=None, n_init=10, eigen_tol=0.0, assign_labels='kmeans', 
                 n_jobs=None, verbose=False):
        
        matrix = nx.adjacency_matrix(graph, weight = "Weight")
        affinity = matrix.toarray()
        
        clusters = SpectralClustering(n_clusters, affinity = "precomputed", eigen_solver=None, n_components=None, 
                                      random_state=None, n_init=10, eigen_tol=0.0, assign_labels='kmeans', 
                                      n_jobs=None, verbose=False).fit(affinity)
        labels = clusters.labels_
        
        """Get the local membership probabilities"""
        graph, labels, probabilities =  super().Get_local_membership_prob(graph, labels, v = 1)
        
        return graph, labels, probabilities
    
    
class Contigous_Community_Discovery(Contiguous_Clustering):
    """Use these algorithms when the number of clusters is not known."""
    
    def __init__(self, locations, measurement_similarity):
        pass
    
    @classmethod
    def RB_pots(graph, resolution = 0.015):
        """Cut the graph using RB_pots"""
        clusters = algorithms.rb_pots(graph, weights='weight', resolution_parameter=resolution)
        
        #Label the graph with the clusters
        for k in range(len(clusters.communities)):
            K = clusters.communities[k]
            for i in K:
                nx.set_node_attributes(graph, {i: k}, name='Labels')

        labels = np.asarray(graph.nodes.data(data='Labels'))[:,1]
        
        """Get the local membership probabilities"""
        graph, labels, probabilities =  super().Get_local_membership_prob(graph, labels, v = 1)
        
        return graph, labels, probabilities
    
    @classmethod
    def generaric_cdlib(graph, method_name, **kwargs):
        """call the community discovery method within CDLIB"""
        method = getattr(algorithms, method_name)
        clusters = method(graph, **kwargs)
        
        #Label the graph with the clusters
        for k in range(len(clusters.communities)):
            K = clusters.communities[k]
            for i in K:
                nx.set_node_attributes(graph, {i: k}, name='Labels')

        labels = np.asarray(graph.nodes.data(data='Labels'))[:,1]
        
        """Get the local membership probabilities"""
        graph, labels, probabilities =  super().Get_local_membership_prob(graph, labels, v = 1)
        return graph, labels, probabilities
    
    
    @classmethod
    def iteriative_fixed_k(locations, graph):
        """Call a fixed k clustering method iteratively 
        using the Gap Statisic method to choose K."""
        
        print("Not implmented yet")

    
    
class test(Clustering):
    """Testing"""
    def __init__(self):
        #self
        pass
    
    @classmethod
    def test(cls):
        print("Success!")