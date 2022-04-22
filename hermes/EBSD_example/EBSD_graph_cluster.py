# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:55:49 2021

@author: asm6

!Assumes Oh symmetry of the cyrstal!!!
"""
import numpy as np

import networkx as nx
from cdlib import algorithms

from scipy.spatial import Delaunay

from orix.quaternion.orientation import Misorientation
from orix.quaternion.symmetry import Oh

''' Function to convert the hard clustering into a probabilitic clustering
'''
def get_local_membership_prob(Graph, clusters, scale):
  '''Given the graph of measurements, and the initial clustering,
     get the probabilites of the cluster membership 
     (using the graph connectivity)'''

  #Label the graph with the clusters
  for k in range(len(clusters.communities)):
    K = clusters.communities[k]
    for i in K:
      nx.set_node_attributes(Graph, {i: k}, name='Labels')

  labels = np.asarray(Graph.nodes.data(data='Labels'))[:,1]

  ## Get membership probabilites:
  '''Create a containers for the distances, connections, and similarities 
       of each node for each label.
       Each lable will be that row number, each node will be that column number. 
  '''
  max_labels = np.max(np.unique(np.array(list(nx.get_node_attributes(Graph, name='Labels').values()))))
  max_nodes = np.max(list(Graph.nodes))
  cumulative_distance_maxtrix = np.zeros((max_labels+1, max_nodes+1))
  connection_matrix = np.zeros((max_labels+1, max_nodes+1))
  similarity_matrix = np.zeros((max_labels+1, max_nodes+1))

  for node in Graph.nodes(data=True):
        node_L = node[1]['Labels']
        #Add the self-connection of this node to its label
        connection_matrix[node_L,node[0]] += 1
        #Find all the nieghbors and iterate over them.
        edges = Graph.edges(node[0], data='Distance')
        for edge in edges:
            connected_node_L = Graph.nodes.data(data='Labels')[edge[1]]
            #Add the connection of this node to the neighboring label
            connection_matrix[connected_node_L,node[0]] += 1
            #Add the distance of this node to the nieghboring lable
            cumulative_distance_maxtrix[connected_node_L, node[0]] += edge[2]

  average_distance_matrix = cumulative_distance_maxtrix/connection_matrix
  std = np.nanstd(average_distance_matrix)
  mean = np.nanmean(average_distance_matrix)
  #Similarity of each node to each cluster
  # similarity_matrix = np.exp(-(average_distance_matrix/mean)**2)
  # similarity_matrix = np.exp(-(average_distance_matrix/0.015)**2)
  similarity_matrix = np.exp(-(average_distance_matrix/scale)**2)
  #Membership probabiltiy of each node to each cluster
  node_sum  = np.nansum(similarity_matrix,axis=0) #sum of similarities across the clusters for each node.
  probability_matrix = np.nan_to_num(similarity_matrix/node_sum, 0)
    
  for node in Graph.nodes(data=True):
        node_L = node[1]['Labels']
        certainty = probability_matrix[node_L, node[0]]
        nx.set_node_attributes(Graph, {node[0]: certainty}, name='Certainty')
        
  P_ik = probability_matrix.T
    
  return Graph, labels, P_ik
      
 

def get_global_membership_prob(Graph, clusters, measurements, distance_table):
  ### Given the graph of measurements, and the initial clustering,
  ###    get the probabilites of the cluster membership (disregarding the graph connectivity)


  #Label the graph with the clusters
  for k in range(len(clusters.communities)):
    K = clusters.communities[k]
    for i in K:
      nx.set_node_attributes(Graph, {i: k}, name='Labels')

  labels = np.asarray(Graph.nodes.data(data='Labels'))[:,1]

  ## Get membership probabilites:
  
  #Get the distances of each point to the members of each of the clusters
  s_ik = np.zeros((len(Graph.nodes), len(clusters.communities)))

  for k in range(len(clusters.communities)):
    for i in range(len(Graph.nodes)):
      #Find the number of nodes in the community not counting the current node
      b = [num for num in clusters.communities[k] if num != i]  
      b = len(b)

      d = np.zeros(())
      
      for j in clusters.communities[k]:
        if j != i:
          d = np.append(d, distance_table[j,i])

      d = np.sum(d)

      if d != 0:
        s_ik[i, k] = b/d  

  # Convert the distances to a membership probablity
  v = 1
  P_ik = np.zeros_like(s_ik)
  new_sum = np.sum(s_ik**v, axis= 1)

  for i in range(new_sum.size):
    P_ik[i,:] = s_ik[i,:]**v/new_sum[i]

  
  # Certainty of the node belonging to the selected cluster
  certainty = np.zeros_like(P_ik[:,0])

  for i in range(P_ik[:,0].size):
    certainty[i] = P_ik[i, labels[i]]
    nx.set_node_attributes(Graph, {i: certainty[i]}, name='Certainty')

  new_clusters = clusters
  # Test if a cluster only has one member, and re-assign to most probable
  for k in range(len(new_clusters.communities)):
    members = new_clusters.communities[k]
    num_members = len(members)
    if num_members == 1:
      most_prob = np.argmax(P_ik[members[0],:]) # the most probable cluster for that point to be in
      if most_prob != k:      
          new_clusters.communities[np.argmax(P_ik[members[0],:])].append(members[0])
          #print('Re-assigning')
      elif most_prob == k:
          sorted_prob = np.argsort(P_ik[members[0],:]) #sorted list of probable clusters
          next_most_prob = sorted_prob[1] # the second most probable (because it is already in the most probable by itself)
          new_clusters.communities[next_most_prob].append(members[0])
          #print('Re-assigning PATH B')
      del new_clusters.communities[k][0]
  
  k = 0
  while k < len(new_clusters.communities):
    members = new_clusters.communities[k]
    num_members = len(members)
    if num_members == 0:
        del new_clusters.communities[k]
        k = 0 #restart the loop
        #print('Cleaning')
    else:
        k += 1 
  
  # If a member was re-assigned, need to re-run to get the probablities
  if len(new_clusters.communities) != len(P_ik[0,:]):
    #print('Running prob calc again!!')
    Graph, labels, P_ik = get_global_membership_prob(Graph, new_clusters, measurements, distance_table)
  
  return Graph, labels, P_ik

def cluster(inputs, measurements, scale = 0.015, resolution = 0.015):
    '''Cluster based on misorientation distance of EBSD pattern 
    assuming Oh symmetry!!
    
    Forms a Delauny Triangulation graph in 3D, then applies a graph cut.
    '''
    # Inputs are the measurement locations and the EBSD Spectra
    # Outputs the labels and thier uncertainty 


    #Measure all the distances:
    distance_table = Misorientation((~measurements).outer(measurements)).set_symmetry(Oh,Oh).angle.data
  
    #Convert distances to wieghts:
    weight_table = np.exp(-(distance_table/scale)**2)
  
    ##### CREATE THE GRAPH #######
    #Create a graph from a Delauny Triangulation:
    #Create the Adjacency Matrix from the Delauny Triangulation
    adj_matrix = np.zeros((inputs[:,0].size,inputs[:,0].size))
    #Check if the data is all on the same layer:
    Is_2d = np.std(inputs[:,2]) < 10e-6
    
    if Is_2d:
        Tri = Delaunay(inputs[:,0:2], qhull_options='i QJ')

        for i in range(np.shape(Tri.simplices)[0]):
            adj_matrix[Tri.simplices[i,0], Tri.simplices[i,1]] = 1
            adj_matrix[Tri.simplices[i,1], Tri.simplices[i,0]] = 1

            adj_matrix[Tri.simplices[i,0], Tri.simplices[i,2]] = 1
            adj_matrix[Tri.simplices[i,2], Tri.simplices[i,0]] = 1

            adj_matrix[Tri.simplices[i,1], Tri.simplices[i,2]] = 1
            adj_matrix[Tri.simplices[i,2], Tri.simplices[i,1]] = 1

    else:
        Tri = Delaunay(inputs, qhull_options='i QJ')
      
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
    positions = list(map(tuple, inputs))

    #Create a graph from the list of nodes and edges
    Graph = nx.Graph()
    for i in range(inputs[:,0].size):
        Graph.add_node(i, pos = positions[i])
    Graph.add_edges_from(edges)

    ##### APPLY DISTANCE & SIMILARITY MEASURE #########
    for i in range(np.array(Graph.edges).shape[0]):
        j = np.array(Graph.edges)[i,0]
        k = np.array(Graph.edges)[i,1]
        nx.set_edge_attributes(Graph, {(j,k): distance_table[j,k]}, name= 'Distance' )
        nx.set_edge_attributes(Graph, {(j,k): weight_table[j,k]}, name= 'weight' )

    ###### APPLY CLUSTERING ALGORITHM ########
    cluster = algorithms.rb_pots(Graph, weights='weight', resolution_parameter=resolution) #For Cosine Distance
    #print('Rough Clustering')

    # Graph, labels, probabilities = get_membership_prob(Graph, cluster, measurements, distance_table)
    Graph, labels, probabilities = get_local_membership_prob(Graph, cluster, scale)
    #print('Cleaned')
    C = len(probabilities[0,:])

    ###### Return the labels and probabilities for each point ######
    ###### Return the number of clusters
    ###### Return the Graph for visualization purposes #############
    return labels, probabilities, C, Graph
