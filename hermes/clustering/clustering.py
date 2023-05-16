# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:57:27 2022

@author: Austin McDannald
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pydantic.dataclasses import dataclass as typesafedataclass

import networkx as nx
from cdlib import algorithms
import numpy as np
from scipy.spatial import Delaunay
from sklearn.cluster import SpectralClustering

from hermes.distance import EuclideanDistance
from hermes.similarity import SquaredExponential
from hermes.utils import _default_ndarray

# from hermes.schemas import PrivateAttr
from hermes.distance import BaseDistance

from hermes.similarity import BaseSimilarity
from hermes.base import Analysis

# class Distance_measures(Intrinsic_data_analysis):

# class Similarity_measures(Intrinsic_data_analysis):
#     sim = exp(-distance^2)

#     sim = 1/(dist+epsilon)


class UnspecifiedType(Exception):
    """Raised when no Distance or Similarity type is specified."""

    pass


# BaseSimilarity = TypeVar("BaseSimilarity", bound=BaseSimilarity)
# BaseDistance = TypeVar("BaseDistance", bound=BaseDistance)


def _compute_distance(
    type_: BaseDistance, X: np.ndarray, Y: Optional[np.ndarray] = None, **kwargs
):
    return type_.calculate(X, Y, **kwargs)  # type: ignore


class _Config:
    arbitrary_types_allowed = True
    # validate_assignment = True


@typesafedataclass(config=_Config)
class Cluster(Analysis):
    """Class for clustering algorithms."""

    measurements: np.ndarray
    measurements_distance_type: BaseDistance = EuclideanDistance()
    measurements_similarity_type: BaseSimilarity = SquaredExponential()
    measurements_distance: np.ndarray = field(
        init=False, default_factory=_default_ndarray, repr=False
    )
    measurements_similarity: np.ndarray = field(
        init=False, default_factory=_default_ndarray, repr=False
    )

    locations: np.ndarray = field(default_factory=_default_ndarray)
    locations_distance_type: BaseDistance = EuclideanDistance()
    locations_similarity_type: BaseSimilarity = SquaredExponential()
    locations_distance: np.ndarray = field(init=False, default_factory=_default_ndarray)
    locations_similarity: np.ndarray = field(
        init=False, default_factory=_default_ndarray, repr=False
    )

    # TODO finish this to init with params below
    '''Automactically re-calculate all the distances and similarities when the atributes are set.
    This prevents miss-labeling the distances when the type is changed after the initial calcuation.'''
    def __setattr__(self, __name: str, __value: Any):
        if __name == "measurements_distance_type":
            if not isinstance(__value, BaseDistance):
                raise TypeError("invalid distance")
            v = __value
            v.X = self.measurements  # type: ignore
            setattr(self, "measurements_distance", v.calculate())  # type: ignore
            return super().__setattr__(__name, v)
        
        if __name == "measurements_similarity_type":
            if not isinstance(__value, BaseSimilarity):
                raise TypeError("invalid distance")
            v = __value
            v.distance_matrix = self.measurements_distance  # type: ignore
            setattr(self, "measurements_similarity", v.calculate())  # type: ignore
            return super().__setattr__(__name, v)
        
        if __name == "locations_distance_type":
            if not isinstance(__value, BaseDistance):
                raise TypeError("invalid distance")
            v = __value
            v.X = self.locations
            setattr(self, "locations_distance", v.calculate())  # type: ignore
            return super().__setattr__(__name, v)
        
        if __name == "locations_similarity_type":
            if not isinstance(__value, BaseSimilarity):
                raise TypeError("invalid distance")
            v = __value
            v.distance_matrix = self.locations_distance
            setattr(self, "locations_similarity", v.calculate())  # type: ignore
            return super().__setattr__(__name, v)



        return super().__setattr__(__name, __value)

    # Can we remove all of the __post_init_post_parse__???
    def __post_init_post_parse__(self):
        self.measurements_distance_type.X = self.measurements  # type: ignore
        self.measurements_distance = self.measurements_distance_type.calculate()  # type: ignore
        self.measurements_similarity_type.distance_matrix = self.measurements_distance
        self.measurements_similarity = self.measurements_similarity_type.calculate()

        self.locations_distance_type.X = self.locations  # type: ignore
        self.locations_distance = self.locations_distance_type.calculate() # type: ignore
        self.locations_similarity_type.distance_matrix = self.locations_distance
        self.locations_similarity = self.locations_similarity_type.calculate() 
        

    def __repr__(self) -> str:
        return f"Cluster(locations={self.locations.shape}, measurements={self.measurements.shape}, locations_distance_type={self.locations_distance_type}, measurements_distance_type={self.measurements_distance_type})"

    # TODO: only create class of train hgpc, hgpc classes can be private

    # make pairwise_metrics smart enough: type defines what is x and y

    def get_global_membership_prob(
        self, cluster_labels: np.ndarray, v: float = 1.0, exclude_self: bool = False
    ):
        """Get the probability of each measurement beloning to each cluster."""

        # cluster_labels is an array of the labels for each measurement.
        # v is a parameter that adjusts the strentgh of the partitioning with the similarities

        # Find the clusters
        clusters, counts = np.unique(cluster_labels, return_counts=True)
        # One hot encoding of the labels
        one_hot = cluster_labels.reshape(-1, 1) == clusters.reshape(1, -1)
        # n x n x clusters stack of one hot encoding
        stack = (
            np.ones_like(self.measurements_similarity[:, :, np.newaxis])
            @ one_hot[:, np.newaxis, :]
        )
        if exclude_self:
            # Identity matrix
            I = np.eye(cluster_labels.shape[0])
            # Block identity tensor
            I_s = I[:, :, np.newaxis] @ one_hot[:, np.newaxis, :]
            # stack excluding self
            stack -= I_s

        # n x n x clusters similarity tensor stack
        sims = self.measurements_similarity[:, :, np.newaxis] @ np.ones(
            (1, 1, len(clusters))
        )
        # Block similarity tensor
        block_sim = stack * sims
        # count the number of members in each cluster to consider
        cluster_counts = np.sum(stack, axis=0)
        # Sum the similarities of each measurement to each member of each cluster
        cluster_sim_sums = np.sum(block_sim, axis=0)
        # Average similarity of each measurement to each cluster
        ave_cluster_sim = cluster_sim_sums / cluster_counts
        # Sum of similarities of each measurement to all clusters
        sum_cluster_sim = np.sum(ave_cluster_sim**v, axis=1).reshape(-1, 1)
        # Convert cluster similarities to probaiblities
        probabilities = ave_cluster_sim**v / sum_cluster_sim
        return probabilities


# locations = np.array([[3, 4], [1, 2]])
# measurements = np.array([2, 8])

# distance = EuclideanDistance()

# c = Cluster(
#     locations=locations,
#     locations_distance_type=distance,
#     measurements=measurements,
#     measurements_distance_type=distance,
# )
# refactor classes to take in params


# To iniliza contiguous comm cluster
# in: measurements, locations, measurement distance, meas similarity
@typesafedataclass(config=_Config)
class ContiguousCluster(Cluster):
    """Use this algorthim to cluster data in domains with a contigious constraint.
    Example domains where this applies: Phase regions in a phase diagram,
    grains in a micrograph, etc.

    Locations of measurements are used to form a graph.
    The similarities of those measureements are used as wieghts for the edges of that graph.
    The graph is partitioned to form the clusters."""
    
    # graph: nx.Graph = field(init=False)  # TODO check if nx.Graph or nx.graph
    # labels: np.ndarray = field(
    #     init=False, default_factory=_default_ndarray, repr=False
    # )
    # probabilities: np.ndarray = field(
    #     init=False, default_factory=_default_ndarray, repr=False
    # )
    
    # def __post_init__(self):
    #     self.graph = self.form_graph(
    #         # self.measurements_similarity
    #     )  # CQ is it similarity or distance? Waiting.


    def form_graph(self) -> nx.Graph:
        """Forms a graph based on the measurement locations
        using a Delauny Triangulation. This type of graph will preserve the
        contiguous constraint when cut.
        Assigns the measurement distance and similarity as edge attributes.
        Returns a networkx graph object."""

        # Create the Adjacency Matrix to fill from the Delauny Triangulation
        adj_matrix = np.zeros((self.locations[:, 0].size, self.locations[:, 0].size))
        
        # Check for dimensions of input:
        if self.locations.shape[1] == 2:
            dims = 2
        elif self.locations.shape[1] == 3:
            #check for near zero values in the z dimension  
            if np.std(self.locations[:, 2]) < 10e-6:
                dims = 2
            #TODO: check for 2D data on a 3D plane (i.e. compositions on the 3-simplex)
            else:
                dims = 3
        else:
            raise TypeError("Not implemented yet for number of dimensions")

        if dims == 2:
            tri = Delaunay(self.locations[:, 0:2], qhull_options="i QJ")

            for i in range(np.shape(tri.simplices)[0]):
                adj_matrix[tri.simplices[i, 0], tri.simplices[i, 1]] = 1
                adj_matrix[tri.simplices[i, 1], tri.simplices[i, 0]] = 1

                adj_matrix[tri.simplices[i, 0], tri.simplices[i, 2]] = 1
                adj_matrix[tri.simplices[i, 2], tri.simplices[i, 0]] = 1

                adj_matrix[tri.simplices[i, 1], tri.simplices[i, 2]] = 1
                adj_matrix[tri.simplices[i, 2], tri.simplices[i, 1]] = 1

        elif dims == 3:
            tri = Delaunay(self.locations, qhull_options="i QJ")

            for i in range(np.shape(tri.simplices)[0]):
                adj_matrix[tri.simplices[i, 0], tri.simplices[i, 1]] = 1
                adj_matrix[tri.simplices[i, 1], tri.simplices[i, 0]] = 1

                adj_matrix[tri.simplices[i, 0], tri.simplices[i, 2]] = 1
                adj_matrix[tri.simplices[i, 2], tri.simplices[i, 0]] = 1

                adj_matrix[tri.simplices[i, 0], tri.simplices[i, 3]] = 1
                adj_matrix[tri.simplices[i, 3], tri.simplices[i, 0]] = 1

                adj_matrix[tri.simplices[i, 1], tri.simplices[i, 2]] = 1
                adj_matrix[tri.simplices[i, 2], tri.simplices[i, 1]] = 1

                adj_matrix[tri.simplices[i, 1], tri.simplices[i, 3]] = 1
                adj_matrix[tri.simplices[i, 3], tri.simplices[i, 1]] = 1

                adj_matrix[tri.simplices[i, 2], tri.simplices[i, 3]] = 1
                adj_matrix[tri.simplices[i, 3], tri.simplices[i, 2]] = 1

        # Convert the Ajacency Matrix into a list of edges between nodes
        rows, columns = np.where(adj_matrix == 1)

        edges = np.concatenate((rows.reshape(-1, 1), columns.reshape(-1, 1)), axis=1)
        edges = list(map(tuple, edges))

        # Create a list of nodes from the positions
        positions = list(map(tuple, self.locations))

        # Create a graph from the list of nodes and edges
        graph = nx.Graph()
        for i in range(self.locations[:, 0].size):
            graph.add_node(i, pos=positions[i])
        graph.add_edges_from(edges)

        ##### APPLY DISTANCE & SIMILARITY MEASURE #########
        for i in range(np.array(graph.edges).shape[0]):
            j = np.array(graph.edges)[i, 0]
            k = np.array(graph.edges)[i, 1]
            nx.set_edge_attributes(
                graph, {(j, k): self.measurements_distance[j, k]}, name="Distance"
            )
            nx.set_edge_attributes(
                graph, {(j, k): self.measurements_similarity[j, k]}, name="Weight"
            )

        return graph

    def get_local_membership_prob(
        self, graph: nx.Graph, cluster_labels: np.ndarray, v: float = 1.0
    ):
        """Get the membership proabilities of each measurement beloning to each cluster
        considering the structure of the graph.
        #v is a parameter that adjusts the strentgh of the partitioning with the similarities
        Get membership probabilites:
        Create a containers for the distances, connections, and similarities
        of each node for each label.
        Each label will be that row number, each node will be that column number.
        """

        max_labels = np.max(cluster_labels) + 1
        max_nodes = len(cluster_labels)
        cumulative_distance_maxtrix = np.zeros((max_labels, max_nodes))
        cumulative_similarity_maxtrix = np.zeros((max_labels, max_nodes))
        connection_matrix = np.zeros((max_labels, max_nodes))

        for node in graph.nodes(data=True):
            node_L = node[1]["Labels"]
            # Add the self-connection of this node to its label
            connection_matrix[node_L, node[0]] += 1
            # Find all the nieghbors and iterate over them.
            edges = graph.edges(node[0], data=True)
            for edge in edges:
                connected_node_L = graph.nodes.data(data="Labels")[edge[1]]
                # Add the connection of this node to the neighboring label
                connection_matrix[connected_node_L, node[0]] += 1
                # Add the distance and wieght of this node to the nieghboring label
                cumulative_distance_maxtrix[connected_node_L, node[0]] += edge[2][
                    "Distance"
                ]
                cumulative_similarity_maxtrix[connected_node_L, node[0]] += edge[2][
                    "Weight"
                ]

        # Find the average distance and similarity
        # of each node to its nieghboring labels:
        average_distance_matrix = cumulative_distance_maxtrix / connection_matrix
        average_similarity_matrix = cumulative_similarity_maxtrix / connection_matrix

        # Convert the average similarities to probabilities
        node_sum = np.nansum(
            average_similarity_matrix**v, axis=0
        )  # sum of similarities across the clusters for each node.
        probability_matrix = np.nan_to_num(average_similarity_matrix**v / node_sum, 0)
        probabilities = probability_matrix.T
        return probabilities


@typesafedataclass(config=_Config)
class ContiguousFixedKClustering(ContiguousCluster):
    """Use these algorithms when the number of clusters is known."""
    
    # Number of clusters
    K: int = field(init=False)

    graph: nx.Graph = field(init=False)  # TODO check if nx.Graph or nx.graph

    def __post_init__(self):
        self.graph = self.form_graph(
            # self.measurements_similarity
        )  # CQ is it similarity or distance? Waiting.

    @classmethod
    def spectral(cls, graph: nx.Graph, n_clusters: int, **kwargs):
        matrix = nx.adjacency_matrix(graph, weight="Weight")
        affinity = matrix.toarray()

        clusters = SpectralClustering(n_clusters, affinity="precomputed", **kwargs).fit(
            affinity
        )
        labels = clusters.labels_
        return labels


@typesafedataclass(config=_Config)
class ContiguousCommunityDiscovery(ContiguousCluster):
    """Use these algorithms when the number of clusters is not known."""

class RBPots(ContiguousCommunityDiscovery):

    self.resolution: float

    def cluster(self):
        G = self.Graph
        res = self.resolution
        #Cluster with RB Pots Algorithm 
        clusters = algorithms.rb_pots(Graph, weights="Weight",
                                    resolution_parameter = resolution)
        
        #Label the graph with the clusters
        for k in range(len(clusters.communities)):
            K = clusters.communities[k]
            for i in K:
                nx.set_node_attributes(Graph, {i: k}, name='Labels')
        #Extract the labels
        self.labels = np.asarray(Graph.nodes.data(data='Labels'))[:,1]

#     @classmethod
#     def rb_pots(cls, Graph, resolution):
#         #Cluster with RB Pots Algorithm 
#         clusters = algorithms.rb_pots(Graph, weights="Weight",
#                                     resolution_parameter = resolution)
        
#         #Label the graph with the clusters
#         for k in range(len(clusters.communities)):
#             K = clusters.communities[k]
#             for i in K:
#                 nx.set_node_attributes(Graph, {i: k}, name='Labels')
#         #Extract the labels
#         labels = np.asarray(Graph.nodes.data(data='Labels'))[:,1]

#         #Return the updated graph and the labels
#         return Graph, labels

#     @classmethod
#     def gl_expansion(cls):
#         return labels

#     @classmethod
#     def iteritative_fixed_k(cls, locations, graph):
#         """Call a fixed k clustering method iteratively
#         using the Gap Statisic method to choose K."""


# class test(Cluster):
#     """Testing"""

#     def __init__(self):
#         # self
#         pass

#     @classmethod
#     def test(cls):
#         print("Success!")
