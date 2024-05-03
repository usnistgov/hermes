"""RBPots algorithm."""

import networkx as nx  # type: ignore
import numpy as np
from cdlib.utils import suppress_stdout

with suppress_stdout():
    from cdlib import algorithms  # type: ignore


def cluster(G, res):
    """Cluster the graph using the RB Pots algorithm.
    G is a networkx graph object
    res is a tunable parameter float for the resolution"""
    # Cluster with RB Pots Algorithm
    clusters = algorithms.rb_pots(G, weights="Weight", resolution_parameter=res)

    # Label the graph with the clusters
    for i, k in enumerate(clusters.communities):
        for q in k:
            nx.set_node_attributes(G, {q: i}, name="Labels")
    # Extract the labels
    labels = np.asarray(G.nodes.data(data="Labels"))[:, 1]

    # Return the labels and the modified graph (which now includes the labels)
    return labels, G
