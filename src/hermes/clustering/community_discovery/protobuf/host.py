import json
import random

import grpc
import networkx as nx
import numpy as np
import rbpots_pb2
import rbpots_pb2_grpc

G = nx.fast_gnp_random_graph(10, 0.5, 42, False)

for u, v in G.edges():
    G.edges[u, v]["Weight"] = random.randint(0, 10)

resolution = 4.4

channel = grpc.insecure_channel("localhost:50051")
stub = rbpots_pb2_grpc.ClusteringStub(channel)
graph_json = json.dumps(nx.node_link_data(G))
graph_to_send = rbpots_pb2.IncomingGraphandResolution(data=graph_json, res=resolution)

response = stub.SendAndModifyGraph(graph_to_send)

G = nx.readwrite.json_graph.node_link_graph(json.loads(response.data))
labels = np.array(response.labels)
# return labels, G
