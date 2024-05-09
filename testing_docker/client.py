import grpc
import graph_pb2
import graph_pb2_grpc
import networkx as nx
import json

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = graph_pb2_grpc.GraphServiceStub(channel)
        # Create a graph
        # global G
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        G.add_edges_from([(1, 2), (2, 3)])

        # Serialize the graph to JSON
        graph_json = json.dumps(nx.node_link_data(G))

        # Create a Graph message
        graph_message = graph_pb2.Graph(data=graph_json)
        
        # Send the graph and receive a modified graph
        modified_graph = stub.SendAndModifyGraph(graph_message)
        global new
        new = nx.readwrite.json_graph.node_link_graph(json.loads(modified_graph.data))
        print("Client received modified graph: ", modified_graph.data)

if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([(1, 2), (2, 3)])
    run()

