from concurrent import futures
import grpc
import graph_pb2
import graph_pb2_grpc
import networkx as nx
import json

class GraphServiceServicer(graph_pb2_grpc.GraphServiceServicer):
    def SendAndModifyGraph(self, request, context):
        # Deserialize the graph JSON to a NetworkX graph
        graph_data = json.loads(request.data)
        G = nx.readwrite.json_graph.node_link_graph(graph_data)

        # Add a random node
        new_node =2345
        G.add_node(new_node)

        # Serialize the graph back to JSON to return
        modified_graph_json = json.dumps(nx.node_link_data(G))
        return graph_pb2.Graph(data=modified_graph_json)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    graph_pb2_grpc.add_GraphServiceServicer_to_server(GraphServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

