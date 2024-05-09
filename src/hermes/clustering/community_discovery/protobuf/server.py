import json
from concurrent import futures

import grpc
import networkx as nx
import rbpots_pb2
import rbpots_pb2_grpc
from rbpots import cluster


class ClusteringServicer(rbpots_pb2_grpc.ClusteringServicer):
    def SendAndModifyGraph(self, request, context):
        # Deserialize the graph JSON to a NetworkX graph
        graph_data = json.loads(request.data)
        G = nx.readwrite.json_graph.node_link_graph(graph_data)
        resolution = request.res
        labels, G = cluster(G, resolution)

        # Serialize the graph back to JSON to return
        modified_graph_json = json.dumps(nx.node_link_data(G))
        return rbpots_pb2.OutgoingGraphandLabels(
            labels=labels, data=modified_graph_json
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rbpots_pb2_grpc.add_ClusteringServicer_to_server(ClusteringServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
