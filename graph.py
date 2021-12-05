import numpy as np
from easydict import EasyDict


class Graph:
    def __init__(self, start_point, end_point):
        self.graph = EasyDict()
        self.start_point = start_point
        self.end_point = end_point

    def add_weights(self, edges_list: list):
        for edge in edges_list:
            if edge[0] not in self.graph:
                # init dict obj for vertex
                self.graph[edge[0]] = EasyDict()
                self.graph[edge[0]].neighbours = list()
                if edge[0] != self.start_point:
                    # init distance to this vertex as inf if it is not start point
                    self.graph[edge[0]].distance = np.inf
                else:
                    # init distance to this vertex as 0 if it is start point and define shortest path
                    self.graph[edge[0]].distance = 0
                    self.graph[edge[0]].path = self.start_point

            # collect all neibs vertex to separate field
            self.graph[edge[0]].neighbours.append(edge[1])
            # set the weight of edge
            self.graph[edge[0]][edge[1]] = edge[2]

    def find_all_dist(self):
        for _ in range(len(self.graph.keys()) - 1):
            for current_src_vertex, value in self.graph.items():
                for neighbour in value['neighbours']:
                    # dist2neib = dist to current point and edge weight
                    dist2neib = self.graph[current_src_vertex].distance + self.graph[current_src_vertex][neighbour]
                    if self.graph[current_src_vertex].distance != np.inf and dist2neib < self.graph[neighbour].distance:
                        # upd the min dist and short path
                        self.graph[neighbour].distance = dist2neib
                        self.graph[neighbour].path = f'{self.graph[current_src_vertex].path}|{neighbour}'


