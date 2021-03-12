import networkx as nx
import numpy as np
from community import community_louvain
from scipy.optimize import bisect


class Network(nx.Graph):
    def __init__(self, filtered, threshold):
        super().__init__()
        self.__cells = len(filtered)
        self.__R = np.corrcoef(filtered)
        if threshold >= 1:
            threshold = bisect(lambda x: self.__build(x)-threshold, 0, 1)
        else:
            self.__build(threshold)

    def get_R(self): return self.__R

    def __build(self, R_threshold):
        self.clear()
        for i in range(self.__cells):
            self.add_node(i)
            for j in range(i):
                if self.__R[i, j] >= R_threshold:
                    self.add_edge(i, j)
        avg_ND = np.mean([self.degree(i) for i in self])
        return avg_ND

    def clustering(self, cell):
        return nx.clustering(self)[cell]

    def average_neighbour_degree(self, cell):
        return nx.average_neighbor_degree(self)[cell]

    def modularity(self):
        partition = community_louvain.best_partition(self)
        return community_louvain.modularity(partition, self)

    def global_efficiency(self):
        return nx.global_efficiency(self)

    def max_connected_component(self):
        return len(max(nx.connected_components(self), key=len))/self.__cells

    def average_correlation(self):
        R = np.matrix(self.__R)
        R_upper = R[np.triu_indices(R.shape[0])]
        return R_upper.mean()

    def average_connection_distances(self, distances_matrix):
        A_dst = distances_matrix
        A = self.to_numpy_matrix()
        A_dst = np.multiply(A_dst, A)
        distances = []
        for c1 in range(self.__cells):
            for c2 in range(c1):
                d = A_dst[c1, c2]
                if d > 0:
                    distances.append(d)
        return distances

    def draw_network(self, positions, ax, color):
        nx.draw(self, pos=positions, ax=ax, with_labels=False, node_size=50,
                width=0.25, font_size=3, node_color=color
                )
