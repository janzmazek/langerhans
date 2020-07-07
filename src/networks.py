import numpy as np
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
from scipy.optimize import bisect

class Networks(object):
    """docstring for Networks."""

# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self, cells, filtered_slow, filtered_fast, ND_avg=8):
        self.__ND_avg = ND_avg
        self.__cells = cells
        self.__filtered_slow = filtered_slow
        self.__filtered_fast = filtered_fast

        self.__R_slow = False
        self.__R_fast = False

        self.__A_slow = False
        self.__A_fast = False

        self.__G_slow = False
        self.__G_fast = False

# --------------------------------- GETTERS ---------------------------------- #

    def get_G_slow(self): return self.__G_slow
    def get_G_fast(self): return self.__G_fast
    def get_R_slow(self): return self.__R_slow
    def get_R_fast(self): return self.__R_fast
    def get_A_slow(self): return self.__A_slow
    def get_A_fast(self): return self.__A_fast

# ----------------------------- NETWORK METHODS ------------------------------ #
    def build_networks(self):
        # Compute correlation matrices
        self.__construct_correlation_matrix()
        # Calculate threshold and construct network
        slow_threshold, r = bisect(lambda x: self.__graph_from_threshold(self.__R_slow, x)["ND"]-self.__ND_avg, 0, 1, full_output=True)
        # Calculate threshold and construct network
        fast_threshold, r = bisect(lambda x: self.__graph_from_threshold(self.__R_fast, x)["ND"]-self.__ND_avg, 0, 1, full_output=True)

        # self.__G_slow = self.__graph_from_threshold(self.__R_slow, slow_threshold)["G"]
        self.__G_slow = self.__graph_from_threshold(self.__R_slow, slow_threshold)["G"]
        self.__G_fast = self.__graph_from_threshold(self.__R_fast, fast_threshold)["G"]

        self.__A_slow = nx.to_numpy_matrix(self.__G_slow)
        self.__A_fast = nx.to_numpy_matrix(self.__G_fast)

    def __construct_correlation_matrix(self):
        self.__R_slow, self.__R_fast = np.eye(self.__cells), np.eye(self.__cells)
        for i in range(self.__cells):
            for j in range(i):
                corr_slow = np.corrcoef(self.__filtered_slow[i], self.__filtered_slow[j])[0,1]
                self.__R_slow[i,j], self.__R_slow[j,i] = corr_slow, corr_slow
                corr_fast = np.corrcoef(self.__filtered_fast[i], self.__filtered_fast[j])[0,1]
                self.__R_fast[i,j], self.__R_fast[j,i] = corr_fast, corr_fast

    def __graph_from_threshold(self, R, R_threshold):
        G = nx.Graph()
        for i in range(self.__cells):
            G.add_node(i)
            for j in range(i):
                if R[i,j] >= R_threshold:
                    G.add_edge(i,j)
        ND = np.mean([G.degree(i) for i in G])
        return {"G": G, "ND": ND}

    def node_degree(self, cell):
        return (self.__G_slow.degree(cell),
                self.__G_fast.degree(cell))

    def clustering(self, cell):
        return (nx.clustering(self.__G_slow)[cell],
                nx.clustering(self.__G_fast)[cell])

    def nearest_neighbour_degree(self, cell):
        return (nx.average_neighbor_degree(self.__G_slow)[cell],
                nx.average_neighbor_degree(self.__G_fast)[cell])

    def modularity(self):
        partition_slow = community_louvain.best_partition(self.__G_slow)
        partition_fast = community_louvain.best_partition(self.__G_fast)
        Q_slow = community_louvain.modularity(partition_slow, self.__G_slow)
        Q_fast = community_louvain.modularity(partition_fast, self.__G_fast)

        return (Q_slow, Q_fast)

    def global_efficiency(self):
        GE_slow = nx.global_efficiency(self.__G_slow)
        GE_fast = nx.global_efficiency(self.__G_fast)
        return (GE_slow, GE_fast)

    def max_connected_component(self):
        MS_slow = len(max(nx.connected_components(self.__G_slow), key=len))/self.__cells
        MS_fast = len(max(nx.connected_components(self.__G_fast), key=len))/self.__cells
        return (MS_slow, MS_fast)

    def average_correlation(self):
        R_slow = np.matrix(self.__R_slow)
        R_fast = np.matrix(self.__R_fast)
        R_slow_upper = R_slow[np.triu_indices(R_slow.shape[0])]
        R_fast_upper = R_fast[np.triu_indices(R_fast.shape[0])]
        return (R_slow_upper.mean(), R_fast_upper.mean())

    def draw_networks(self, positions, ax1, ax2, colors):
        nx.draw(self.__G_slow, pos=positions, ax=ax1, with_labels=True, node_size=50, width=0.25, font_size=3, node_color=colors[0])
        nx.draw(self.__G_fast, pos=positions, ax=ax2, with_labels=True, node_size=50, width=0.25, font_size=3, node_color=colors[1])
