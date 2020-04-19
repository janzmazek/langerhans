import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import bisect

class Networks(object):
    """docstring for Networks."""

# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self, cells, filtered_slow, filtered_fast, ND_avg=7):
        self.__ND_avg = ND_avg
        self.__cells = cells
        self.__filtered_slow = filtered_slow
        self.__filtered_fast = filtered_fast

        self.__R_slow = False
        self.__R_fast = False

        self.__G_slow = False
        self.__G_fast = False

# --------------------------------- GETTERS ---------------------------------- #

    def get_G_slow(self): return self.__G_slow
    def get_G_fast(self): return self.__G_fast

# ----------------------------- NETWORK METHODS ------------------------------ #

    def build_networks(self):
        # Calculate threshold and construct network
        slow_threshold = bisect(lambda x: self.__f(self.__filtered_slow, x), 0, 1, xtol=0.01)
        self.__G_slow, self.__R_slow = self.__construct_network(self.__filtered_slow, slow_threshold)

        # Calculate threshold and construct network
        fast_threshold = bisect(lambda x: self.__f(self.__filtered_fast, x), 0, 1, xtol=0.01)
        self.__G_fast, self.__R_fast = self.__construct_network(self.__filtered_fast, fast_threshold)

    # Threshold is found by calculating root of this function
    def __f(self, filtered, threshold):
        (G, R) = self.__construct_network(filtered, threshold)
        average_degree = np.mean([G.degree[i] for i in G])
        return average_degree-self.__ND_avg

    def __construct_network(self, filtered, threshold):
        correlation_matrix = np.eye(self.__cells)

        # Construct correlation matrix from filtered signal
        for i in range(self.__cells):
            for j in range(i):
                correlation = np.corrcoef(filtered[i], filtered[j])[0,1]
                correlation_matrix[i,j] = correlation
                correlation_matrix[j,i] = correlation

        # Construct adjacency matrix from correlation matrix and threshold
        A = np.zeros((self.__cells, self.__cells))
        for i in range(self.__cells):
            for j in range(i):
                if correlation_matrix[i,j]>threshold:
                    A[i,j] = 1
                    A[j,i] = 1

        return (nx.from_numpy_matrix(A), correlation_matrix)

    def adjacency_matrix(self):
        return (nx.to_numpy_matrix(self.__G_slow), nx.to_numpy_matrix(self.__G_fast))

    def node_degree(self, cell):
        return (self.__G_slow.degree(cell),
                self.__G_fast.degree(cell))

    def clustering(self, cell):
        return (nx.clustering(self.__G_slow)[cell],
                nx.clustering(self.__G_fast)[cell])

    def nearest_neighbour_degree(self, cell):
        return (nx.average_neighbor_degree(self.__G_slow)[cell],
                nx.average_neighbor_degree(self.__G_fast)[cell])

    def average_correlation(self):
        R_slow = np.matrix(self.__R_slow)
        R_fast = np.matrix(self.__R_fast)
        np.fill_diagonal(R_slow, 0)
        np.fill_diagonal(R_fast, 0)
        return (R_slow.mean(), R_fast.mean())

    def draw_networks(self, positions):
        fig, ax = plt.subplots(nrows=2, ncols=1)
        nx.draw(self.__G_slow[s], pos=positions, ax=ax[0], with_labels=True, node_size=50, width=0.25, font_size=3, node_color="blue")
        nx.draw(self.__G_fast[s], pos=positions, ax=ax[1], with_labels=True, node_size=50, width=0.25, font_size=3, node_color="red")
        return fig
