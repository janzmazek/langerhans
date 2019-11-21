import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import bisect

class Networks(object):
    """docstring for Networks."""

# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self, settings, number_of_cells, filtered_slow, filtered_fast):
        self.__settings = settings
        self.__number_of_cells = number_of_cells
        self.__filtered_slow = filtered_slow
        self.__filtered_fast = filtered_fast

        self.__G_slow = False
        self.__G_fast = False

# --------------------------------- GETTERS ---------------------------------- #

    def get_G_slow(self): return self.__G_slow
    def get_G_fast(self): return self.__G_fast

# ----------------------------- NETWORK METHODS ------------------------------ #

    def build_networks(self):
        self.__G_slow = []
        # Calculate threshold and construct network for each slice
        for interval in self.__filtered_slow:
            slow_threshold = bisect(lambda x: self.__f(interval, x), 0, 1, xtol=0.01)
            G_slow = self.__construct_network(interval, slow_threshold)
            self.__G_slow.append(G_slow)

        self.__G_fast = []
        # Calculate threshold and construct network for each slice
        for interval in self.__filtered_fast:
            fast_threshold = bisect(lambda x: self.__f(interval, x), 0, 1, xtol=0.01)
            G_fast = self.__construct_network(interval, fast_threshold)
            self.__G_fast.append(G_fast)

    # Threshold is found by calculating root of this function
    def __f(self, filtered, threshold):
        G = self.__construct_network(filtered, threshold)
        average_degree = np.mean([G.degree[i] for i in G])
        print(average_degree)
        return average_degree-self.__settings["network"]["average_degree"]

    def __construct_network(self, filtered, threshold):
        correlation_matrix = np.eye(self.__number_of_cells)

        # Construct correlation matrix from filtered signal
        for i in range(self.__number_of_cells):
            for j in range(i):
                correlation = np.corrcoef(filtered[:,i], filtered[:,j])[0,1]
                correlation_matrix[i,j] = correlation
                correlation_matrix[j,i] = correlation

        # Construct adjacency matrix from correlation matrix and threshold
        A = np.zeros((self.__number_of_cells, self.__number_of_cells))
        for i in range(self.__number_of_cells):
            for j in range(i):
                if correlation_matrix[i,j]>threshold:
                    A[i,j] = 1
                    A[j,i] = 1

        return nx.from_numpy_matrix(A)

    def node_degree(self, slice, cell):
        return self.__G_slow[slice].degree(cell),
               self.__G_fast[slice].degree(cell)

    def clustering(self, slice, cell):
        return nx.clustering(self.__G_slow[slice])[cell],
               nx.clustering(self.__G_fast[slice])[cell]

    def nearest_neighbour_degree(self, slice, cell):
        return nx.average_neighbor_degree(self.__G_slow[slice])[cell],
               nx.average_neighbor_degree(self.__G_fast[slice])[cell]

    def draw_networks(self, positions, directory):
        slices = len(self.__G_slow)
        for s in range(slices):
            fig, ax = plt.subplots(nrows=2, ncols=1)
            nx.draw(self.__G_slow[s], pos=positions, ax=ax[0], with_labels=True, node_size=50, width=0.25, font_size=3, node_color="blue")
            nx.draw(self.__G_fast[s], pos=positions, ax=ax[1], with_labels=True, node_size=50, width=0.25, font_size=3, node_color="red")
            plt.savefig("{0}/{1}.pdf".format(directory, s))
            plt.close()
