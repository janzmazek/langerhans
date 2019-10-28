import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import bisect

class Networks(object):
    """docstring for Networks."""

# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self, data):
        self.__positions = data.get_positions()
        self.__settings = data.get_settings()
        self.__data_points = data.get_data_points()
        self.__number_of_cells = data.get_number_of_cells()
        self.__filtered_slow = data.get_filtered_slow()
        self.__filtered_fast = data.get_filtered_fast()

        self.__G_slow = False
        self.__G_fast = False

# --------------------------------- GETTERS ---------------------------------- #
    def get_positions(self): return self.__positions
    def get_G_slow(self): return self.__G_slow
    def get_G_fast(self): return self.__G_fast

# ----------------------------- NETWORK METHODS ------------------------------ #

    def build_network(self, splitting_number=2):
        self.__G_slow = []
        for interval in np.array_split(self.__filtered_slow, splitting_number):
            slow_threshold = bisect(lambda x: self.__f(interval, x), 0, 1, xtol=0.01)
            G_slow = self.__construct_network(interval, slow_threshold)
            self.__G_slow.append(G_slow)

        self.__G_fast = []
        for interval in np.array_split(self.__filtered_fast, splitting_number):
            fast_threshold = bisect(lambda x: self.__f(interval, x), 0, 1, xtol=0.01)
            G_fast = self.__construct_network(interval, fast_threshold)
            self.__G_fast.append(G_fast)

    def __f(self, filtered, threshold):
        G = self.__construct_network(filtered, threshold)
        average_degree = np.mean([G.degree[i] for i in G])
        print(average_degree)
        return average_degree-self.__settings["network"]["average_degree"]

    def __construct_network(self, filtered, threshold):
        correlation_matrix = np.eye(self.__number_of_cells)

        for i in range(self.__number_of_cells):
            for j in range(i):
                correlation = np.corrcoef(filtered[:,i], filtered[:,j])[0,1]
                correlation_matrix[i,j] = correlation
                correlation_matrix[j,i] = correlation

        A = np.zeros((self.__number_of_cells, self.__number_of_cells))
        for i in range(self.__number_of_cells):
            for j in range(i):
                if correlation_matrix[i,j]>threshold:
                    A[i,j] = 1
                    A[j,i] = 1

        return nx.from_numpy_matrix(A)

    def draw_network(self,):
        slices = len(self.__G_slow)
        fig, ax = plt.subplots(nrows=slices, ncols=2)
        for s in range(slices):
            nx.draw(self.__G_slow[s], pos=self.__positions, ax=ax[s,0], with_labels=True, node_color="blue")
            nx.draw(self.__G_fast[s], pos=self.__positions, ax=ax[s,1], with_labels=True, node_color="red")
        plt.show()
