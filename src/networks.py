import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Networks(object):
    """docstring for Networks."""

    def __init__(self, data, positions):
        filtered_slow = data.get_filtered_slow()
        filtered_fast = data.get_filtered_fast()
        distributions = data.get_distributions()

        good_cells = [not cell["exclude"] for cell in distributions]
        self.__settings = data.get_settings()


        self.__positions = np.loadtxt(positions)[good_cells]
        self.__filtered_slow = filtered_slow[:,good_cells]
        self.__filtered_fast = filtered_fast[:,good_cells]

        self.__number_of_cells = np.sum(good_cells)

        self.__G_slow = False
        self.__G_fast = False

    def build_network(self):
        correlation_matrix_slow = np.eye(self.__number_of_cells)
        correlation_matrix_fast = np.eye(self.__number_of_cells)

        for i in range(self.__number_of_cells):
            for j in range(i):
                correlation = np.corrcoef(self.__filtered_slow[:,i], self.__filtered_slow[:,j])[0,1]
                correlation_matrix_slow[i,j] = correlation
                correlation_matrix_slow[j,i] = correlation
                correlation = np.corrcoef(self.__filtered_fast[:,i], self.__filtered_fast[:,j])[0,1]
                correlation_matrix_fast[i,j] = correlation
                correlation_matrix_fast[j,i] = correlation

        A_slow = np.zeros((self.__number_of_cells, self.__number_of_cells))
        A_fast = np.zeros((self.__number_of_cells, self.__number_of_cells))
        for i in range(self.__number_of_cells):
            for j in range(i):
                if correlation_matrix_slow[i,j]>self.__settings["network"]["slow_threshold"]:
                    A_slow[i,j] = 1
                    A_slow[j,i] = 1
                if correlation_matrix_fast[i,j]>self.__settings["network"]["fast_threshold"]:
                    A_fast[i,j] = 1
                    A_fast[j,i] = 1

        self.__G_slow = nx.from_numpy_matrix(A_slow)
        self.__G_fast = nx.from_numpy_matrix(A_fast)

    def get_topology_parameters(self):
        if self.__G_slow is False or self.__G_fast is False:
            raise ValueError("No built network.")
        parameters = [dict() for i in range(self.__number_of_cells)]

        for cell in range(self.__number_of_cells):
            parameters[cell]["NDs"] = self.__G_slow.degree[cell]
            parameters[cell]["NDf"] = self.__G_fast.degree[cell]
            parameters[cell]["Cs"] = nx.clustering(self.__G_slow)[cell]
            parameters[cell]["Cf"] = nx.clustering(self.__G_fast)[cell]
            parameters[cell]["NNDs"] = nx.average_neighbor_degree(self.__G_slow)[cell]
            parameters[cell]["NNDf"] = nx.average_neighbor_degree(self.__G_fast)[cell]

        return parameters

    def draw_network(self):
        pass

        # fig, ax = plt.subplots(nrows=2, ncols=1)
        # nx.draw(self.__G_slow, pos=self.__positions, ax=ax[0], with_labels=True, node_color="pink")
        # nx.draw(self.__G_fast, pos=self.__positions, ax=ax[1], with_labels=True, node_color="purple")
        # plt.show()
        #
        # average_slow_degree = np.mean([self.__G_slow.degree[i] for i in self.__G_slow])
        # average_fast_degree = np.mean([self.__G_fast.degree[i] for i in self.__G_fast])
        #
        # # return (average_slow_degree, average_fast_degree)
        # return (average_slow_degree, average_fast_degree)
