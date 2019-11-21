import numpy as np
import matplotlib.pyplot as plt

from .networks import Networks

class Analysis(object):
    """docstring for Analysis."""

# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self, data):
        assert data.is_analyzed()

        settings = data.get_settings()

        start, stop = settings["analysis"]["interval"]
        sampling = settings["sampling"]
        start, stop = start*sampling, stop*sampling

        self.__slices = settings["network"]["slices"]

        self.__data_points = stop-start
        self.__number_of_cells = data.get_number_of_cells()
        self.__positions = data.get_positions()

        self.__filtered_slow = np.array_split(data.get_filtered_slow()[start:stop], slices)
        self.__filtered_fast = np.array_split(data.get_filtered_fast()[start:stop], slices)

        self.__binarized_slow = np.array_split(data.get_binarized_slow()[start:stop], slices)
        self.__binarized_fast = np.array_split(data.get_binarized_fast()[start:stop], slices)

        self.__networks = Networks(settings, self.__number_of_cells, self.__filtered_slow, self.__filtered_fast)
        self.__networks.build_networks()

        self.__network_parameters = self.__compute_network_parameters()
        self.__dynamics_parameters = self.__compute_dynamics_parameters()

# --------------------------------- GETTERS ---------------------------------- #

# ----------------------------- ANALYSIS METHODS ----------------------------- #

    def __compute_network_parameters(self):
        G_slow = self.__networks.get_G_slow()
        G_fast = self.__networks.get_G_fast()

        par = [[dict() for c in self.__number_of_cells] for s in self.slices]
        for slice in range(self.__slices):
            for cell in self.__number_of_cells:
                par[slice][cell]["ND"] = self.__networks.node_degree(slice, cell)
                par[slice][cell]["C"] = self.__networks.clustering(slice, cell)
                par[slice][cell]["NND"] = self.__networks.nearest_neighbour_degree(slice, cell)

        return par

    def __compute_dynamics_parameters(self):
        spikes = 0
        state = 0
        active_time = 0
        interspike = 0
        interspike_slow = []
        interspike_fast = []

        par = [[dict() for c in self.__number_of_cells] for s in self.slices]
        for slice in range(self.__slices):
            for cell in self.__number_of_cells:
                par[slice][cell]["active_time"] = self.__active_time(slice, cell)
                par[slice][cell]["spiking_density"] = self.__spiking_density(slice, cell)
                par[slice][cell]["interspike_variation"] = self.__interspike_variation(slice, cell)

        return par

    def __active_time(self, slice, cell):
        return np.sum(self.__binarized_fast[slice][cell])

    def __spiking_density(self, slice, cell):
        bin = self.__binarized_fast[slice][cell]
        # Count elements which change value by comparing each element with it's
        # neighbor
        return np.sum(bin[:-1] != bin[1:])/len(bin)

    def __interspike_variation(self, slice, cell):
        bin = self.__binarized_fast[slice][cell]
        interspikes = np.logical_not(bin)
        changes_in_value = np.where(interspikes[:-1] != interspikes[1:])[0]
        if bin[changes_in_value[0]] == 1:
            changes_in_value = changes_in_value[1:]
        if len(changes_in_value)%2 != 0:
            changes_in_value = changes_in_value[:-1]
        lengths = [changes_in_value[i+1]-changes_in_value[i] for i in range(0, len(changes_in_value)-1, 2)]



    def draw_networks(self, location):
        self.__networks.draw_networks(self.__positions, location)

    # def compare_slow_fast(self):
    #     properties = {i:{"spikes": 0, "lengths": 0} for i in range(13)}
    #
    #     for cell in range(self.__number_of_cells):
    #         start = 0
    #         end = 0
    #
    #         for i in range(self.__data_points-1):
    #             if self.__binarized_slow[i,cell] != self.__binarized_slow[i+1,cell]:
    #                 end = i+1
    #
    #                 angle_index = self.__binarized_slow[i,cell]
    #                 fast_interval = self.__binarized_fast[start:end,cell]
    #
    #                 bincnt = np.bincount(fast_interval)
    #                 number_of_spikes = 0 if len(bincnt)==1 else bincnt[1]
    #
    #                 properties[angle_index]["spikes"] += number_of_spikes
    #                 properties[angle_index]["lengths"] += end-start
    #
    #                 start = end
    #
    #     densities = {i: properties[i]["spikes"]/properties[i]["lengths"] for i in range(13)}
    #
    #
    #     x0 = (np.pi/3 - np.pi/6)/2
    #     d = np.pi/6
    #     phi = [x0+d*i for i in range(12)]
    #     bars = np.array([(-np.cos(i)+1)/2 for i in phi])
    #     data = np.array([densities[i] for i in range(1, len(densities))])
    #     norm_data = data/np.max(data)
    #
    #     plt.plot(phi, bars, phi, norm_data)
    #     plt.show()
    #
    #     colors = plt.cm.seismic(norm_data)
    #     ax = plt.subplot(111, projection="polar")
    #     ax.bar(phi, norm_data, width=2*np.pi/12, bottom=0.0, color=colors)
    #     plt.show()
    #
    #     return (phi, bars, norm_data)
    #
    # def compare_correlation_distance(self):
    #     distances = []
    #     correlations_slow = []
    #     correlations_fast = []
    #     for c1 in range(self.__number_of_cells):
    #         for c2 in range(c1):
    #             x1, y1 = self.__positions[c1,0], self.__positions[c1,1]
    #             x2, y2 = self.__positions[c2,0], self.__positions[c2,1]
    #             distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    #             correlation_slow = np.corrcoef(self.__filtered_slow[:,c1], self.__filtered_slow[:,c2])[0,1]
    #             correlation_fast = np.corrcoef(self.__filtered_fast[:,c1], self.__filtered_fast[:,c2])[0,1]
    #             distances.append(distance)
    #             correlations_slow.append(correlation_slow)
    #             correlations_fast.append(correlation_fast)
    #
    #     return (distances, correlations_slow, correlations_fast)
    #
    # def compute_topology_parameters(self):
    #     if self.__G_slow is False or self.__G_fast is False:
    #         raise ValueError("No built network.")
    #     parameters = [dict() for i in range(self.__number_of_cells)]
    #
    #     for cell in range(self.__number_of_cells):
    #         parameters[cell]["NDs"] = self.__G_slow.degree[cell]
    #         parameters[cell]["NDf"] = self.__G_fast.degree[cell]
    #         parameters[cell]["Cs"] = nx.clustering(self.__G_slow)[cell]
    #         parameters[cell]["Cf"] = nx.clustering(self.__G_fast)[cell]
    #         parameters[cell]["NNDs"] = nx.average_neighbor_degree(self.__G_slow)[cell]
    #         parameters[cell]["NNDf"] = nx.average_neighbor_degree(self.__G_fast)[cell]
    #
    #     return parameters
    #
    # def compute_dynamics_parameters(self):
    #     parameters = [dict() for i in range(self.__number_of_cells)]
    #
    #     for cell in range(self.__number_of_cells):
    #         spikes = 0
    #         state = 0
    #         active_time = 0
    #         interspike = 0
    #         interspike_slow = []
    #         interspike_fast = []
    #
    #         # Find average frequency of fast data
    #         start = int(Tzac3*self.__sampling)
    #         stop = int(Tkon3*self.__sampling)
    #         for i in range(start, stop):
    #             if self.__binarized_fast[i,cell] == 1 and state == 0:
    #                 spikes += 1
    #                 state = 1
    #                 interspike_fast.append(interspike/self.__sampling)
    #             elif self.__binarized_fast[i,cell] == 0 and state == 1:
    #                 state = 0
    #                 interspike = 1
    #             elif self.__binarized_fast[i,cell] == 0 and state == 0:
    #                 interspike += 1
    #
    #         parameters[cell]["Ff"] = spikes/self.__data_points*self.__sampling
    #
    #         # Find average frequency of slow data
    #         spikes = 0
    #         state = 0
    #         interspike = 0
    #         i_start = 0
    #         i_end = 0
    #         end_check = 0
    #         for i in range(self.__data_points):
    #             if self.__binarized_slow[i, cell] == 12 and state == 0:
    #                 if i_start == 0:
    #                     i_start = i
    #                 spikes += 1
    #                 if i_end==0 or end_check!=spikes:
    #                     i_end = i
    #                     end_check = spikes
    #                 state = 1
    #                 interspike_slow.append(interspike/self.__sampling)
    #             elif self.__binarized_slow[i,cell] != 12 and state == 1:
    #                 state = 0
    #                 interspike = 1
    #
    #             elif self.__binarized_slow[i,cell] != 12 and state == 0:
    #                 interspike += 1
    #         parameters[cell]["Fs"] = (spikes-1)/(i_end-i_start)*self.__sampling
    #
    #         # Find first peak of fast data
    #         for i in range(150*self.__sampling, self.__data_points):
    #             if self.__binarized_fast[i,cell] == 1:
    #                 parameters[cell]["Tfirst"] = i/self.__sampling
    #                 break
    #
    #         # Find last peak of fast data
    #         for i in range(self.__data_points-1, -1, -1):
    #             if self.__binarized_fast[i,cell] == 1:
    #                 parameters[cell]["Tlast"] = i/self.__sampling
    #                 break
    #
    #
    #         # Find active time of fast data
    #         active_time = np.bincount(self.__binarized_fast[:,cell])[1]
    #         parameters[cell]["Ton"] = active_time/self.__sampling
    #
    #         interspike_fast = interspike_fast[1:-1]
    #
    #         # Find interspike means
    #         parameters[cell]["ISIMs"] = np.mean(interspike_slow)
    #         parameters[cell]["ISIMf"] = np.mean(interspike_fast)
    #
    #         # Find interspike variations
    #         parameters[cell]["ISIVs"] = np.std(interspike_slow)/np.mean(interspike_slow)
    #         parameters[cell]["ISIVf"] = np.std(interspike_fast)/np.mean(interspike_fast)
    #
    #     return parameters
    #
    # def correlations(self, path):
    #     dynamics = self.dynamics_parameters()
    #     topology = self.topology_parameters()
    #     merged = [{**dynamics[cell], **topology[cell]} for cell in range(self.__number_of_cells)]
    #
    #     with open(path, "w") as f:
    #         f.write("p1/p2\t")
    #         for p2 in merged[0]:
    #             f.write("{}\t".format(p2))
    #         f.write("\n")
    #         for p1 in merged[0]:
    #             f.write("{}\t".format(p1))
    #             for p2 in merged[0]:
    #                 x = [merged[cell][p1] for cell in range(self.__number_of_cells)]
    #                 y = [merged[cell][p2] for cell in range(self.__number_of_cells)]
    #                 slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #                 f.write("{}\t".format(round(r_value, 2)))
    #             f.write("\n")
