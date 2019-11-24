import numpy as np
import matplotlib.pyplot as plt

from .networks import Networks

class Analysis(object):
    """docstring for Analysis."""

# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self, data):
        assert data.is_analyzed()

        # Get settings from data object
        settings = data.get_settings()
        start, stop = settings["analysis"]["interval"]
        self.__sampling = settings["sampling"]
        start, stop = start*self.__sampling, stop*self.__sampling

        self.__slices = settings["network"]["slices"]

        self.__points = stop-start
        self.__cells = data.get_cells()
        self.__positions = data.get_positions()

        # Split filtered data into slices
        self.__filtered_slow = np.array_split(data.get_filtered_slow()[start:stop], self.__slices)
        self.__filtered_fast = np.array_split(data.get_filtered_fast()[start:stop], self.__slices)

        # Split binarized data into slices
        self.__binarized_slow = np.array_split(data.get_binarized_slow()[start:stop], self.__slices)
        self.__binarized_fast = np.array_split(data.get_binarized_fast()[start:stop], self.__slices)

        # Construct networks and build networks from sliced data
        self.__networks = Networks(settings, self.__cells, self.__filtered_slow, self.__filtered_fast)
        self.__networks.build_networks()

        # Compute network and dynamics parameters
        self.__network_parameters = self.__compute_network_parameters()
        self.__dynamics_parameters = self.__compute_dynamics_parameters()

# --------------------------------- GETTERS ---------------------------------- #

# ----------------------------- ANALYSIS METHODS ----------------------------- #

    def print_parameters(self):
        print("Network parameters:")
        print(self.__network_parameters)
        print("Dynamics parameters:")
        print(self.__dynamics_parameters)

    def __compute_network_parameters(self):
        G_slow = self.__networks.get_G_slow()
        G_fast = self.__networks.get_G_fast()

        par = [[dict() for c in range(self.__cells)] for s in range(self.__slices)]
        for slice in range(self.__slices):
            for cell in range(self.__cells):
                par[slice][cell]["ND"] = self.__networks.node_degree(slice, cell)
                par[slice][cell]["C"] = self.__networks.clustering(slice, cell)
                par[slice][cell]["NND"] = self.__networks.nearest_neighbour_degree(slice, cell)

        return par

    def __compute_dynamics_parameters(self):
        par = [[dict() for c in range(self.__cells)] for s in range(self.__slices)]
        for slice in range(self.__slices):
            for cell in range(self.__cells):
                par[slice][cell]["active_time"] = self.__active_time(slice, cell)
                par[slice][cell]["frequency"] = self.__frequency(slice, cell)
                par[slice][cell]["interspike_interval"] = self.__interspike_interval(slice, cell)

        return par

    def __active_time(self, slice, cell):
        bin = self.__binarized_fast[slice][:,cell]
        return np.sum(bin)/len(bin)

    def __frequency(self, slice, cell):
        bin_slow = self.__binarized_slow[slice][:,cell]
        bin_fast = self.__binarized_fast[slice][:,cell]

        peaks = bin_slow == 12
        slow = np.sum(peaks[:-1] != peaks[1:])/2/len(peaks)*self.__sampling
        fast = np.sum(bin_fast[:-1] != bin_fast[1:])/2/len(bin_fast)*self.__sampling
        return (slow, fast)

    def __spike_borders(self, slice, cell):
        bin = self.__binarized_fast[slice][:,cell]
        spike_borders = np.where(bin[:-1] != bin[1:])[0]
        if len(spike_borders) == 0: # no spikes in this slice
            return None
        if bin[spike_borders[0]] == 1: # slice starts with spike
            spike_borders = spike_borders[1:] # remove initial spike
        if len(spike_borders)%2 != 0: # slice end with spike
            spike_borders = spike_borders[:-1] # remove final spike
        if len(spike_borders) == 0: # only initial and final spikes were in this slice
            return None
        else:
            return np.append(0, spike_borders, len(bin))

    def __interspike_interval(self, slice, cell):
        spike_borders = self.__spike_borders(slice, cell)
        if spike_borders is None:
            return None
        interspike_lengths = spike_borders[1::2]-spike_borders[:-1:2]
        mean_interspike_interval = np.mean(interspike_lengths)
        interspike_variation = np.std(interspike_lengths)/mean_interspike_interval

        return (mean_interspike_interval, interspike_variation)

    def draw_networks(self, location):
        self.__networks.draw_networks(self.__positions, location)

    def compare_slow_fast(self, draw=True):
        phases = np.arange((np.pi/3 - np.pi/6)/2, 2*np.pi, np.pi/6)
        spikes = np.zeros(12)
        for phase in range(1,13):
            for slice in range(self.__slices):
                for cell in range(self.__cells):
                    isolated_phase = self.__binarized_slow[slice][:,cell] == phase
                    spike_borders = self.__spike_borders(slice, cell)
                    if spike_borders is None:
                        continue
                    spike_indices = spike_borders[1::2]
                    unitized_spikes = np.zeros(len(isolated_phase))
                    unitized_spikes[spike_indices] = 1
                    isolated_spikes = np.logical_and(isolated_phase, unitized_spikes)

                    spikes[phase-1] += np.sum(isolated_spikes)

        if draw:
            norm_spikes = spikes/np.max(spikes)
            colors = plt.cm.seismic(norm_spikes)
            ax = plt.subplot(111, projection="polar")
            ax.bar(phases, norm_spikes, width=2*np.pi/12, bottom=0.0, color=colors)
            plt.show()

        return (phases, spikes)



    # def compare_slow_fast(self):
    #     properties = {i:{"spikes": 0, "lengths": 0} for i in range(13)}
    #
    #     for cell in range(self.__cells):
    #         start = 0
    #         end = 0
    #
    #         for i in range(self.__points-1):
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
    #     for c1 in range(self.__cells):
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
