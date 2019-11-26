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
        filtered_slow = data.get_filtered_slow().T[start:stop]
        filtered_slow = np.array(np.array_split(filtered_slow, self.__slices))
        self.__filtered_slow = np.transpose(filtered_slow, (0,2,1))
        filtered_fast = data.get_filtered_fast().T[start:stop]
        filtered_fast = np.array(np.array_split(filtered_fast, self.__slices))
        self.__filtered_fast = np.transpose(filtered_fast, (0,2,1))

        # Split binarized data into slices
        binarized_slow = data.get_binarized_slow().T[start:stop]
        binarized_slow = np.array(np.array_split(binarized_slow, self.__slices))
        self.__binarized_slow = np.transpose(binarized_slow, (0,2,1))

        binarized_fast = data.get_binarized_fast().T[start:stop]
        binarized_fast = np.array(np.array_split(binarized_fast, self.__slices))
        self.__binarized_fast = np.transpose(binarized_fast, (0,2,1))

        # Construct networks and build networks from sliced data
        self.__networks = Networks(settings, self.__cells, self.__filtered_slow, self.__filtered_fast)
        self.__networks.build_networks()

        # Compute network and dynamics parameters
        self.__network_parameters = self.__compute_network_parameters()
        self.__dynamics_parameters = self.__compute_dynamics_parameters()

# --------------------------------- GETTERS ---------------------------------- #


# ---------------------------- ANALYSIS FUNCTIONS ---------------------------- #
    def __search_sequence(self, arr, seq):
        """ Find sequence in an array using NumPy only.

        Parameters
        ----------
        arr    : input 1D array
        seq    : input 1D array

        Output
        ------
        Output : 1D Array of indices in the input array that satisfy the
        matching of input sequence in the input array.
        In case of no match, an empty list is returned.
        """
        # Store sizes of input array and sequence
        seq = np.array(seq)
        Na, Nseq = arr.size, seq.size

        # Range of sequence
        r_seq = np.arange(Nseq)

        # Create a 2D array of sliding indices across the entire length of input array.
        # Match up with the input sequence & get the matching starting indices.
        M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

        # Get the range of those indices as final output
        if M.any() > 0:
            return np.where(M == True)[0]
        else:
            return np.array([], dtype="int") # No match found

# ----------------------------- ANALYSIS METHODS ----------------------------- #
    def draw_networks(self, location):
        self.__networks.draw_networks(self.__positions, location)

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
                par[slice][cell]["interspike_variation"] = self.__interspike_variation(slice, cell)

        return par

    def __active_time(self, slice, cell):
        bin = self.__binarized_fast[slice][cell]
        return np.sum(bin)/bin.size

    def __frequency(self, slice, cell):
        bin_slow = self.__binarized_slow[slice][cell]
        bin_fast = self.__binarized_fast[slice][cell]

        frequency_slow = self.__search_sequence(bin_slow, [11,12]).size
        frequency_slow = frequency_slow/len(bin_slow)*self.__sampling
        frequency_fast = self.__search_sequence(bin_fast, [0,1]).size
        frequency_fast = frequency_fast/len(bin_fast)*self.__sampling

        return (frequency_slow, frequency_fast)

    def __interspike_variation(self, slice, cell):
        bin_fast = self.__binarized_fast[slice][cell]

        interspike_start = self.__search_sequence(bin_fast, [1,0])
        interspike_end = self.__search_sequence(bin_fast, [0,1])

        if interspike_start.size == 0 or interspike_end.size == 0:
            return None
        # First interspike_start must be before first interspike_end
        if interspike_end[-1] < interspike_start[-1]:
            interspike_start = interspike_start[:-1]
        if interspike_start.size == 0:
            return None
        # Last interspike_start must be before last interspike_end
        if interspike_end[0] < interspike_start[0]:
            interspike_end = interspike_end[1:]

        assert interspike_start.size == interspike_end.size

        interspike_lengths = [interspike_end[i+1]-interspike_start[i] for i in range(len(interspike_start)-1)]
        mean_interspike_interval = np.mean(interspike_lengths)
        interspike_variation = np.std(interspike_lengths)/mean_interspike_interval

        return (mean_interspike_interval, interspike_variation)

    def compare_slow_fast(self, plot=True):
        phases = np.arange((np.pi/3 - np.pi/6)/2, 2*np.pi, np.pi/6)
        spikes = np.zeros(12)
        for phase in range(1,13):
            for slice in range(self.__slices):
                for cell in range(self.__cells):
                    slow_isolated = self.__binarized_slow[slice][cell] == phase

                    bin_fast = self.__binarized_fast[slice][cell]
                    spike_indices = self.__search_sequence(bin_fast, [0,1]) + 1
                    fast_unitized = np.zeros(len(bin_fast))
                    fast_unitized[spike_indices] = 1

                    fast_isolated_unitized = np.logical_and(slow_isolated, fast_unitized)

                    spikes[phase-1] += np.sum(fast_isolated_unitized)

        if plot:
            norm_spikes = spikes/np.max(spikes)
            colors = plt.cm.seismic(norm_spikes)
            ax = plt.subplot(111, projection="polar")
            ax.bar(phases, norm_spikes, width=2*np.pi/12, bottom=0.0, color=colors)
            plt.show()

        return (phases, spikes)


    def compare_correlation_distance(self, plot=True):
        distances = []
        correlations_slow = [[] for s in range(self.__slices)]
        correlations_fast = [[] for s in range(self.__slices)]
        for cell1 in range(self.__cells):
            for cell2 in range(cell1):
                x1, y1 = self.__positions[cell1,0], self.__positions[cell1,1]
                x2, y2 = self.__positions[cell2,0], self.__positions[cell2,1]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                distances.append(distance)

                for slice in range(self.__slices):
                    corr_slow = np.corrcoef(self.__filtered_slow[slice][cell1],
                                            self.__filtered_slow[slice][cell2])[0,1]
                    corr_fast = np.corrcoef(self.__filtered_fast[slice][cell1],
                                            self.__filtered_fast[slice][cell2])[0,1]
                    correlations_slow[slice].append(corr_slow)
                    correlations_fast[slice].append(corr_fast)

        if plot:
            fig, ax = plt.subplots(self.__slices)
            for slice in range(self.__slices):
                ax[slice].scatter(distances, correlations_slow[slice])
                ax[slice].scatter(distances, correlations_fast[slice])
            plt.show()

        return (distances, correlations_slow, correlations_fast)
