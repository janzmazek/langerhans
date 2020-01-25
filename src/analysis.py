import numpy as np
import matplotlib.pyplot as plt

from .networks import Networks

class Analysis(object):
    """docstring for Analysis."""

# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self):
        self.__sampling = None
        self.__slices = None
        self.__points = None
        self.__cells = None
        self.__positions = None
        self.__filtered_slow = None
        self.__filtered_fast = None
        self.__binarized_slow = None
        self.__binarized_fast = None

        self.__networks = False

    def import_data(self, data, positions):
        assert data.is_analyzed()

        # Get settings from data object
        settings = data.get_settings()
        start, stop = settings["analysis"]["interval"]
        self.__sampling = settings["sampling"]
        start, stop = start*self.__sampling, stop*self.__sampling
        good_cells = data.get_good_cells()

        self.__slices = settings["network"]["slices"]

        self.__points = stop-start
        self.__cells = np.sum(good_cells)
        self.__positions = positions

        # Split filtered data into slices
        filtered_slow = data.get_filtered_slow().T[start:stop, good_cells]
        filtered_slow = np.array_split(filtered_slow, self.__slices)
        self.__filtered_slow = [filtered_slow[i].T for i in range(self.__slices)]

        filtered_fast = data.get_filtered_fast().T[start:stop, good_cells]
        filtered_fast = np.array_split(filtered_fast, self.__slices)
        self.__filtered_fast = [filtered_fast[i].T for i in range(self.__slices)]

        # Split binarized data into slices
        binarized_slow = data.get_binarized_slow().T[start:stop, good_cells]
        binarized_slow = np.array_split(binarized_slow, self.__slices)
        self.__binarized_slow = [binarized_slow[i].T for i in range(self.__slices)]

        binarized_fast = data.get_binarized_fast().T[start:stop, good_cells]
        binarized_fast = np.array_split(binarized_fast, self.__slices)
        self.__binarized_fast = [binarized_fast[i].T for i in range(self.__slices)]

    def build_networks(self):
        # Construct networks and build networks from sliced data
        self.__networks = Networks(self.__cells, self.__filtered_slow, self.__filtered_fast)
        self.__networks.build_networks()


# ---------------------------- ANALYSIS FUNCTIONS ---------------------------- #
    def __search_sequence(self, arr, seq):
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

    def compute_parameters(self):
        if self.__networks is not False:
            G_slow = self.__networks.get_G_slow()
            G_fast = self.__networks.get_G_fast()

        par = [[dict() for c in range(self.__cells)] for s in range(self.__slices)]
        for slice in range(self.__slices):
            for cell in range(self.__cells):
                par[slice][cell]["AT"] = self.active_time(slice, cell)
                par[slice][cell]["Fs"] = self.frequency(slice, cell)[0]
                par[slice][cell]["Ff"] = self.frequency(slice, cell)[1]
                par[slice][cell]["ISI"] = self.interspike_variation(slice, cell)[0]
                par[slice][cell]["ISIV"] = self.interspike_variation(slice, cell)[1]
                if self.__networks is not False:
                    par[slice][cell]["NDs"] = self.__networks.node_degree(slice, cell)[0]
                    par[slice][cell]["NDf"] = self.__networks.node_degree(slice, cell)[1]
                    par[slice][cell]["Cs"] = self.__networks.clustering(slice, cell)[0]
                    par[slice][cell]["Cf"] = self.__networks.clustering(slice, cell)[1]
                    par[slice][cell]["NNDs"] = self.__networks.nearest_neighbour_degree(slice, cell)[0]
                    par[slice][cell]["NNDf"] = self.__networks.nearest_neighbour_degree(slice, cell)[1]

        return par

    def active_time(self, slice, cell):
        bin = self.__binarized_fast[slice][cell]
        return np.sum(bin)/bin.size

    def frequency(self, slice, cell):
        bin_slow = self.__binarized_slow[slice][cell]
        bin_fast = self.__binarized_fast[slice][cell]

        slow_peaks = self.__search_sequence(bin_slow, [11,12])
        if slow_peaks.size < 2:
            frequency_slow = None
        else:
            slow_interval = slow_peaks[-1]-slow_peaks[0]
            frequency_slow = (slow_peaks.size-1)/slow_interval*self.__sampling

        fast_peaks = self.__search_sequence(bin_fast, [0,1])
        if fast_peaks.size < 2:
            frequency_fast = None
        else:
            fast_interval = fast_peaks[-1]-fast_peaks[0]
            frequency_fast = (fast_peaks.size-1)/fast_interval*self.__sampling

        return (frequency_slow, frequency_fast)

    def interspike_variation(self, slice, cell):
        bin_fast = self.__binarized_fast[slice][cell]

        interspike_start = self.__search_sequence(bin_fast, [1,0])
        interspike_end = self.__search_sequence(bin_fast, [0,1])

        if interspike_start.size == 0 or interspike_end.size == 0:
            return (None, None)
        # First interspike_start must be before first interspike_end
        if interspike_end[-1] < interspike_start[-1]:
            interspike_start = interspike_start[:-1]
        if interspike_start.size == 0:
            return (None, None)
        # Last interspike_start must be before last interspike_end
        if interspike_end[0] < interspike_start[0]:
            interspike_end = interspike_end[1:]

        assert interspike_start.size == interspike_end.size

        interspike_lengths = [interspike_end[i]-interspike_start[i] for i in range(interspike_start.size)]
        mean_interspike_interval = np.mean(interspike_lengths)
        interspike_variation = np.std(interspike_lengths)/mean_interspike_interval

        return (mean_interspike_interval, interspike_variation)

    def node_degree(self, slice, cell):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.node_degree(slice, cell)

    def clustering(self, slice, cell):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.clustering(slice, cell)

    def clustering(self, slice, cell):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.nearest_neighbour_degree(slice, cell)

    def spikes_vs_phase(self):
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
        return (phases, spikes)

    def correlation_vs_distance(self):
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
        return (distances, correlations_slow, correlations_fast)
