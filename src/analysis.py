import numpy as np
import matplotlib.pyplot as plt

from .networks import Networks

class Analysis(object):
    """docstring for Analysis."""

# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self):
        self.__sampling = None
        self.__points = None
        self.__cells = None
        self.__positions = None
        self.__filtered_slow = None
        self.__filtered_fast = None
        self.__binarized_slow = None
        self.__binarized_fast = None
        self.__activity = None

        self.__networks = False

    def import_data(self, data, positions):
        assert data.is_analyzed()

        # Get settings from data object
        settings = data.get_settings()
        good_cells = data.get_good_cells()
        activity = data.get_activity()

        self.__points = data.get_points()
        self.__positions = positions
        self.__cells = np.sum(good_cells)
        self.__sampling = settings["Sampling [Hz]"]

        self.__filtered_slow = data.get_filtered_slow()[good_cells]
        self.__filtered_fast = data.get_filtered_fast()[good_cells]

        binarized_slow = data.get_binarized_slow()[good_cells]
        binarized_fast = data.get_binarized_fast()[good_cells]
        self.__binarized_slow = []
        self.__binarized_fast = []

        for cell in range(self.__cells):
            start = int(self.__sampling*activity[cell][0])
            stop = int(self.__sampling*activity[cell][1])
            self.__binarized_slow.append(binarized_slow[cell][start:stop])
            self.__binarized_fast.append(binarized_fast[cell][start:stop])

    def build_networks(self):
        print("Building networks...")
        # Construct networks and build networks from data
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

        par1 = [dict() for c in range(self.__cells)]
        par2 = dict()

        if self.__networks is not False:
            par2["Rs"] = self.average_correlation()[0]
            par2["Rf"] = self.average_correlation()[1]
        for cell in range(self.__cells):
            par1[cell]["AD"] = self.activity(cell)[0]
            par1[cell]["AT"] = self.activity(cell)[1]
            par1[cell]["Fs"] = self.frequency(cell)[0]
            par1[cell]["Ff"] = self.frequency(cell)[1]
            par1[cell]["ISI"] = self.interspike_variation(cell)[0]
            par1[cell]["ISIV"] = self.interspike_variation(cell)[1]
            if self.__networks is not False:
                par1[cell]["NDs"] = self.__networks.node_degree(cell)[0]
                par1[cell]["NDf"] = self.__networks.node_degree(cell)[1]
                par1[cell]["Cs"] = self.__networks.clustering(cell)[0]
                par1[cell]["Cf"] = self.__networks.clustering(cell)[1]
                par1[cell]["NNDs"] = self.__networks.nearest_neighbour_degree(cell)[0]
                par1[cell]["NNDf"] = self.__networks.nearest_neighbour_degree(cell)[1]

        return (par1, par2)

    def average_correlation(self):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.average_correlation()

    def activity(self, cell):
        bin = self.__binarized_fast[cell]
        sum = np.sum(bin)
        length = bin.size
        if sum == 0:
            return (length, np.nan)
        return (length, sum/length)

    def frequency(self, cell):
        bin_slow = self.__binarized_slow[cell]
        bin_fast = self.__binarized_fast[cell]

        slow_peaks = self.__search_sequence(bin_slow, [11,12])
        if slow_peaks.size < 2:
            frequency_slow = np.nan
        else:
            slow_interval = slow_peaks[-1]-slow_peaks[0]
            frequency_slow = (slow_peaks.size-1)/slow_interval*self.__sampling

        fast_peaks = self.__search_sequence(bin_fast, [0,1])
        if fast_peaks.size < 2:
            frequency_fast = np.nan
        else:
            fast_interval = fast_peaks[-1]-fast_peaks[0]
            frequency_fast = (fast_peaks.size-1)/fast_interval*self.__sampling

        return (frequency_slow, frequency_fast)

    def interspike_variation(self, cell):
        bin_fast = self.__binarized_fast[cell]

        interspike_start = self.__search_sequence(bin_fast, [1,0])
        interspike_end = self.__search_sequence(bin_fast, [0,1])

        if interspike_start.size == 0 or interspike_end.size == 0:
            return (np.nan, np.nan)
        # First interspike_start must be before first interspike_end
        if interspike_end[-1] < interspike_start[-1]:
            interspike_start = interspike_start[:-1]
        if interspike_start.size == 0:
            return (np.nan, np.nan)
        # Last interspike_start must be before last interspike_end
        if interspike_end[0] < interspike_start[0]:
            interspike_end = interspike_end[1:]

        assert interspike_start.size == interspike_end.size

        interspike_lengths = [interspike_end[i]-interspike_start[i] for i in range(interspike_start.size)]
        mean_interspike_interval = np.mean(interspike_lengths)
        interspike_variation = np.std(interspike_lengths)/mean_interspike_interval

        return (mean_interspike_interval, interspike_variation)

    def node_degree(self, cell):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.node_degree(cell)

    def clustering(self, cell):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.clustering(cell)

    def nearest_neighbour_degree(self, cell):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.nearest_neighbour_degree(cell)

    def spikes_vs_phase(self):
        phases = np.arange((np.pi/3 - np.pi/6)/2, 2*np.pi, np.pi/6)
        spikes = np.zeros(12)
        for phase in range(1,13):
            for cell in range(self.__cells):
                slow_isolated = self.__binarized_slow[cell] == phase

                bin_fast = self.__binarized_fast[cell]
                spike_indices = self.__search_sequence(bin_fast, [0,1]) + 1
                fast_unitized = np.zeros(len(bin_fast))
                fast_unitized[spike_indices] = 1

                fast_isolated_unitized = np.logical_and(slow_isolated, fast_unitized)

                spikes[phase-1] += np.sum(fast_isolated_unitized)
        return (phases, spikes)

    def correlation_vs_distance(self):
        distances = []
        correlations_slow = []
        correlations_fast = []
        for cell1 in range(self.__cells):
            for cell2 in range(cell1):
                x1, y1 = self.__positions[cell1,0], self.__positions[cell1,1]
                x2, y2 = self.__positions[cell2,0], self.__positions[cell2,1]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                distances.append(distance)

                corr_slow = np.corrcoef(self.__filtered_slow[cell1],
                                        self.__filtered_slow[cell2])[0,1]
                corr_fast = np.corrcoef(self.__filtered_fast[cell1],
                                        self.__filtered_fast[cell2])[0,1]
                correlations_slow.append(corr_slow)
                correlations_fast.append(corr_fast)
        return (distances, correlations_slow, correlations_fast)
