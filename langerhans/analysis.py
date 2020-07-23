import numpy as np
import matplotlib.pyplot as plt

from .networks import Networks

class Analysis(object):
    """docstring for Analysis."""

# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self):
        self.__settings = None
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

        good_cells = data.get_good_cells()

        self.__settings = data.get_settings()
        self.__sampling = self.__settings["Sampling [Hz]"]
        self.__points = data.get_points()
        self.__positions = positions[good_cells]*self.__settings["Distance [um]"]
        self.__cells = np.sum(good_cells)

        self.__filtered_slow = data.get_filtered_slow()[good_cells]
        self.__filtered_fast = data.get_filtered_fast()[good_cells]

        self.__binarized_slow = data.get_binarized_slow()[good_cells]
        self.__binarized_fast = data.get_binarized_fast()[good_cells]

        self.__activity = np.array(data.get_activity())[good_cells]


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

    def __distances_matrix(self):
        A_dst = np.zeros((self.__cells, self.__cells))
        for cell1 in range(self.__cells):
            for cell2 in range(cell1):
                x1, y1 = self.__positions[cell1,0], self.__positions[cell1,1]
                x2, y2 = self.__positions[cell2,0], self.__positions[cell2,1]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                A_dst[cell1, cell2] = distance
                A_dst[cell2, cell1] = distance
        return A_dst

# ------------------------------ GETTER METHODS ------------------------------ #

    def get_positions(self): return self.__positions
    def get_filtered_slow(self): return self.__filtered_slow
    def get_filtered_fast(self): return self.__filtered_fast
    def get_networks(self): return self.__networks

# ----------------------------- PARAMETER METHODS ---------------------------- #

    def draw_networks(self, ax1, ax2, colors):
        return self.__networks.draw_networks(self.__positions, ax1, ax2, colors)

    def compute_parameters(self):

        par_cell = [dict() for c in range(self.__cells)]
        par_network = dict()

        if self.__networks is not False:
            par_network["Rs"] = self.average_correlation()[0]
            par_network["Rf"] = self.average_correlation()[1]
            par_network["Ds"] = self.connection_distances()[0]
            par_network["Df"] = self.connection_distances()[1]
            par_network["Qs"] = self.modularity()[0]
            par_network["Qf"] = self.modularity()[1]
            par_network["GEs"] = self.global_efficiency()[0]
            par_network["GEf"] = self.global_efficiency()[1]
            par_network["MCCs"] = self.max_connected_component()[0]
            par_network["MCCf"] = self.max_connected_component()[1]
        par_network["MA"] = self.mean_amplitude()
        for cell in range(self.__cells):
            par_cell[cell]["AD"] = self.activity(cell)[0]
            par_cell[cell]["AT"] = self.activity(cell)[1]
            par_cell[cell]["OD"] = self.activity(cell)[2]
            par_cell[cell]["Fs"] = self.frequency(cell)[0]
            par_cell[cell]["Ff"] = self.frequency(cell)[1]
            par_cell[cell]["ISI"] = self.interspike(cell)[0]
            par_cell[cell]["ISIV"] = self.interspike(cell)[1]
            par_cell[cell]["TP"] = self.time(cell)["plateau_start"]
            par_cell[cell]["TS"] = self.time(cell)["spike_start"]
            par_cell[cell]["TI"] = self.time(cell)["plateau_end"]

            if self.__networks is not False:
                par_cell[cell]["NDs"] = self.node_degree(cell)[0]
                par_cell[cell]["NDf"] = self.node_degree(cell)[1]
                par_cell[cell]["Cs"] = self.clustering(cell)[0]
                par_cell[cell]["Cf"] = self.clustering(cell)[1]
                par_cell[cell]["NNDs"] = self.nearest_neighbour_degree(cell)[0]
                par_cell[cell]["NNDf"] = self.nearest_neighbour_degree(cell)[1]

        return (par_cell, par_network)

# ----------------------- INDIVIDUAL PARAMETER METHODS ----------------------- #

    def average_correlation(self):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.average_correlation()

    def connection_distances(self):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        A_dst = self.__distances_matrix()
        A_slow, A_fast = self.__networks.get_A_slow(), self.__networks.get_A_fast()

        A_dst_slow = np.multiply(A_dst, A_slow)
        A_dst_fast = np.multiply(A_dst, A_fast)

        slow_distances, fast_distances = [], []
        for c1 in range(self.__cells):
            for c2 in range(c1):
                ds = A_dst_slow[c1,c2]
                df = A_dst_fast[c1,c2]
                if ds > 0: slow_distances.append(ds)
                if df > 0: fast_distances.append(df)

        return (np.array(slow_distances), np.array(fast_distances))

    def modularity(self):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.modularity()

    def global_efficiency(self):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.global_efficiency()

    def max_connected_component(self):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.max_connected_component()

    def mean_amplitude(self):
        amplitudes = []
        for cell in range(self.__cells):
            heavisided_gradient = np.heaviside(np.gradient(self.__filtered_slow[cell]), 0)
            minima = self.__search_sequence(heavisided_gradient, [0,1])
            maxima = self.__search_sequence(heavisided_gradient, [1,0])

            if maxima[0] < minima[0]: maxima = np.delete(maxima, 0)
            if maxima[-1] < minima[-1]: minima = np.delete(minima, 0)

            for i, j in zip(minima, maxima):
                amplitudes.append(self.__filtered_slow[cell][j]-self.__filtered_slow[cell][i])
        return np.mean(amplitudes)

    def activity(self, cell):
        start = int(self.__activity[cell][0]*self.__sampling)
        stop = int(self.__activity[cell][1]*self.__sampling)
        bin = self.__binarized_fast[cell][start:stop]
        sum = np.sum(bin)
        length = bin.size
        Nf = self.frequency(cell)[1]*length/self.__sampling
        if sum == 0:
            return (length, np.nan, np.nan)
        return (length/self.__sampling, sum/length, (sum/self.__sampling)/Nf)

    def frequency(self, cell):
        start = int(self.__activity[cell][0]*self.__sampling)
        stop = int(self.__activity[cell][1]*self.__sampling)
        bin_slow = self.__binarized_slow[cell][start:stop]
        bin_fast = self.__binarized_fast[cell][start:stop]

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

    def interspike(self, cell):
        start = int(self.__activity[cell][0]*self.__sampling)
        stop = int(self.__activity[cell][1]*self.__sampling)
        bin_fast = self.__binarized_fast[cell][start:stop]

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

    def time(self, cell):
        bin_fast = self.__binarized_fast[cell]
        time = {}
        stimulation_start = self.__settings["Stimulation [frame]"][0]
        stimulation_end = self.__settings["Stimulation [frame]"][1]

        time["plateau_start"] = self.__activity[cell][0] - stimulation_start/self.__sampling
        time["plateau_end"] = self.__activity[cell][1] - stimulation_end/self.__sampling

        fast_peaks = self.__search_sequence(bin_fast[stimulation_start:], [0,1])
        if len(fast_peaks) < 3:
            time["spike_start"] = np.nan
        else:
            time["spike_start"] = (np.mean(fast_peaks[:3]))/self.__sampling

        return time


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

# ----------------------------- ANALYSIS METHODS ----------------------------- #
# ----------------------------- Spikes vs phases ----------------------------- #

    def spikes_vs_phase(self, mode="normal"):
        phases = np.arange((np.pi/3 - np.pi/6)/2, 2*np.pi, np.pi/6)
        spikes = np.zeros((self.__cells, 12))

        # Iterate through cells
        for cell in range(self.__cells):
            start = int(self.__activity[cell][0]*self.__sampling)
            stop = int(self.__activity[cell][1]*self.__sampling)

            bin_slow = self.__binarized_slow[cell][start:stop]
            bin_fast = self.__binarized_fast[cell][start:stop]

            # Iterate through phases (1–12)
            for phase in range(1,13):
                # Bool array with True at slow phase:
                slow_isolated = bin_slow == phase

                # Bool array with True at fast spike:
                spike_indices = self.__search_sequence(bin_fast, [0,1]) + 1
                fast_unitized = np.zeros(len(bin_fast))
                fast_unitized[spike_indices] = 1

                # Bool array with True at fast spike AND slow phase
                fast_isolated_unitized = np.logical_and(slow_isolated, fast_unitized)

                # Append result
                spikes[cell, phase-1] = np.sum(fast_isolated_unitized)

        if mode == "normal":
            result = np.sum(spikes, axis=0)
        elif mode == "separate":
            result = spikes
        return (phases, result)

# ------------------------- Correlation vs distance -------------------------- #

    def correlation_vs_distance(self):
        A_dst = self.__distances_matrix()
        distances = []
        correlations_slow = []
        correlations_fast = []
        for cell1 in range(self.__cells):
            for cell2 in range(cell1):
                distances.append(A_dst[cell1, cell2])
                corr_slow = np.corrcoef(self.__filtered_slow[cell1],
                                        self.__filtered_slow[cell2])[0,1]
                corr_fast = np.corrcoef(self.__filtered_fast[cell1],
                                        self.__filtered_fast[cell2])[0,1]
                correlations_slow.append(corr_slow)
                correlations_fast.append(corr_fast)
        return (distances, correlations_slow, correlations_fast)
