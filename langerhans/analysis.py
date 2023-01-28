import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .network import Network

PAR_NAMES = {
    "AD": "Activity Duration (s)",
    "AT": "Relative Active Time (%)",
    "OD": "Duration of Oscillations (s)",
    "F": "Frequency (1/s)",
    "ISI": "Interoscillation Interval (s)",
    "ISIV": "Interoscillation Interval Variation (s)",
    "TP": "Time to Plateau (s)",
    "TS": "Time to Activation (s)",
    "AMPs": "Amplitude Slow",
    "ND": "Node Degree",
    "C": "Clustering Coefficient",
    "NND": "Nearest Neighbour Degree",
    "R": "Average Correlation",
    "GE": "Global Efficiency",
    "MCC": "Largest Connected Component",
    "D": "Average Connection Distances (μm)"
}


class Analysis(object):
    """docstring for Analysis."""

# ------------------------------- INITIALIZER --------------------------------
    def __init__(self):
        self.__settings = False
        self.__sampling = False
        self.__cells = False
        self.__positions = False
        self.__distances_matrix = False
        self.__filtered_slow = False
        self.__binarized_slow = False
        self.__filtered_fast = False
        self.__binarized_fast = False
        self.__activity = False
        self.__network_slow = False
        self.__network_fast = False

        self.__dataframe = False

    def import_data(self, data, threshold=8):
        assert data.is_analyzed()

        good_cells = data.get_good_cells()

        self.__settings = data.get_settings()
        self.__sampling = self.__settings["Sampling [Hz]"]
        self.__cells = np.sum(good_cells)

        if self.__filtered_slow:
            self.__filtered_slow = data.get_filtered_slow()[good_cells]
            self.__binarized_slow = data.get_binarized_slow()[good_cells]
            self.__network_slow = Network(self.__filtered_slow, threshold)

        self.__filtered_fast = data.get_filtered_fast()[good_cells]
        self.__binarized_fast = data.get_binarized_fast()[good_cells]
        self.__network_fast = Network(self.__filtered_fast, threshold)

        self.__activity = data.get_activity()
        self.__good_cells = good_cells

        positions = data.get_positions()
        if positions is not False:
            distance = self.__settings["Distance [um]"]
            self.__positions = positions[self.__good_cells]*distance
            self.__distances_matrix = data.distances_matrix()

        self.__dataframe = False

# ---------------------------- ANALYSIS FUNCTIONS ----------------------------
    def __search_sequence(self, arr, seq):
        # Store sizes of input array and sequence
        seq = np.array(seq)
        Na, Nseq = arr.size, seq.size

        # Range of sequence
        r_seq = np.arange(Nseq)

        # Create a 2D array of sliding indices across the entire length of
        # input array. Match up with the input sequence & get the matching
        # starting indices.
        M = (arr[np.arange(Na-Nseq+1)[:, None] + r_seq] == seq).all(1)

        # Get the range of those indices as final output
        if M.any() > 0:
            return np.where(M)[0]
        else:
            return np.array([], dtype="int")  # No match found

# ------------------------------ GETTER METHODS ------------------------------

    def get_networks(self): return (self.__network_slow, self.__network_fast)
    def get_positions(self): return self.__positions
    def get_dataframe(self):
        if not self.__dataframe:
            raise ValueError("No dataframe.")
        else:
            return self.__dataframe

    def get_dynamic_parameters(self):
        if self.__cells is False:
            raise ValueError("Data incomplete.")
        par = [dict() for c in range(self.__cells)]
        for cell in range(self.__cells):
            par[cell]["AD"] = self.activity(cell)[0]
            par[cell]["AT"] = self.activity(cell)[1]
            par[cell]["OD"] = self.activity(cell)[2]
            par[cell]["Ff"] = self.frequency(cell)
            par[cell]["ISI"] = self.interspike(cell)[0]
            par[cell]["ISIV"] = self.interspike(cell)[1]
            par[cell]["TP"] = self.time(cell)["plateau_start"]
            par[cell]["TS"] = self.time(cell)["spike_start"]

        return par

    def get_glob_network_parameters(self):
        if self.__network_fast is False:
            raise ValueError("Data incomplete.")
        par = dict()
        par["Rf"] = self.__network_fast.average_correlation()
        # par["Qf"] = self.__network_fast.modularity()
        par["GEf"] = self.__network_fast.global_efficiency()
        par["MCCf"] = self.__network_fast.max_connected_component()
        if self.__positions is not False:
            A_dst = self.__distances_matrix
            par["Df"] = self.__network_fast.average_connection_distances(A_dst)

        return par

    def get_ind_network_parameters(self):
        if self.__network_fast is False:
            raise ValueError("Data incomplete.")
        par = [dict() for c in range(self.__cells)]
        for cell in range(self.__cells):
            par[cell]["NDf"] = self.__network_fast.degree(cell)
            par[cell]["Cf"] = self.__network_fast.clustering(cell)
            par[cell]["NNDf"] = self.__network_fast.average_neighbour_degree(
                cell
                )

        return par

    def to_pandas(self):
        data = []
        dyn_par = self.get_dynamic_parameters()
        net_par = self.get_ind_network_parameters()
        ind = [{**dyn_par[c], **net_par[c]} for c in range(self.__cells)]
        for c in range(self.__cells):
            for p in ind[c]:
                if p[-1] in ("s", "f"):
                    mode = "Slow" if p[-1] == "s" else "Fast"
                    p_stripped = p[:-1]
                else:
                    mode = "Undefined"
                    p_stripped = p
                data.append({"Islet ID": self.__settings["Islet ID"],
                             "Cell ID": c,
                             "Par ID": p,
                             "Glucose": self.__settings["Glucose [mM]"],
                             "Parameter": PAR_NAMES[p_stripped],
                             "Mode": mode,
                             "Value": ind[c][p]
                             })
        self.__dataframe = pd.DataFrame(data=data)

# ----------------------- INDIVIDUAL PARAMETER METHODS -----------------------

    def amplitudes(self):
        amplitudes = []
        for cell in range(self.__cells):
            heavisided_gradient = np.heaviside(
                np.gradient(self.__filtered_slow[cell]), 0
                )
            minima = self.__search_sequence(heavisided_gradient, [0, 1])
            maxima = self.__search_sequence(heavisided_gradient, [1, 0])

            if maxima[0] < minima[0]:
                maxima = np.delete(maxima, 0)
            if maxima[-1] < minima[-1]:
                minima = np.delete(minima, 0)

            for i, j in zip(minima, maxima):
                amplitudes.append(
                    self.__filtered_slow[cell][j]-self.__filtered_slow[cell][i]
                    )
        return amplitudes

    def activity(self, cell):
        start = int(self.__activity[0]*self.__sampling)
        stop = int(self.__activity[1]*self.__sampling)
        bin = self.__binarized_fast[cell][start:stop]
        sum = np.sum(bin)
        length = bin.size
        Nf = self.frequency(cell)*length/self.__sampling
        if sum == 0:
            return (length, np.nan, np.nan)
        return (length/self.__sampling, sum/length, (sum/self.__sampling)/Nf)

    def frequency(self, cell):
        start = int(self.__activity[0]*self.__sampling)
        stop = int(self.__activity[1]*self.__sampling)
        bin_fast = self.__binarized_fast[cell][start:stop]

        fast_peaks = self.__search_sequence(bin_fast, [0, 1])
        if fast_peaks.size < 2:
            frequency_fast = np.nan
        else:
            fast_interval = fast_peaks[-1]-fast_peaks[0]
            frequency_fast = (fast_peaks.size-1)/fast_interval*self.__sampling

        return frequency_fast

    def frequency_slow(self, cell):
        start = int(self.__activity[0]*self.__sampling)
        stop = int(self.__activity[1]*self.__sampling)
        bin_slow = self.__binarized_slow[cell][start:stop]

        slow_peaks = self.__search_sequence(bin_slow, [11, 12])
        if slow_peaks.size < 2:
            frequency_slow = np.nan
        else:
            slow_interval = slow_peaks[-1]-slow_peaks[0]
            frequency_slow = (slow_peaks.size-1)/slow_interval*self.__sampling
        return frequency_slow

    def interspike(self, cell):
        start = int(self.__activity[0]*self.__sampling)
        stop = int(self.__activity[1]*self.__sampling)
        bin_fast = self.__binarized_fast[cell][start:stop]

        IS_start = self.__search_sequence(bin_fast, [1, 0])/self.__sampling
        IS_end = self.__search_sequence(bin_fast, [0, 1])/self.__sampling

        if IS_start.size == 0 or IS_end.size == 0:
            return (np.nan, np.nan)
        # First IS_start must be before first interspike_end
        if IS_end[-1] < IS_start[-1]:
            IS_start = IS_start[:-1]
        if IS_start.size == 0:
            return (np.nan, np.nan)
        # Last IS_start must be before last interspike_end
        if IS_end[0] < IS_start[0]:
            IS_end = IS_end[1:]

        assert IS_start.size == IS_end.size

        IS_lengths = [IS_end[i]-IS_start[i] for i in range(IS_start.size)]
        mean_IS_interval = np.mean(IS_lengths)
        IS_variation = np.std(IS_lengths)/mean_IS_interval

        return (mean_IS_interval, IS_variation)

    def time(self, cell):
        bin_fast = self.__binarized_fast[cell]
        time = {}
        stim_start = 0# int(self.__settings["Stimulation [s]"][0])
        stim_end = 0#int(self.__settings["Stimulation [s]"][1])

        time["plateau_start"] = self.__activity[0] - stim_start
        time["plateau_end"] = self.__activity[1] - stim_end

        fast_peaks = self.__search_sequence(bin_fast[stim_start:], [0, 1])
        if len(fast_peaks) < 3:
            time["spike_start"] = np.nan
        else:
            time["spike_start"] = (np.mean(fast_peaks[:3]))/self.__sampling
        return time

# ----------------------------- ANALYSIS METHODS ------------------------------
# ----------------------------- Spikes vs phases ------------------------------

    def spikes_vs_phase(self, mode="normal"):
        assert self.__binarized_slow and self.__binarized_fast
        phases = np.arange((np.pi/3 - np.pi/6)/2, 2*np.pi, np.pi/6)
        spikes = np.zeros((self.__cells, 12))

        # Iterate through cells
        for cell in range(self.__cells):
            start = int(self.__activity[0]*self.__sampling)
            stop = int(self.__activity[1]*self.__sampling)

            bin_slow = self.__binarized_slow[cell][start:stop]
            bin_fast = self.__binarized_fast[cell][start:stop]

            # Iterate through phases (1–12)
            for phase in range(1, 13):
                # Bool array with True at slow phase:
                slow_isolated = bin_slow == phase

                # Bool array with True at fast spike:
                spike_indices = self.__search_sequence(bin_fast, [0, 1]) + 1
                fast_unitized = np.zeros(len(bin_fast))
                fast_unitized[spike_indices] = 1

                # Bool array with True at fast spike AND slow phase
                fast_isolated_unitized = np.logical_and(
                    slow_isolated, fast_unitized
                    )

                # Append result
                spikes[cell, phase-1] = np.sum(fast_isolated_unitized)

        if mode == "normal":
            result = np.sum(spikes, axis=0)
        elif mode == "separate":
            result = spikes
        return (phases, result)

# ------------------------- Correlation vs distance ---------------------------

    def correlation_vs_distance(self):
        assert self.__positions
        A_dst = self.__distances_matrix
        distances = []
        correlations_slow = []
        correlations_fast = []
        for cell1 in range(self.__cells):
            for cell2 in range(cell1):
                distances.append(A_dst[cell1, cell2])
                corr_slow = np.corrcoef(self.__filtered_slow[cell1],
                                        self.__filtered_slow[cell2])[0, 1]
                corr_fast = np.corrcoef(self.__filtered_fast[cell1],
                                        self.__filtered_fast[cell2])[0, 1]
                correlations_slow.append(corr_slow)
                correlations_fast.append(corr_fast)
        return (distances, correlations_slow, correlations_fast)

# ------------------------------ DRAWING METHODS ------------------------------

    def draw_network(self, ax, cell=False):
        col = None
        if cell is not False:
            col = ["C0" for _ in self.__network_fast.nodes()]
            col[cell] = "C3"
        if self.__positions is False:
            self.__network_fast.draw_network(ax=ax, node_color=col)
        else:
            self.__network_fast.draw_network(ax=ax, pos=self.__positions, node_color=col)
