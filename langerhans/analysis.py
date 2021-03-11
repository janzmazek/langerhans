import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .network import Network

PAR_NAMES = {
    "AD": "Activity Duration (s)",
    "AT": "Relative Active Time (%)",
    "OD": "Duration of Oscillations (s)",
    "F": "Frequency (1/s)",
    "ISI": "Interspike Interval (s)",
    "ISIV": "Interspike Variation (s)",
    "TP": "Time to Plateau (s)",
    "TS": "Time to Spike (s)",
    "AMPs": "Amplitude Slow",
    "ND": "Node Degree",
    "C": "Clustering Coefficient",
    "NND": "Nearest Neighbour Degree",
    "R": "Average Correlation",
    "GE": "Global Efficiency",
    "MCC": "Largest Connected Component",
}


class Analysis(object):
    """docstring for Analysis."""

# ------------------------------- INITIALIZER --------------------------------
    def __init__(self, id=None):
        self.__id = id

        self.__settings = None
        self.__points = None
        self.__cells = None
        self.__positions = None
        self.__filtered_slow = None
        self.__filtered_fast = None
        self.__binarized_slow = None
        self.__binarized_fast = None
        self.__activity = None

        self.__act_sig = False

    def import_data(self, data, positions):
        assert data.is_analyzed()

        good_cells = data.get_good_cells()

        self.__settings = data.get_settings()
        self.__sampling = self.__settings["Sampling [Hz]"]
        self.__points = data.get_points()
        distance = self.__settings["Distance [um]"]
        self.__positions = positions[good_cells]*distance
        self.__cells = np.sum(good_cells)

        self.__filtered_slow = data.get_filtered_slow()[good_cells]
        self.__filtered_fast = data.get_filtered_fast()[good_cells]

        self.__binarized_slow = data.get_binarized_slow()[good_cells]
        self.__binarized_fast = data.get_binarized_fast()[good_cells]

        self.__activity = np.array(data.get_activity())[good_cells]

    def build_networks(self, threshold=8):
        print("Building networks...")
        # Construct networks and build networks from data
        self.__network_slow = Network(
            self.__cells, self.__filtered_slow, threshold
            )
        self.__network_fast = Network(
            self.__cells, self.__filtered_fast, threshold
            )

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

    def __distances_matrix(self):
        A_dst = np.zeros((self.__cells, self.__cells))
        for cell1 in range(self.__cells):
            for cell2 in range(cell1):
                x1, y1 = self.__positions[cell1, 0], self.__positions[cell1, 1]
                x2, y2 = self.__positions[cell2, 0], self.__positions[cell2, 1]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                A_dst[cell1, cell2] = distance
                A_dst[cell2, cell1] = distance
        return A_dst

# ------------------------------ GETTER METHODS ------------------------------

    def get_positions(self): return self.__positions
    def get_filtered_slow(self): return self.__filtered_slow
    def get_filtered_fast(self): return self.__filtered_fast
    def get_act_sig(self): return self.__act_sig

    def get_dynamic_parameters(self):
        par = [dict() for c in range(self.__cells)]
        for cell in range(self.__cells):
            par[cell]["AD"] = self.activity(cell)[0]
            par[cell]["AT"] = self.activity(cell)[1]
            par[cell]["OD"] = self.activity(cell)[2]
            par[cell]["Fs"] = self.frequency(cell)[0]
            par[cell]["Ff"] = self.frequency(cell)[1]
            par[cell]["ISI"] = self.interspike(cell)[0]
            par[cell]["ISIV"] = self.interspike(cell)[1]
            par[cell]["TP"] = self.time(cell)["plateau_start"]
            par[cell]["TS"] = self.time(cell)["spike_start"]
            # par[cell]["TI"] = self.time(cell)["plateau_end"]
            # par[cell]["AMPs"] = self.amplitudes()

        return par

    def get_glob_network_parameters(self):
        if self.__network_slow is False:
            raise ValueError("Network is not built.")

        par = dict()
        par["Rs"] = self.__network_slow.average_correlation()
        par["Rf"] = self.__network_fast.average_correlation()
        par["Ds"] = self.__network_slow.connection_distances()
        par["Df"] = self.__network_fast.connection_distances()
        par["Qs"] = self.__network_slow.modularity()
        par["Qf"] = self.__network_fast.modularity()
        par["GEs"] = self.__network_slow.global_efficiency()
        par["GEf"] = self.__network_fast.global_efficiency()
        par["MCCs"] = self.__network_slow.max_connected_component()
        par["MCCf"] = self.__network_fast.max_connected_component()

        return par

    def get_ind_network_parameters(self):
        if self.__network_slow is False:
            raise ValueError("Network is not build.")

        par = [dict() for c in range(self.__cells)]
        for cell in range(self.__cells):
            par[cell]["NDs"] = self.__network_slow.degree(cell)
            par[cell]["NDf"] = self.__network_fast.degree(cell)
            par[cell]["Cs"] = self.__network_slow.clustering(cell)
            par[cell]["Cf"] = self.__network_fast.clustering(cell)
            par[cell]["NNDs"] = self.__network_slow.average_neighbour_degree(
                cell
                )
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
                data.append({"Islet ID": self.__id,
                             "Cell ID": c,
                             "Par ID": p,
                             "Glucose": self.__settings["Glucose [mM]"],
                             "Parameter": PAR_NAMES[p_stripped],
                             "Mode": mode,
                             "Value": ind[c][p]
                             })
        df = pd.DataFrame(data=data)
        return df

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

        slow_peaks = self.__search_sequence(bin_slow, [11, 12])
        if slow_peaks.size < 2:
            frequency_slow = np.nan
        else:
            slow_interval = slow_peaks[-1]-slow_peaks[0]
            frequency_slow = (slow_peaks.size-1)/slow_interval*self.__sampling

        fast_peaks = self.__search_sequence(bin_fast, [0, 1])
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

        IS_start = self.__search_sequence(bin_fast, [1, 0])
        IS_end = self.__search_sequence(bin_fast, [0, 1])

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
        stim_start = self.__settings["Stimulation [frame]"][0]
        stim_end = self.__settings["Stimulation [frame]"][1]

        time["plateau_start"] = self.__activity[cell][0] - stim_start
        time["plateau_end"] = self.__activity[cell][1] - stim_end

        fast_peaks = self.__search_sequence(bin_fast[stim_start:], [0, 1])
        if len(fast_peaks) < 3:
            time["spike_start"] = np.nan
        else:
            time["spike_start"] = (np.mean(fast_peaks[:3]))/self.__sampling

        return time

# ----------------------------- ANALYSIS METHODS ------------------------------
# ----------------------------- Spikes vs phases ------------------------------

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
        A_dst = self.__distances_matrix()
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

# -------------------------- WAVE DETECTION METHODS ---------------------------
    def detect_waves(self, time_th=0.5):
        if self.__act_sig is not False:
            return
        print("Detecting waves...")

        bin_sig = self.__binarized_fast
        self.__act_sig = np.zeros_like(bin_sig, int)
        frame_th = int(time_th*self.__sampling)
        R = self.__distances_matrix()
        R_th = np.average(R) - np.std(R)

        neighbours = []
        for i in range(self.__cells):
            neighbours.append(np.where((R[i, :] < R_th) & (R[i, :] != 0))[0])

        # Find frames with at least 1 active cell
        active_frames = np.where(bin_sig.T == 1)[0]
        active_cells = {}
        for frame in active_frames:
            # Find indices of active cells inside active frames
            active_cells[frame] = list(np.where(bin_sig.T[frame, :] == 1)[0])

        # Define new iterator from dictionary REQUIRES: python 3.6
        iter_active_cells = iter(active_cells)
        # First active frame in new iterator
        frame = next(iter_active_cells)
        for k, cell in enumerate(active_cells[frame]):
            self.__act_sig[cell, frame] = k

        for nn in active_cells[frame]:
            current = set(active_cells[frame])
            neighbours_nn = set(neighbours[nn])
            for nnn in list(neighbours_nn.intersection(current)):
                self.__act_sig[nn, frame] = min(self.__act_sig[nn, frame],
                                                self.__act_sig[nnn, frame]
                                                )
                self.__act_sig[nnn, frame] = self.__act_sig[nn, frame]

        un_num = np.unique(self.__act_sig[:, frame])
        event_num = set(un_num)
        max_event_num = max(event_num)

        # The rest of active frames in new iterator
        for frame in iter_active_cells:
            for k, cell in enumerate(active_cells[frame], max_event_num):
                # Če je celica v tem frameu aktivna, v prejsnjem pa ne, ji
                # dodeli novo številko dogodka
                if bin_sig[cell, frame-1] == 0:
                    self.__act_sig[cell, frame] = k
                # Če je celica bila že v prejsnjem frejmu aktivna, ji
                # prepiše indeks dogodka
                else:
                    self.__act_sig[cell, frame] = self.__act_sig[cell, frame-1]

            for nn in active_cells[frame]:
                current = set(active_cells[frame])
                neighbours_nn = set(neighbours[nn])
                for nnn in list(neighbours_nn.intersection(current)):
                    start, end = frame-frame_th, frame+1
                    condx = (np.sum(bin_sig[nn, start:end]) <= frame_th)
                    condy = (np.sum(bin_sig[nnn, start:end]) <= frame_th)
                    self.__conditions(frame, nn, nnn, condx, condy)

            un_num = np.unique(self.__act_sig[:, frame])
            event_num = list(set(event_num).union(set(un_num)))
            max_event_num = max(event_num)

    def __conditions(self, frame, nn, nnn, condx, condy):
        cond0 = (nn != nnn)
        cond1 = (self.__act_sig[nn, frame] != 0)
        cond2 = (self.__act_sig[nnn, frame] != 0)
        cond3 = (self.__act_sig[nn, frame-1] != 0)
        cond4 = (self.__act_sig[nnn, frame-1] != 0)

        if cond0 and cond1 and cond2 and cond3 and not cond4 and condx:
            self.__act_sig[nnn, frame] = self.__act_sig[nn, frame]
        if cond0 and cond1 and cond2 and not cond3 and cond4 and condy:
            self.__act_sig[nn, frame] = self.__act_sig[nnn, frame]
        if cond0 and cond1 and cond2 and not cond3 and not cond4:
            self.__act_sig[nn, frame] = min(self.__act_sig[nn, frame],
                                            self.__act_sig[nnn, frame]
                                            )
            self.__act_sig[nnn, frame] = self.__act_sig[nn, frame]

    def characterize_waves(self, small_th=0.1, big_th=0.45, time_th=0.5):
        if self.__act_sig is False:
            self.detect_waves(time_th)
        print("Characterizing waves...")
        # vse stevilke dogodkov razen nicle - 0=neaktivne celice
        events = np.unique(self.__act_sig[self.__act_sig != 0])
        # print(events)
        # print(events.size, np.min(events), np.max(events))

        big_events = []
        all_events = []

        for e in events:
            e = int(e)
            cells, frames = np.where(self.__act_sig == e)
            active_cell_number = np.unique(cells).size

            start_time, end_time = np.min(frames), np.max(frames)

            characteristics = {
                "event number": e,
                "start time": start_time,
                "end time": end_time,
                "active cell number": active_cell_number,
                "rel active cell number": active_cell_number/self.__cells
            }
            if active_cell_number > int(big_th*self.__cells):
                big_events.append(characteristics)
            if active_cell_number > int(small_th*self.__cells):
                all_events.append(characteristics)

        return (big_events, all_events)

# ------------------------------ DRAWING METHODS ------------------------------

    def draw_networks(self, ax1, ax2, colors):
        self.__network_slow.draw_network(self.__positions, ax1, colors[0])
        self.__network_fast.draw_network(self.__positions, ax2, colors[1])

    def plot_events(self, events, all_events):
        for e in (events, all_events):
            rast_plot = []
            zacetki = []
            k = 0
            kk = 0
            for c in e:
                zacetki.append([])
                event_num = int(c["event number"])
                start_time = int(c["start time"])
                end_time = int(c["end time"])
                active_cell_number = c["active cell number"]

                step = 0
                used = []
                init_cells = 0
                for i in range(self.__cells):
                    for j in range(start_time, end_time+1, 1):
                        if self.__act_sig[i, j] == event_num and i not in used:
                            rast_plot.append([])
                            rast_plot[k].append(
                                (start_time+step)/self.__sampling
                                )
                            rast_plot[k].append(i)
                            rast_plot[k].append(event_num)
                            rast_plot[k].append(active_cell_number)
                            used.append(i)
                            k += 1
                    init_cells += 1
                    step += 1

                zacetki[kk].append(start_time/self.__sampling)
                zacetki[kk].append(-5)
                zacetki[kk].append(event_num)
                kk += 1

            fzacetki = np.array(zacetki, float)
            frast_plot = np.array(rast_plot, float)

            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.scatter(frast_plot[:, 0], frast_plot[:, 1],
                       s=0.5, c=frast_plot[:, 2], marker='o'
                       )
            ax.scatter(fzacetki[:, 0], fzacetki[:, 1], s=10.0, marker='+')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Cell $i$')
            plt.show()
