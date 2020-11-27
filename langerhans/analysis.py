import numpy as np
import matplotlib.pyplot as plt

from .networks import Networks


class Analysis(object):
    """docstring for Analysis."""

# ------------------------------- INITIALIZER --------------------------------
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
        self.__act_sig = None

        self.__networks = False

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

    def build_networks(self):
        print("Building networks...")
        # Construct networks and build networks from data
        self.__networks = Networks(self.__cells,
                                   self.__filtered_slow,
                                   self.__filtered_fast
                                   )
        self.__networks.build_networks()

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
            return np.where(M is True)[0]
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
    def get_networks(self): return self.__networks

    def get_parameters(self):
        par_cell = [dict() for c in range(self.__cells)]
        par_network = False
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
            par_cell[cell]["AMP"] = self.amplitudes()

        if self.__networks is not False:
            par_network = dict()
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

            for cell in range(self.__cells):
                par_cell[cell]["NDs"] = self.node_degree(cell)[0]
                par_cell[cell]["NDf"] = self.node_degree(cell)[1]
                par_cell[cell]["Cs"] = self.clustering(cell)[0]
                par_cell[cell]["Cf"] = self.clustering(cell)[1]
                par_cell[cell]["NNDs"] = self.nearest_neighbour_degree(cell)[0]
                par_cell[cell]["NNDf"] = self.nearest_neighbour_degree(cell)[1]

# ----------------------- INDIVIDUAL PARAMETER METHODS -----------------------

    def average_correlation(self):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        return self.__networks.average_correlation()

    def connection_distances(self):
        if self.__networks is False:
            raise ValueError("Network is not built.")
        A_dst = self.__distances_matrix()
        A_slow = self.__networks.get_A_slow()
        A_fast = self.__networks.get_A_fast()

        A_dst_slow = np.multiply(A_dst, A_slow)
        A_dst_fast = np.multiply(A_dst, A_fast)

        slow_distances, fast_distances = [], []
        for c1 in range(self.__cells):
            for c2 in range(c1):
                ds = A_dst_slow[c1, c2]
                df = A_dst_fast[c1, c2]
                if ds > 0:
                    slow_distances.append(ds)
                if df > 0:
                    fast_distances.append(df)

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
    def wave_detection(self, cell_th=0.1, time_th=0.5):
        print("Detecting waves")
        event_num = []

        cell_th = int(cell_th*self.__cells)
        bin_sig = self.__binarized_fast
        act_sig = np.zeros_like(bin_sig, int)
        frame_th = int(time_th*self.__sampling)
        R = self.__distances_matrix()
        Rth = np.average(R) - np.std(R)

        neighbours = []
        for j in range(self.__cells):
            neighbours.append(np.where((R[:, j] < Rth) & (R[:, j] != 0))[0])

        nonzero = {}
        # Poisce vse frejme, kjer je kakšna celica aktivna
        nonzero_frames = np.where(bin_sig == 1)[1]
        # zanka po frejmih z aktivnostjo
        for frame in nonzero_frames:
            # v frejmu z aktivnostjo poisce vse celice, ki so dejansko aktivne
            nonzero[frame] = list(np.where(bin_sig[:, frame] == 1)[0])

        counter = 0
        # zanka po frame-ih z aktivnostjo
        for frame in nonzero:
            # Obdela prvi aktivni frame v binsig matriki
            if counter == 0:
                k = 1
                # Zanka po aktivnih celicah v frejmu
                for cell in nonzero[frame]:
                    act_sig[cell, frame] = k
                    k += 1

                for nn in nonzero[frame]:
                    current = set(nonzero[frame])
                    neighbours_nn = set(neighbours[nn])
                    for nnn in list(neighbours_nn.intersection(current)):
                        act_sig[nn, frame] = min(act_sig[nn, frame],
                                                 act_sig[nnn, frame]
                                                 )
                        act_sig[nnn, frame] = act_sig[nn, frame]

                un_num = np.unique(act_sig[:, frame])
                event_num = list(set(event_num).union(set(un_num)))

                max_event_num = max(event_num)
                counter += 1

            else:
                k = max_event_num + 1
                # Zanka po aktivnih celicah
                for cell in nonzero[frame]:
                    # Če je celica v tem frameu aktivna, v prejsnjem pa ne, ji
                    # dodeli novo številko dogodka
                    if bin_sig[cell, frame-1] == 0:
                        act_sig[cell, frame] = k
                        k += 1
                    # Če je celica bila že v prejsnjem frejmu aktivna, ji
                    # prepiše indeks dogodka
                    else:
                        act_sig[cell, frame] = act_sig[cell-1, frame]

                for nn in nonzero[frame]:
                    current = set(nonzero[frame])
                    neighbours_nn = set(neighbours[nn])
                    for nnn in list(neighbours_nn.intersection(current)):
                        if act_sig[nn, frame] != 0 and \
                                act_sig[nnn, frame] != 0 and \
                                act_sig[nn, frame-1] != 0 and \
                                np.sum(bin_sig[nn, frame-frame_th:frame+1]) \
                                <= frame_th and \
                                act_sig[nnn, frame-1] == 0 and \
                                nn != nnn:
                            act_sig[nnn, frame] = act_sig[nn, frame]
                        if act_sig[nn, frame] != 0 and \
                                act_sig[nnn, frame] != 0 and \
                                act_sig[nn, frame-1] == 0 and \
                                act_sig[nnn, frame-1] != 0 and \
                                np.sum(bin_sig[nnn, frame-frame_th:frame+1]) \
                                <= frame_th and \
                                nn != nnn:
                            act_sig[nn, frame] = act_sig[nnn, frame]
                        if act_sig[nn, frame] != 0 and \
                                act_sig[nnn, frame] != 0 and \
                                act_sig[nn, frame-1] == 0 and \
                                act_sig[nnn, frame-1] == 0 and \
                                nn != nnn:
                            act_sig[nn, frame] = min(act_sig[nn, frame],
                                                     act_sig[nnn, frame]
                                                     )
                            act_sig[nnn, frame] = act_sig[nn, frame]

                un_num = np.unique(act_sig[:, frame])
                event_num = list(set(event_num).union(set(un_num)))

                max_event_num = max(event_num)
                counter += 1
        self.__act_sig = act_sig

    def wave_characterization(self, time_th=0.5, cell_th=0.1):
        if self.__act_sig is None:
            self.wave_detection(cell_th, time_th)
        print("Characterizing waves")
        # vse stevilke dogodkov razen nicle - 0=neaktivne celice
        un_events = np.unique(self.__act_sig[self.__act_sig != 0])

        events = []
        all_events = []

        for e in un_events:
            e = int(e)
            cells, frames = np.where(self.__act_sig == e)
            active_cell_number = np.unique(cells).size

            start_time, end_time = np.min(frames), np.max(frames)

            characteristics = {
                "event number": e,
                "start time": start_time,
                "end time": end_time,
                "active cell number": np.unique(cells).size,
                "rel active cell number": np.unique(cells).size/self.__cells
            }
            if active_cell_number > cell_th:
                events.append(characteristics)
            all_events.append(characteristics)

        return (events, all_events)

# ------------------------------ DRAWING METHODS ------------------------------

    def draw_networks(self, ax1, ax2, colors):
        return self.__networks.draw_networks(
            self.__positions, ax1, ax2, colors
            )

    def plot_events(self):
        events, all_events = self.wave_characterization()
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
