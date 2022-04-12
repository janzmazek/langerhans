import numpy as np
from scipy.stats import rankdata


class Waves(object):
    """
    Docstring for Waves.
    """

    def __init__(self):
        self.__cells = False
        self.__sampling = False
        self.__binarized_fast = False
        self.__distances_matrix = False

        self.__act_sig = False
        self.__big_events = False
        self.__all_events = False

    def get_act_sig(self): return self.__act_sig
    def get_big_events(self): return self.__big_events
    def get_all_events(self): return self.__all_events

    def import_data(self, data):
        assert data.is_analyzed()
        assert data.get_positions() is not False

        self.__cells = data.get_cells()
        self.__sampling = data.get_settings()["Sampling [Hz]"]
        self.__binarized_fast = data.get_binarized_fast()
        self.__distances_matrix = data.distances_matrix()

# -------------------------- WAVE DETECTION METHODS ---------------------------
    def detect_waves(self, time_th=0.5):
        self.__act_sig = np.zeros_like(self.__binarized_fast, int)
        frame_th = int(time_th*self.__sampling)
        R = self.__distances_matrix
        R_th = np.average(R) - np.std(R)

        neighbours = []
        for i in range(self.__cells):
            neighbours.append(np.where((R[i, :] < R_th) & (R[i, :] != 0))[0])

        # Find frames with at least 1 active cell
        active_frames = np.where(self.__binarized_fast.T == 1)[0]
        active_cells = {}
        for frame in active_frames:
            # Find indices of active cells inside active frames
            active_cells[frame] = list(
                np.where(self.__binarized_fast.T[frame, :] == 1)[0]
                )

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

        length = len(active_cells)
        count = 1

        # The rest of active frames in new iterator
        for frame in iter_active_cells:
            for k, cell in enumerate(active_cells[frame], max_event_num):
                # New event index
                if self.__binarized_fast[cell, frame-1] == 0:
                    self.__act_sig[cell, frame] = k
                # Previous event index
                else:
                    self.__act_sig[cell, frame] = self.__act_sig[cell, frame-1]

            for nn in active_cells[frame]:
                current = set(active_cells[frame])
                neighbours_nn = set(neighbours[nn])
                for nnn in list(neighbours_nn.intersection(current)):
                    self.__conditions(
                        frame, nn, nnn, frame-frame_th, frame+1, frame_th
                        )

            un_num = np.unique(self.__act_sig[:, frame])
            event_num = list(set(event_num).union(set(un_num)))
            max_event_num = max(event_num)

            count += 1
            yield count/length

    def __conditions(self, frame, nn, nnn, start, end, frame_th):
        cond0 = (nn != nnn)
        cond1 = (self.__act_sig[nn, frame] != 0)
        cond2 = (self.__act_sig[nnn, frame] != 0)
        cond3 = (self.__act_sig[nn, frame-1] != 0)
        cond4 = (self.__act_sig[nnn, frame-1] != 0)

        condx = (np.sum(self.__binarized_fast[nn, start:end]) <= frame_th)
        condy = (np.sum(self.__binarized_fast[nnn, start:end]) <= frame_th)

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
            raise ValueError("Waves not detected.")
        print("Characterizing waves...")
        # vse stevilke dogodkov razen nicle - 0=neaktivne celice
        events = np.unique(self.__act_sig[self.__act_sig != 0])
        # print(events)
        # print(events.size, np.min(events), np.max(events))

        self.__big_events, self.__all_events = [], []

        count, length = 1, events.size
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
                self.__big_events.append(characteristics)
            if active_cell_number > int(small_th*self.__cells):
                self.__all_events.append(characteristics)

            count += 1
            yield count/length

    def plot_act_sig(self, ax):
        ax.imshow(self.__act_sig, aspect='auto')

    def plot_events(self, ax):
        if self.__act_sig is False:
            raise ValueError("Waves not detected.")
        if self.__big_events is False:
            raise ValueError("Waves not characterized.")
        for e in (self.__big_events, self.__all_events):
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

                # ranks = rankdata(rast_plot[c], method="min")
                zacetki[kk].append(start_time/self.__sampling)
                zacetki[kk].append(-5)
                zacetki[kk].append(event_num)
                kk += 1

            fzacetki = np.array(zacetki, float)
            frast_plot = np.array(rast_plot, float)

            ax.scatter(frast_plot[:, 0], frast_plot[:, 1],
                       s=0.5, c=frast_plot[:, 2], marker='o'
                       )
            ax.scatter(fzacetki[:, 0], fzacetki[:, 1], s=10.0, marker='+')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Cell $i$')
