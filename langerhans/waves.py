import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt


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
    def detect_waves(self):
        for _ in self.detect_waves_progress():
            pass
    
    def detect_waves_progress(self, time_th=0.5, real_time=False):
        self.__act_sig = np.zeros_like(self.__binarized_fast, int)
        frame_th = int(time_th*self.__sampling)
        R = self.__distances_matrix
        R_th = np.average(R) - np.std(R)

        # Calculate neighbours based on pairwise correlation
        neighbours = {}
        for cell in range(self.__cells):
            neighbours[cell] = np.where((R[cell, :] < R_th) & (R[cell, :] != 0))[0]

        # Calculate active frames with at least 1 active cell
        active_frames = np.where(self.__binarized_fast.sum(axis=0))[0]
        
        # Calculate indices of active cells inside active frames
        active_cells = {}
        for frame in active_frames:
            active_cells[frame] = list(
                np.where(self.__binarized_fast.T[frame, :] == 1)[0]
                )

        for i in self.__fill_act_sig(neighbours, active_frames, active_cells, frame_th, real_time):
            yield i

    def __fill_act_sig(self, neighbours, active_frames, active_cells, frame_th, real_time):
        event_num = self.__fill_first_frame(neighbours, active_frames[0], active_cells)
        max_event_num = max(event_num)

        length = len(active_frames)
        count = 1

        for frame in active_frames[1:]:
            for k, cell in enumerate(active_cells[frame], max_event_num):
                # If newly active cell, assign a new unique event number
                if self.__binarized_fast[cell, frame-1] == 0:
                    self.__act_sig[cell, frame] = k
                # If already active cell, assign previous event number
                else:
                    self.__act_sig[cell, frame] = self.__act_sig[cell, frame-1]

            for nn in active_cells[frame]:
                current = set(active_cells[frame])
                neighbours_nn = set(neighbours[nn])
                for nnn in list(neighbours_nn.intersection(current)):
                    self.__conditions(
                        frame, nn, nnn, frame-frame_th, frame+1, frame_th
                        )

            count += 1

            to_remove = []
            for i in event_num:
                if i not in self.__act_sig[:,frame]:
                    to_remove.append(i)
                    if real_time:
                        char = self.__characterize_event(i)
                        if char["active cell number"] > int(0.45*self.__cells):
                            yield char
                    else:
                        yield count/length
            for i in to_remove:
                event_num.remove(i)

            un_num = set(np.unique(self.__act_sig[:, frame]))
            event_num = event_num.union(un_num)
            max_event_num = max(event_num)

    def __fill_first_frame(self, neighbours, frame, active_cells):
        # First active frame in new iterator
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
        return event_num

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

    def characterize_waves(self, small_th=0.1, big_th=0.45):
        if self.__act_sig is False:
            raise ValueError("Waves not detected.")
        print("Characterizing waves...")
        # vse stevilke dogodkov razen nicle - 0=neaktivne celice
        events = np.unique(self.__act_sig[self.__act_sig != 0])

        self.__big_events, self.__all_events = [], []

        count, length = 1, events.size
        for e in events:
            char = self.__characterize_event(e)
            if char["active cell number"] > int(big_th*self.__cells):
                self.__big_events.append(char)
            if char["active cell number"] > int(small_th*self.__cells):
                self.__all_events.append(char)

            count += 1
            yield count/length

    def __characterize_event(self, e):
            e = int(e)
            cells, frames = np.where(self.__act_sig == e)
            active_cell_number = np.unique(cells).size

            start_time, end_time = np.min(frames), np.max(frames)

            return {"event number": e,
                    "start time": start_time,
                    "end time": end_time,
                    "active cell number": active_cell_number,
                    "rel active cell number": active_cell_number/self.__cells
                    }

    def plot_act_sig(self, ax):
        ax.imshow(self.__act_sig, cmap=plt.get_cmap("jet_r"), aspect='auto', vmin=0, vmax=np.max(self.__act_sig))

    def plot_events_real_time(self, ax, e):
        rast_plot = []
        zacetki = []
        k = 0
        kk = 0
        zacetki.append([])
        event_num = int(e["event number"])
        start_time = int(e["start time"])
        end_time = int(e["end time"])
        active_cell_number = e["active cell number"]

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

    def plot_events(self, ax):
        if self.__act_sig is False:
            raise ValueError("Waves not detected.")
        if self.__big_events is False:
            raise ValueError("Waves not characterized.")
        for e in (self.__big_events, self.__all_events):
            self.plot_events_ral_time(ax, e)
