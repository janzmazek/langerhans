import numpy as np

from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew
from scipy.optimize import differential_evolution

import matplotlib.transforms as transforms
import matplotlib.patches as patches

EXCLUDE_COLOR = 'xkcd:salmon'
SAMPLE_SETTINGS = {
    "Islet ID": "S",
    "Glucose [mM]": [6, 8],
    "Stimulation [s]": [120],
    "Sampling [Hz]": 10,
    "Filter":
        {
        "Slow [Hz]": [0.001, 0.005],
        "Fast [Hz]": [0.04, 0.4],
        "Plot [s]": [250, 1750]
        },
    "Exclude":
        {
        "Score threshold": 1,
        "Spikes threshold": 0.01
        },
    "Distance [um]": 1,
    "Network threshold": 8
    }
STD_RATIO = 2


class Data(object):
    """
    A class for signal analysis.
    """
# ------------------------------- INITIALIZER ---------------------------------
    def __init__(self):
        self.__signal = False
        self.__mean_islet = False
        self.__time = False
        self.__settings = SAMPLE_SETTINGS
        self.__points = False
        self.__cells = False
        self.__positions = False
        self.__filtered_slow = False
        self.__filtered_fast = False
        self.__distributions = False
        self.__binarized_slow = False
        self.__binarized_fast = False
        self.__activity = False
        self.__good_cells = False

# --------------------------------- IMPORTS -----------------------------------
    def import_data_custom(self, signal):
        signal = np.around(signal[:, 1:].transpose(), decimals=3)
        self.import_data(signal)

    def import_data(self, signal):
        if not len(signal.shape) == 2:
            raise ValueError("Signal shape not 2D.")
        self.__signal = signal
        self.__mean_islet = np.mean(self.__signal, 0)  # average over 0 axis
        self.__mean_islet = self.__mean_islet - np.mean(self.__mean_islet)
        sampling = self.__settings["Sampling [Hz]"]
        self.__time = np.arange(len(self.__signal[0]))*(1/sampling)

        self.__points = len(self.__time)
        self.__cells = len(self.__signal)

        self.__good_cells = np.ones(self.__cells, dtype="bool")

    def import_positions(self, positions):
        if self.__signal is False:
            raise ValueError("No imported data!")
        if len(positions) != self.__cells:
            raise ValueError("Cell number does not match.")
        self.__positions = positions

    def distances_matrix(self):
        if self.__positions is False:
            raise ValueError("No positions specified.")
        A_dst = np.zeros((self.__cells, self.__cells))
        for cell1 in range(self.__cells):
            for cell2 in range(cell1):
                x1, y1 = self.__positions[cell1, 0], self.__positions[cell1, 1]
                x2, y2 = self.__positions[cell2, 0], self.__positions[cell2, 1]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                A_dst[cell1, cell2] = distance
                A_dst[cell2, cell1] = distance
        return A_dst

    def import_settings(self, settings):
        for key in settings:
            if key not in self.__settings:
                continue
            if isinstance(settings[key], dict):
                for subkey in settings[key]:
                    if subkey in self.__settings[key]:
                        self.__settings[key][subkey] = settings[key][subkey]
            else:
                self.__settings[key] = settings[key]

    def import_good_cells(self, cells):
        if self.__signal is False:
            raise ValueError("No imported data!")
        if len(cells) != self.__cells:
            raise ValueError("Cell number does not match.")
        self.__good_cells = cells

    def reset_computations(self):
        self.__filtered_slow = False
        self.__filtered_fast = False
        self.__distributions = False
        self.__binarized_slow = False
        self.__binarized_fast = False
        self.__activity = False
        self.__good_cells = np.ones(self.__cells, dtype="bool")

# --------------------------------- GETTERS -----------------------------------

    def get_signal(self): return self.__signal
    def get_mean_islet(self): return self.__mean_islet
    def get_time(self): return self.__time
    def get_settings(self): return self.__settings
    def get_points(self): return self.__points
    def get_cells(self): return self.__cells
    def get_positions(self): return self.__positions
    def get_filtered_slow(self): return self.__filtered_slow
    def get_filtered_fast(self): return self.__filtered_fast
    def get_distributions(self): return self.__distributions
    def get_binarized_slow(self): return self.__binarized_slow
    def get_binarized_fast(self): return self.__binarized_fast
    def get_activity(self): return self.__activity
    def get_good_cells(self): return self.__good_cells

# ----------------------------- ANALYSIS METHODS ------------------------------
# ---------- Filter + smooth ---------- #

    def filter_fast(self):
        for _ in self.filter_fast_progress():
            pass

    def filter_fast_progress(self):
        if self.__signal is False:
            raise ValueError("No imported data!")
        fast = self.__settings["Filter"]["Fast [Hz]"]
        self.__filtered_fast = np.zeros((self.__cells, self.__points))

        for cell in range(self.__cells):
            self.__filtered_fast[cell] = self.__bandpass(self.__signal[cell],
                                                         *fast
                                                         )
            yield (cell+1)/self.__cells

    def filter_slow(self):
        for _ in self.filter_slow_progress():
            pass

    def filter_slow_progress(self):
        if self.__signal is False:
            raise ValueError("No imported data!")
        slow = self.__settings["Filter"]["Slow [Hz]"]
        self.__filtered_slow = np.zeros((self.__cells, self.__points))

        for cell in range(self.__cells):
            self.__filtered_slow[cell] = self.__bandpass(self.__signal[cell],
                                                         *slow
                                                         )
            yield (cell+1)/self.__cells

    def __bandpass(self, data, lowcut, highcut, order=5):
        nyq = 0.5*self.__settings["Sampling [Hz]"]
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(
            order, [low, high], analog=False, btype='band', output='sos'
            )
        y = sosfiltfilt(sos, data)
        return y

# ---------- Distributions ---------- #

    def compute_distributions(self):
        for _ in self.compute_distributions_progress():
            pass

    def compute_distributions_progress(self):
        if self.__filtered_fast is False:
            raise ValueError("No filtered data.")
        self.__distributions = [dict() for i in range(self.__cells)]

        for cell in range(self.__cells):
            signal = self.__filtered_fast[cell]
            signal /= np.max(np.abs(signal))

            # Define noise from time 0 to start of stimulation
            midpoints = self.__settings["Stimulation [s]"]
            if len(midpoints) > 1:
                start, end = midpoints[0], midpoints[-1]
            else:
                start, end = midpoints[0], self.__time[-1]
            sampling = self.__settings["Sampling [Hz]"]
            start, end = int(start*sampling), int(end*sampling)
            noise = signal[:start]
            if end != 0:
                spikes = signal[start:end]
            else:
                spikes = signal[start:]

            # Remove outliers
            q1 = np.quantile(noise, 0.25)
            q3 = np.quantile(noise, 0.75)
            iqr = q3 - q1
            noise = noise[np.logical_and(noise > q1-1.5*iqr,
                                         noise < q3+1.5*iqr
                                         )]

            # Distribution parameters of noise
            noise_params = (skew(noise), np.mean(noise), np.std(noise))
            spikes_params = (skew(spikes), np.mean(spikes), np.std(spikes))

            self.__distributions[cell]["noise_params"] = noise_params
            self.__distributions[cell]["spikes_params"] = spikes_params
            self.__distributions[cell]["noise_hist"] = np.histogram(noise, 20)
            self.__distributions[cell]["spikes_hist"] = np.histogram(
                spikes, 100
                )
            yield (cell+1)/self.__cells

# ---------- Exclude ---------- #
    def autoexclude(self):
        for _ in self.autoexclude_progress():
            pass

    def autoexclude_progress(self):
        if self.__distributions is False:
            raise ValueError("No distributions.")
        # Excluding thresholds
        score_threshold = self.__settings["Exclude"]["Score threshold"]

        for cell in range(self.__cells):
            skew = self.__distributions[cell]["spikes_params"][0]
            noise_std = self.__distributions[cell]["noise_params"][2]
            spikes_std = self.__distributions[cell]["spikes_params"][2]

            # Excluding algorithm
            if skew < score_threshold and spikes_std < STD_RATIO*noise_std:
                self.__good_cells[cell] = False
            yield (cell+1)/self.__cells
        print("{} of {} good cells ({:0.0f}%)".format(
            np.sum(self.__good_cells), self.__cells,
            np.sum(self.__good_cells)/self.__cells*100)
            )

    def exclude(self, i):
        if i not in range(self.__cells):
            raise ValueError("Cell not in range.")
        self.__good_cells[i] = False

    def unexclude(self, i):
        if i not in range(self.__cells):
            raise ValueError("Cell not in range.")
        self.__good_cells[i] = True

# ---------- Binarize ---------- #

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

    def binarize_fast(self):
        for _ in self.binarize_fast_progress():
            pass

    def binarize_fast_progress(self):
        if self.__filtered_fast is False:
            raise ValueError("No filtered data.")
        if self.__distributions is False:
            raise ValueError("No distributions.")

        spikes_th = self.__settings["Exclude"]["Spikes threshold"]
        self.__binarized_fast = np.zeros((self.__cells, self.__points), int)
        for cell in range(self.__cells):
            threshold = 3*self.__distributions[cell]["noise_params"][2]
            self.__binarized_fast[cell] = np.where(
                self.__filtered_fast[cell] > threshold, 1, 0
                )
            if np.sum(self.__binarized_fast[cell]) < spikes_th*self.__points:
                self.__good_cells[cell] = False
            yield (cell+1)/self.__cells

    def binarize_slow(self):
        for _ in self.binarize_slow_progress():
            pass

    def binarize_slow_progress(self):
        if self.__filtered_slow is False:
            raise ValueError("No filtered data.")
        self.__binarized_slow = np.zeros((self.__cells, self.__points), int)
        for cell in range(self.__cells):
            signal = self.__filtered_slow[cell]
            heavisided_gradient = np.heaviside(np.gradient(signal), 0)
            minima = self.__search_sequence(heavisided_gradient, [0, 1])
            maxima = self.__search_sequence(heavisided_gradient, [1, 0])
            extremes = np.sort(np.concatenate((minima, maxima)))

            reverse_mode = False if minima[0] < maxima[0] else True

            self.__binarized_slow[cell, 0:extremes[0]] = 0
            for i in range(len(extremes)-1):
                e1, e2 = extremes[i], extremes[i+1]
                if i % 2 == int(reverse_mode):
                    self.__binarized_slow[cell, e1:e2] = np.floor(
                        np.linspace(1, 7, e2-e1, endpoint=False)
                        )
                else:
                    self.__binarized_slow[cell, e1:e2] = np.floor(
                        np.linspace(7, 13, e2-e1, endpoint=False)
                        )
            self.__binarized_slow[cell, extremes[-1]:] = 0
            yield (cell+1)/self.__cells
        self.__binarized_slow = self.__binarized_slow.astype(int)

    def crop(self, fixed_boundaries=True):
        if self.__binarized_fast is False:
            raise ValueError("No binarized data.")
        if fixed_boundaries is True:
            end = self.__time[-1]
            self.__activity = (0, end)
        else:
            self.__activity = list(fixed_boundaries)
        self.__activity = np.array(self.__activity)

    def autolimit(self):
        if self.__binarized_fast is False:
            raise ValueError("No binarized data.")
        data = np.mean(self.__binarized_fast, axis=0)
        cumsum = np.cumsum(data)

        sampling = self.__settings["Sampling [Hz]"]
        lower_limit = cumsum[cumsum < 0.1*cumsum[-1]].size
        lower_limit /= sampling  # lower limit in seconds
        upper_limit = (cumsum.size - cumsum[cumsum > 0.9*cumsum[-1]].size)
        upper_limit /= sampling  # upper limit in seconds

        def box(t, a, t_start, t_end):
            return a*(np.heaviside(t-t_start, 0)-np.heaviside(t-t_end, 0))
        res = differential_evolution(
            lambda p: np.sum((box(self.__time, *p) - data)**2),
            [[0, 100],
            [0, lower_limit+1],
            [upper_limit-1, self.__time[-1]]]
            )
        self.__activity = res.x[1:]

    def is_analyzed(self, slow=False):
        if slow:
            if self.__filtered_slow is False or self.__binarized_slow is False:
                return False
        if self.__filtered_fast is False or self.__binarized_fast is False:
            return False
        if self.__distributions is False:
            return False
        elif self.__good_cells is False:
            return False
        elif self.__activity is False:
            return False
        return True

# ----------------------------- PLOTTING METHODS ------------------------------

    def plot(self, ax, cell, plots=("raw",),
             protocol=True, stimulation=True, activity=True, excluded=True, 
             noise=False):
        if "mean" in plots:
            signal = self.__mean_islet
            signal /= np.max(np.abs(signal))
            ax.plot(self.__time, signal, "k", alpha=0.25, lw=0.1)
        if "raw" in plots:
            signal = self.__signal[cell]
            signal = signal - np.mean(signal)
            signal /= np.max(np.abs(signal))
            ax.plot(self.__time, signal, "k", alpha=0.5, lw=0.1)
        if "slow" in plots:
            filtered_slow = self.__filtered_slow[cell]
            filtered_slow = filtered_slow/np.max(filtered_slow)
            ax.plot(self.__time, filtered_slow, color="C0", lw=2)
        if "fast" in plots:
            filtered_fast = self.__filtered_fast[cell]
            filtered_fast = filtered_fast/np.max(filtered_fast)
            ax.plot(self.__time, filtered_fast, color="C3", lw=0.2)
        if "bin_slow" in plots:
            binarized_slow = self.__binarized_slow[cell]
            ax2 = ax.twinx()
            ax2.plot(self.__time, binarized_slow, color="k", lw=1)
            ax2.set_ylabel("Phase")
        if "bin_fast" in plots:
            filtered_fast = self.__filtered_fast[cell]
            binarized_fast = self.__binarized_fast[cell]/np.max(filtered_fast)
            binarized_fast *= 3*self.__distributions[cell]["noise_params"][2]
            ax.plot(self.__time, binarized_fast, color="k", lw=1)
            ax2 = ax.twinx()
            ax2.set_ylabel("Action potentials")
        if "bin_mean" in plots:
            binarized_mean = np.mean(self.__binarized_fast, axis=0)
            ax.plot(self.__time, binarized_mean, color="k", lw=1)

        ax.set_xlim(0, self.__time[-1])
        ax.set_ylim(None, 1.1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")

        if protocol:
            self.__plot_protocol(ax)

        if stimulation:
            self.__plot_stimulation(ax)

        if activity:
            self.__plot_activity(ax)

        if noise:
            self.__plot_noise(ax, cell)

        if excluded:
            self.__plot_excluded(ax, cell)

    def __plot_protocol(self, ax):
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes
            )

        midpoints = [0] + self.__settings["Stimulation [s]"] + [self.__time[-1]]
        widths = [midpoints[i+1]-midpoints[i] for i in range(len(midpoints)-1)]

        glucose = self.__settings["Glucose [mM]"]
        rectangles = []
        max_stim = np.max(glucose)

        for i, g in enumerate(glucose):
            rectangles.append(
                patches.Rectangle(
                (midpoints[i], 1), widths[i], 0.1*g/max_stim,
                color='grey', alpha=0.5, transform=trans, clip_on=False
                )
            )

        for i, r in enumerate(rectangles):
            ax.add_artist(r)
            rx, ry = r.get_xy()
            cx = rx + r.get_width()/2.0
            cy = ry + r.get_height()/2.0
            ax.annotate(glucose[i], (cx, cy), color='k', fontsize=12,
                        ha='center', va='center', xycoords=trans,
                        annotation_clip=False
                        )

    def __plot_stimulation(self, ax):
        for i in self.__settings["Stimulation [s]"]:
            ax.axvline(i, c="grey")

    def __plot_activity(self, ax):
        if self.__activity is not False:
            if self.__activity is not False:
                low, high = self.__activity
                ax.axvspan(0, low, alpha=0.25, color="grey")
                ax.axvspan(
                    high, self.__time[-1], alpha=0.25, color="grey"
                    )
    def __plot_excluded(self, ax, cell):
            if not self.__good_cells[cell]:
                ax.axvspan(0, self.__time[-1], alpha=0.5, color=EXCLUDE_COLOR)

    def __plot_noise(self, ax, cell):
        noise = 3*self.__distributions[cell]["noise_params"][2]
        ax.fill_between(self.__time, -noise, noise, color="C3", alpha=0.25,
                        label=r"$3\cdot$STD"
                        )

    def plot_distributions(self, ax, i, interval="noise"):
        if self.__distributions is False:
            raise ValueError("No distribution data.")

        if interval == "noise":
            _, mean, std = self.__distributions[i]["noise_params"]
            skew = False
            h, bins = self.__distributions[i]["noise_hist"]
            color = EXCLUDE_COLOR
            ax.invert_xaxis()
            ax.set_ylabel("Noise amplitude")
        elif interval == "signal":
            skew, mean, std = self.__distributions[i]["spikes_params"]
            h, bins = self.__distributions[i]["spikes_hist"]
            color = "C2"
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("Signal amplitude")
            ax.yaxis.tick_right()
        ax.set_xlabel("Distribution")

        delta_noise = bins[1] - bins[0]
        label = "Skew: {:.2f}".format(skew) if skew else None
        ax.barh(bins[:-1], h, delta_noise, color="grey", label=label)
        ax.axhline(mean, c="k", label="Mean")
        ax.axhspan(mean-std, mean+std, alpha=0.5,
                   color=color, label="STD: {:.2f}".format(std)
                   )
        ax.legend(loc="upper center", prop={'size': 6})
        if skew:
            # Arrow direction
            direc = 0.5 if skew > 0 else -0.5
            trans = transforms.blended_transform_factory(
                ax.transAxes, ax.transData)
            # Skew arrow
            ax.annotate("", xy=(0.5, 0), xytext=(0.5, direc), xycoords=trans,
                        arrowprops=dict(arrowstyle="<-", facecolor='black'),
                        label="Hello"
                        )

    def plot_events(self, ax):
        if self.__binarized_fast is False:
            raise ValueError("No binarized data!")

        bin_fast = self.__binarized_fast[self.__good_cells]
        raster = [[] for i in range(len(bin_fast))]
        sampling = self.__settings["Sampling [Hz]"]

        for i in range(len(bin_fast)):
            for j in range(len(bin_fast[0])):
                yield (i*len(bin_fast[0])+j+1)/len(bin_fast)/len(bin_fast[0])
                if bin_fast[i, j] == 1:
                    raster[i].append(j/sampling)

        ax.eventplot(raster, linewidths=0.1)
