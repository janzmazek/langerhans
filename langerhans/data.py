import numpy as np
from copy import deepcopy

from scipy.signal import butter, sosfiltfilt, filtfilt
from scipy.stats import skew
from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import matplotlib.transforms as transforms

EXCLUDE_COLOR = 'xkcd:salmon'
SAMPLE_SETTINGS = {
    "Islet ID": "None",
    "Protocol": 8,
    "Sampling [Hz]": 10,
    "Stimulation [frame]": [1200, 0],
    "Filter [Hz]": [0.002, 0.01, 0.6],
    "Distribution order": 5,
    "Exclude":
        {
        "Score threshold": 1,
        "Spikes threshold": 0.01
        },
    "Distance [um]": 1,
    "Plateau [s]": [0, 0]
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
        
    def copy(self):
        return deepcopy(self)

    def strip(self, frame_start, frame_end):
        signal = self.__signal[:, frame_start:frame_end]
        good_cells = self.__good_cells
        time = self.__time[frame_start:frame_end]
        self.import_data(signal)
        self.__good_cells = good_cells
        self.__time = time
        
        self.__filtered_slow = self.__filtered_slow[:, frame_start:frame_end]
        self.__filtered_fast = self.__filtered_fast[:, frame_start:frame_end]
        self.__binarized_slow = self.__binarized_slow[:, frame_start:frame_end]
        self.__binarized_fast = self.__binarized_fast[:, frame_start:frame_end]
        start, end = self.__time[0], self.__time[-1]
        self.__activity = np.clip(self.__activity, a_min=start, a_max=end)

        frames = self.__settings["Stimulation [frame]"]
        protocol = self.__settings["Protocol"]
        while frames and frame_start >= frames[0]:
            frames.pop(0)
            protocol.pop(0)

        while frames and frame_end <= frames[-1]:
            frames.pop(-1)
            protocol.pop(-1)

    def import_settings(self, settings):
        for key in settings:
            if key in SAMPLE_SETTINGS:
                if isinstance(key, dict):
                    for subkey in key:
                        if subkey in SAMPLE_SETTINGS[key]:
                            self.__settings[subkey] = settings[subkey]
                else:
                    self.__settings[key] = settings[key]
        plateau = self.__settings["Plateau [s]"]
        if plateau[0] != 0 or plateau[1] != 0:
            self.__activity = np.array(
                [plateau for cell in range(self.__cells)]
                )

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

    def get_settings(self): return self.__settings
    def get_time(self): return self.__time
    def get_signal(self): return self.__signal
    def get_mean_islet(self): return self.__mean_islet
    def get_points(self): return self.__points
    def get_cells(self): return self.__cells
    def get_filtered_slow(self): return self.__filtered_slow
    def get_filtered_fast(self): return self.__filtered_fast
    def get_distributions(self): return self.__distributions
    def get_binarized_slow(self): return self.__binarized_slow
    def get_binarized_fast(self): return self.__binarized_fast
    def get_activity(self): return self.__activity
    def get_good_cells(self): return self.__good_cells

# ----------------------------- ANALYSIS METHODS ------------------------------
# ---------- Filter + smooth ---------- #

    def filter(self):
        if self.__signal is False:
            raise ValueError("No imported data!")
        lowcut, midcut, highcut = self.__settings["Filter [Hz]"]
        
        self.__filtered_slow = np.zeros((self.__cells, self.__points), dtype=np.float32)
        self.__filtered_fast = np.zeros((self.__cells, self.__points), dtype=np.float32)

        
        for cell in range(self.__cells):
            # Bandpass filter for trend and noise removal
            detrended = self.__bandpass(
                self.__signal[cell], lowcut=lowcut, highcut=highcut, order=4
                )
            detrended -= np.mean(detrended)
            detrended /= np.max(detrended)
            
            # Low-pass filter for slow component
            slow = self.__lowpass(detrended, cutoff=midcut, order=4)
            fast = detrended - slow
            
            self.__signal[cell] = detrended
            self.__filtered_slow[cell] = slow
            self.__filtered_fast[cell] = fast
            yield (cell+1)/self.__cells

    def __bandpass(self, data, lowcut, highcut, order=4):
        nyq = 0.5 * self.__settings["Sampling [Hz]"]
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def __lowpass(self, data, cutoff, order=4):
        nyq = 0.5 * self.__settings["Sampling [Hz]"]
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, data)

# ---------- Distributions ---------- #

    def compute_distributions(self):
        if self.__filtered_slow is False:
            raise ValueError("No filtered data.")
        if not self.__settings["Stimulation [frame]"]:
            raise ValueError("No stimulation frames defined.")
        self.__distributions = [dict() for i in range(self.__cells)]

        for cell in range(self.__cells):
            signal = self.__filtered_fast[cell]
            signal /= np.max(np.abs(signal))

            # Define noise from time 0 to start of stimulation
            start = self.__settings["Stimulation [frame]"][0]
            end = self.__settings["Stimulation [frame]"][-1]
            noise = signal[:int(start)]
            if end != 0:
                spikes = signal[int(start):int(end)]
            else:
                spikes = signal[int(start):]

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
        if self.__distributions is False or self.__filtered_fast is False:
            raise ValueError("No distribution or filtered data.")

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

    def autolimit(self):
        if self.__binarized_fast is False:
            raise ValueError("No binarized data.")
        print("Computing activity...")
        self.__activity = [False for i in range(self.__cells)]
        for cell in range(self.__cells):
            data = self.__binarized_fast[cell]
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
            self.__activity[cell] = res.x[1:]

            if self.__settings["Stimulation [frame]"]:
                stimulation = self.__settings["Stimulation [frame]"][0]
                if self.__activity[cell][0] < stimulation/sampling:
                    self.__good_cells[cell] = False
            yield (cell+1)/self.__cells
        self.__activity = np.array(self.__activity)

    def is_analyzed(self):
        if self.__filtered_slow is False or self.__filtered_fast is False:
            return False
        elif self.__distributions is False:
            return False
        elif self.__binarized_slow is False or self.__binarized_fast is False:
            return False
        elif self.__good_cells is False:
            return False
        elif self.__activity is False:
            return False
        else:
            return True

# ----------------------------- PLOTTING METHODS ------------------------------

    def plot(self, ax, cell, plots=("raw",),
             protocol=True, activity=False, noise=False
             ):
        if "mean" in plots:
            signal = self.__mean_islet
            signal /= np.max(np.abs(signal))
            ax.plot(self.__time, signal, "k", alpha=0.25, lw=0.1)
        if "raw" in plots:
            signal = self.__signal[cell]
            # signal = signal - np.mean(signal)
            # signal /= np.max(np.abs(signal))
            ax.plot(self.__time, signal, "k", alpha=1, lw=0.05)
        if "slow" in plots:
            filtered_slow = self.__filtered_slow[cell]
            # filtered_slow = filtered_slow/np.max(filtered_slow)
            ax.plot(self.__time, filtered_slow, color="C0", lw=2)
        if "fast" in plots:
            filtered_fast = self.__filtered_fast[cell]
            # filtered_fast = filtered_fast/np.max(filtered_fast)
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

        if protocol:
            self.__plot_protocol(ax)

        if activity:
            self.__plot_activity(ax, cell)

        if noise:
            self.__plot_noise(ax, cell)

        ax.set_xlim(self.__time[0], self.__time[-1])
        ax.set_ylim(None, 1.1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")

    def __plot_protocol(self, ax):
        sampling = self.__settings["Sampling [Hz]"]
        protocol = self.__settings["Protocol"]
        stimulation = np.concatenate(
            ([self.__time[0]], 
             np.array(self.__settings["Stimulation [frame]"])/sampling, 
             [self.__time[-1]])
            )

        # Skip if invalid stimulation
        if stimulation[1] == 0 or stimulation[-1] == 0:
            return

        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        height = 0.05  # Small bar height
        y_top = 1.01   # Place just above the top of the plot

        # Use ax.barh for bulk drawing (faster than adding patches individually)
        starts = stimulation[:-1]
        widths = stimulation[1:] - stimulation[:-1]

        for start, width, label in zip(starts, widths, protocol):
            # Draw rectangle
            rect = plt.Rectangle((start, y_top), width, height, transform=trans,
                                 color='grey', alpha=0.5, clip_on=False)
            ax.add_patch(rect)

            # Add centered text
            cx = start + width / 2
            cy = y_top + height / 2
            ax.text(cx, cy, label, ha='center', va='center', fontsize=7,
                    transform=trans, clip_on=False)

        for s in stimulation:
            ax.axvline(s, c="grey")

    def __plot_activity(self, ax, cell):
        if self.__good_cells[cell]:
            if self.__activity is not False:
                if self.__activity[cell] is not False:
                    border = self.__activity[cell]
                    ax.axvspan(0, border[0], alpha=0.25, color="grey")
                    ax.axvspan(
                        border[1], self.__time[-1], alpha=0.25, color="grey"
                        )
        else:
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
        if self.__binarized_slow is False or self.__binarized_fast is False:
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
