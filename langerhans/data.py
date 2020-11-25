import numpy as np

from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew, skewnorm, norm
from scipy.optimize import curve_fit, differential_evolution

import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import matplotlib.transforms as transforms
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

EXCLUDE_COLOR = 'xkcd:salmon'
SAMPLE_SETTINGS = {
    "Glucose [mM]": 8,
    "Sampling [Hz]": 10,
    "Stimulation [frame]": [1200, 0],
    "Filter":
        {
        "Slow [Hz]": [0.001, 0.005],
        "Fast [Hz]": [0.04, 0.4],
        "Plot [s]": [250, 1750]
        },
    "Distribution order": 5,
    "Exclude":
        {
        "Score threshold": 1,
        "Spikes threshold": 0.01
        },
    "Distance [um]": 1
    }


class Data(object):
    """
    A class for signal analysis.
    """
# ------------------------------- INITIALIZER ---------------------------------
    def __init__(self):
        self.__signal = False
        self.__mean_islet = False
        self.__time = False
        self.__settings = False

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
    def import_data(self, signal):
        if not len(signal.shape) == 2:
            raise ValueError("Signal shape not 2D.")
        self.__signal = np.around(signal[:, 1:].transpose(), decimals=3)
        self.__mean_islet = np.mean(self.__signal, 0)  # average over 0 axis
        self.__mean_islet = self.__mean_islet - np.mean(self.__mean_islet)
        if self.__settings is False:
            self.__settings = SAMPLE_SETTINGS
        sampling = self.__settings["Sampling [Hz]"]
        self.__time = np.arange(len(self.__signal[0]))*(1/sampling)

        self.__points = len(self.__time)
        self.__cells = len(self.__signal)

        self.__good_cells = np.ones(self.__cells, dtype="bool")

    def import_settings(self, settings=SAMPLE_SETTINGS):
        if "Sampling [Hz]" not in settings and \
                "Stimulation [frame]" not in settings and \
                "Filter" not in settings and \
                "Exclude" not in settings and \
                "Distance [um]" not in settings:
            raise ValueError("Bad keys in settings.")
        if "Slow [Hz]" not in settings["Filter"] and \
                "Fast [Hz]" not in settings["Filter"] and \
                "Plot [s]" not in settings["Filter"]:
            raise ValueError("Bad keys in settings[filter].")
        if "Score threshold" not in settings["Exclude"] and \
                "Spikes threshold" not in settings["Exclude"]:
            raise ValueError("Bad keys in settings[exclude].")
        self.__settings = settings

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

    def plot(self, ax, cell,
             plots=("mean", "raw", "slow", "fast"), protocol=True
             ):
        time = self.__time
        sampling = self.__settings["Sampling [Hz]"]
        glucose = self.__settings["Glucose [mM]"]
        TA, TAE = self.__settings["Stimulation [frame]"]
        TA, TAE = TA/sampling, TAE/sampling

        if "mean" in plots:
            signal = self.__mean_islet
            signal = signal/np.max(signal)
            ax.plot(time, signal, "k", alpha=0.25, lw=0.1)
        if "raw" in plots:
            signal = self.__signal[cell]
            signal = signal - np.mean(signal)
            signal = signal/np.max(signal)
            ax.plot(time, signal, "k", alpha=0.5, lw=0.1)
        if "slow" in plots:
            filtered_slow = self.__filtered_slow[cell]
            filtered_slow = filtered_slow/np.max(filtered_slow)
            ax.plot(time, filtered_slow, color="C0", lw=2)
        if "fast" in plots:
            filtered_fast = self.__filtered_fast[cell]
            filtered_fast = filtered_fast/np.max(filtered_fast)
            ax.plot(time, filtered_fast, color="C3", lw=0.2)
        if "bin_slow" in plots:
            binarized_slow = self.__binarized_slow[cell]
            ax2 = ax.twinx()
            ax2.plot(time, binarized_slow, color="k", lw=1)
            ax2.set_ylabel("Phase")
        if "bin_fast" in plots:
            # threshold = self.__distributions[cell]["noise_params"][2]
            filtered_fast = self.__filtered_fast[cell]
            binarized_fast = self.__binarized_fast[cell]/np.max(filtered_fast)
            binarized_fast *= 0.5  # *=threshold
            ax.plot(time, binarized_fast, color="k", lw=1)
            ax2 = ax.twinx()
            ax2.set_ylabel("Action potentials")

        ax.set_xlim(0, self.__time[-1])
        ax.set_ylim(None, 1.1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")

        frame_start = self.__settings["Stimulation [frame]"][0]
        frame_end = self.__settings["Stimulation [frame]"][1]
        sampling = self.__settings["Sampling [Hz]"]
        ax.axvline(frame_start/sampling, c="grey")
        ax.axvline(frame_end/sampling, c="grey")

        if self.__good_cells[cell]:
            if self.__activity is not False:
                border = self.__activity[cell]
                ax.axvspan(0, border[0], alpha=0.25, color="grey")
                ax.axvspan(
                    border[1], self.__time[-1], alpha=0.25, color="grey"
                    )
        else:
            ax.axvspan(0, self.__time[-1], alpha=0.5, color=EXCLUDE_COLOR)

        if protocol and TA != 0 and TAE != 0:
            color = "C0" if glucose == 8 else "C3"
            # tform = transforms.blended_transform_factory(
            # ax.transData, ax.transAxes
            # )

            rectangles = {
                '': patches.Rectangle((0, 1.1), TA, 0.15,
                                      color='grey', alpha=0.5,
                                      transform=ax.transData, clip_on=False
                                      ),
                '{} mM'.format(glucose): patches.Rectangle(
                                      (TA, 1.1), TAE-TA, 0.3,
                                      color=color, alpha=0.8,
                                      transform=ax.transData, clip_on=False
                                      ),
                '6 mM': patches.Rectangle((TAE, 1.1), time[-1]-TAE, 0.15,
                                          color='grey', alpha=0.5,
                                          transform=ax.transData, clip_on=False
                                          )
                }
            for r in rectangles:
                ax.add_artist(rectangles[r])
                rx, ry = rectangles[r].get_xy()
                cx = rx + rectangles[r].get_width()/2.0
                cy = ry + rectangles[r].get_height()/2.0
                ax.annotate(r, (cx, cy), color='k', fontsize=12,
                            ha='center', va='center', xycoords=ax.transData,
                            annotation_clip=False
                            )

# ----------------------------- ANALYSIS METHODS ------------------------------

    def plot_raw(self, i):
        if self.__signal is False:
            raise ValueError("No imported data!")
        if i not in range(self.__cells):
            raise ValueError("Cell index not in range.")

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        self.plot(ax1, i, plots=("mean"))
        self.plot(ax2, i, plots=("raw"), protocol=False)

        return fig

# ---------- Filter + smooth ---------- #

    def filter(self):
        if self.__signal is False:
            raise ValueError("No imported data!")
        slow = self.__settings["Filter"]["Slow [Hz]"]
        fast = self.__settings["Filter"]["Fast [Hz]"]
        self.__filtered_slow = np.zeros((self.__cells, self.__points))
        self.__filtered_fast = np.zeros((self.__cells, self.__points))

        for i in range(self.__cells):
            self.__filtered_slow[i] = self.__bandpass(self.__signal[i],
                                                      (*slow)
                                                      )
            self.__filtered_fast[i] = self.__bandpass(self.__signal[i],
                                                      (*fast)
                                                      )

    def __bandpass(self, data, lowcut, highcut, order=5):
        nyq = 0.5*self.__settings["Sampling [Hz]"]
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(
            order, [low, high], analog=False, btype='band', output='sos'
            )
        y = sosfiltfilt(sos, data)
        return y

    def plot_filtered(self, i):
        if self.__filtered_slow is False or self.__filtered_fast is False:
            raise ValueError("No filtered data!")
        if i not in range(self.__cells):
            raise ValueError("Cell index not in range.")

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Filtered data")

        self.plot(ax1, i, plots=("raw, slow"))
        ax1.set_xlabel(None)

        self.plot(ax2, i, plots=("raw, fast"), protocol=False)
        ax2.set_xlim(*self.__settings["Filter"]["Plot [s]"])

        return fig

# ---------- Distributions ---------- #

    def compute_distributions(self):
        if self.__filtered_slow is False:
            raise ValueError("No filtered data.")
        self.__distributions = [dict() for i in range(self.__cells)]

        for cell in range(self.__cells):
            signal = self.__filtered_fast[cell]
            signal /= np.max(np.abs(signal))

            # Define noise from time 0 to start of stimulation
            stimulation = int(self.__settings["Stimulation [frame]"][0])
            noise = signal[:int(stimulation)]
            spikes = signal[int(stimulation):]

            # Distribution parameters of noise
            noise_params = (skew(noise), np.mean(noise), np.std(noise))
            spikes_params = (skew(spikes), np.mean(spikes), np.std(spikes))

            self.__distributions[cell]["noise_params"] = noise_params
            self.__distributions[cell]["spikes_params"] = spikes_params
            self.__distributions[cell]["noise_hist"] = np.histogram(noise, 20)
            self.__distributions[cell]["spikes_hist"] = np.histogram(
                spikes, 100
                )

    def plot_distributions(self, i):
        if self.__distributions is False:
            raise ValueError("No distribution data.")

        noise_params = self.__distributions[i]["noise_params"]
        spikes_params = self.__distributions[i]["spikes_params"]
        noise_h, noise_bins = self.__distributions[i]["noise_hist"]
        spikes_h, spikes_bins = self.__distributions[i]["spikes_hist"]

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax11 = fig.add_subplot(gs[0, 0])
        ax11.set_title("Distribution of noise")
        ax12 = fig.add_subplot(gs[0, 1])
        ax12.set_title("Distribution of active signal")
        ax2 = fig.add_subplot(gs[1, :])

        delta_noise = noise_bins[1]-noise_bins[0]
        ax11.bar(noise_bins[:-1], noise_h, delta_noise, color="grey")

        delta_spikes = spikes_bins[1]-spikes_bins[0]
        ax12.bar(spikes_bins[:-1], spikes_h, delta_spikes, color="grey")

        self.plot(ax2, i, plots=("raw, fast"), protocol=True)
        std = noise_params[2]
        ax2.fill_between(self.__time, -3*std, 3*std,
                         color="C3", alpha=0.25
                         )
        print(spikes_params[0])
        return fig

# ---------- Exclude ---------- #
    def autoexclude(self):
        if self.__distributions is False:
            raise ValueError("No distributions.")
        # Excluding thresholds
        score_threshold = self.__settings["Exclude"]["Score threshold"]

        for cell in range(self.__cells):
            skew = self.__distributions[cell]["spikes_params"][0]
            if skew < score_threshold:
                self.__good_cells[cell] = False
        print("{} of {} good cells".format(
            np.sum(self.__good_cells), self.__cells)
            )

    def exclude(self, cell):
        if cell not in range(self.__cells):
            raise ValueError("Cell not in range.")
        self.__good_cells[cell] = False

    def unexclude(self, cell):
        if cell not in range(self.__cells):
            raise ValueError("Cell not in range.")
        self.__good_cells[cell] = True

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
        self.__binarized_fast = np.zeros((self.__cells, self.__points))
        for cell in range(self.__cells):
            threshold = 3*self.__distributions[cell]["noise_params"][2]
            self.__binarized_fast[cell] = np.where(
                self.__filtered_fast[cell] > threshold, 1, 0
                )
        self.__binarized_fast = self.__binarized_fast.astype(int)

    def binarize_slow(self):
        if self.__filtered_slow is False:
            raise ValueError("No filtered data.")
        self.__binarized_slow = np.zeros((self.__cells, self.__points))
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
        self.__binarized_slow = self.__binarized_slow.astype(int)

    def autolimit(self):
        if self.__binarized_fast is False:
            raise ValueError("No binarized data.")
        print("Computing activity...")
        self.__activity = []
        for cell in range(self.__cells):
            data = self.__binarized_fast[cell]
            cumsum = np.cumsum(data)

            sampling = self.__settings["Sampling [Hz]"]
            stimulation = self.__settings["Stimulation [frame]"][0]
            lower_limit = cumsum[cumsum < 0.1*cumsum[-1]].size
            lower_limit /= sampling
            upper_limit = (cumsum.size - cumsum[cumsum > 0.9*cumsum[-1]].size)
            upper_limit /= sampling

            def box(t, a, t_start, t_end):
                return a*(np.heaviside(t-t_start, 0)-np.heaviside(t-t_end, 0))
            res = differential_evolution(
                lambda p: np.sum((box(self.__time, *p) - data)**2),
                [[0, 100], [0, lower_limit], [upper_limit, self.__time[-1]]]
                )
            self.__activity.append(res.x[1:])

            if self.__activity[cell][0] < stimulation/sampling:
                self.__good_cells[cell] = False
        self.__activity = np.array(self.__activity)

    def plot_binarized(self, cell):
        if self.__binarized_slow is False or self.__binarized_fast is False:
            raise ValueError("No binarized data!")

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.suptitle("Binarized data")

        self.plot(ax1, cell, plots=("slow", "bin_slow"))
        self.plot(ax2, cell, plots=("fast", "bin_fast"), protocol=False)

        return fig

    def plot_events(self):
        if self.__binarized_slow is False or self.__binarized_fast is False:
            raise ValueError("No binarized data!")
        fig, ax = plt.subplots()

        bin_fast = self.__binarized_fast[self.__good_cells]
        raster = [[] for i in range(len(bin_fast))]
        sampling = self.__settings["Sampling [Hz]"]

        for i in range(len(bin_fast)):
            for j in range(len(bin_fast[0])):
                if bin_fast[i, j] == 1:
                    raster[i].append(j/sampling)

        ax.eventplot(raster, linewidths=0.1)
        return fig

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
