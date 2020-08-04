import numpy as np

from scipy.signal import butter, sosfiltfilt, ricker, cwt, correlate, savgol_filter
from scipy.optimize import curve_fit, differential_evolution

import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import matplotlib.transforms as transforms
import matplotlib.patches as patches

EXCLUDE_COLOR = 'xkcd:salmon'
SAMPLE_SETTINGS = {
    "Glucose [mM]": 8,
    "Sampling [Hz]": 10,
    "Stimulation [frame]": [0, 0],
    "Filter":
        {
        "Slow [Hz]": [0.001, 0.005],
        "Fast [Hz]": [0.04, 0.4],
        "Plot [s]": [250, 1750]
        },
    "Distribution order": 5,
    "Exclude":
        {
        "Score threshold": 1.5,
        "Spikes threshold": 0.01
        },
    "Distance [um]": 1
    }

class Data(object):
    """
    A class for signal analysis.
    """
# ------------------------------- INITIALIZER -------------------------------- #
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

# --------------------------------- IMPORTS ---------------------------------- #
    def import_data(self, signal):
        if not len(signal.shape)==2:
            raise ValueError("Signal shape not 2D.")
        self.__signal = np.around(signal[:,1:].transpose(), decimals=3)
        self.__mean_islet = np.mean(self.__signal, 0) # average over 0 axis
        self.__mean_islet = self.__mean_islet - np.mean(self.__mean_islet)
        if self.__settings is False:
            self.__settings = SAMPLE_SETTINGS
        sampling = self.__settings["Sampling [Hz]"]
        self.__time = np.arange(len(self.__signal[0]))*(1/sampling)

        self.__points = len(self.__time)
        self.__cells = len(self.__signal)

        self.__good_cells = np.ones(self.__cells, dtype="bool")

    def import_settings(self, settings=SAMPLE_SETTINGS):
        if not "Sampling [Hz]" in settings and not "Stimulation [frame]" in settings and not "Filter" in settings and not "Exclude" in settings and not "Distance [um]" in settings:
            raise ValueError("Bad keys in settings.")
        if not "Slow [Hz]" in settings["Filter"] and not "Fast [Hz]" in settings["Filter"] and not "Plot [s]" in settings["Filter"]:
            raise ValueError("Bad keys in settings[filter].")
        if not "Score threshold" in settings["Exclude"] and not "Spikes threshold" in settings["Exclude"]:
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


# --------------------------------- GETTERS ---------------------------------- #
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

    def plot(self, ax, cell, plots=("mean", "raw", "slow", "fast"), protocol=True):
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
            threshold = self.__distributions[cell]["p_root"]
            filtered_fast = self.__filtered_fast[cell]
            binarized_slow = self.__binarized_fast[cell]/np.max(filtered_fast)*threshold
            ax.plot(time, binarized_slow, color="k", lw=1)
            ax2 = ax.twinx()
            ax2.set_ylabel("Action potentials")

        ax.set_xlim(0,self.__time[-1])
        ax.set_ylim(None, 1.1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")

        ax.axvline(self.__settings["Stimulation [frame]"][0]/self.__settings["Sampling [Hz]"], c="grey")
        ax.axvline(self.__settings["Stimulation [frame]"][1]/self.__settings["Sampling [Hz]"], c="grey")

        if self.__good_cells[cell]:
            if self.__activity is not False:
                border = self.__activity[cell]
                ax.axvspan(0, border[0], alpha=0.25, color="grey")
                ax.axvspan(border[1], self.__time[-1], alpha=0.25, color="grey")
        else:
            ax.axvspan(0,self.__time[-1], alpha=0.5, color=EXCLUDE_COLOR)

        if protocol and TA!=0 and TAE!=0:
            color = "C0" if glucose == 8 else "C3"
            # tform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            rectangles = {'' : patches.Rectangle((0, 1.1), TA, 0.15, color ='grey', alpha=0.5, transform=ax.transData, clip_on=False),
                          '{} mM'.format(glucose) : patches.Rectangle((TA, 1.1), TAE-TA, 0.3, color=color, alpha=0.8, transform=ax.transData, clip_on=False),
                          '6 mM' : patches.Rectangle((TAE, 1.1), time[-1]-TAE, 0.15, color ='grey', alpha=0.5, transform=ax.transData, clip_on=False)
                         }
            for r in rectangles:
                ax.add_artist(rectangles[r])
                rx, ry = rectangles[r].get_xy()
                cx = rx + rectangles[r].get_width()/2.0
                cy = ry + rectangles[r].get_height()/2.0
                ax.annotate(r, (cx, cy), color='k', fontsize=12, ha='center', va='center', xycoords=ax.transData, annotation_clip=False)

# ----------------------------- ANALYSIS METHODS ----------------------------- #
    def plot_raw(self, i):
        if self.__signal is False:
            raise ValueError("No imported data!")
        if i not in range(self.__cells):
            raise ValueError("Cell index not in range.")
        mean = np.mean(self.__signal[i])

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        self.plot(ax1, i, plots=("mean"))
        self.plot(ax2, i, plots=("raw"), protocol=False)

        return fig
# ---------- Filter + smooth ---------- #
    def filter(self):
        if self.__signal is False:
            raise ValueError("No imported data!")
        slow, fast = self.__settings["Filter"]["Slow [Hz]"], self.__settings["Filter"]["Fast [Hz]"]
        self.__filtered_slow = np.zeros((self.__cells, self.__points))
        self.__filtered_fast = np.zeros((self.__cells, self.__points))

        for i in range(self.__cells):
            self.__filtered_slow[i] = self.__bandpass(self.__signal[i], (*slow))
            self.__filtered_fast[i] = self.__bandpass(self.__signal[i], (*fast))

    def __bandpass(self, data, lowcut, highcut, order=5):
        nyq = 0.5*self.__settings["Sampling [Hz]"]
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
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

# ---------- Binarize ---------- #

    def compute_distributions(self):
        if self.__filtered_slow is False:
            raise ValueError("No filtered data.")
        self.__distributions = [dict() for i in range(self.__cells)]

        for cell in range(self.__cells):
            # Compute cumulative histogram and bins
            signal = np.clip(self.__filtered_fast[cell], 0, None)
            hist = np.histogram(signal, 50)
            cumulative_hist = np.flip(np.cumsum(np.flip(hist[0])))
            bins = hist[1]
            x = (bins[1:] + bins[:-1])/2 # middle points of bins

            # Fit polynomial of nth order
            order = self.__settings["Distribution order"]
            z = np.polyfit(x, np.log(cumulative_hist), order, w=np.sqrt(cumulative_hist))
            p = np.poly1d(z)

            # Transitions of polynomial
            first_derivative = np.polyder(p, 1)
            second_derivative = np.polyder(p, 2)
            roots = np.roots(second_derivative)
            real_roots = roots.real[abs(roots.imag) < 1e-5]
            final_roots = real_roots[np.logical_and(real_roots > 0, real_roots < bins[-1])]
            p_root = min(final_roots)
            min_value = first_derivative(p_root)

            # Quadratic function around 0
            q = np.poly1d([second_derivative(0)/2, first_derivative(0), p(0)])
            q_root = np.roots(np.polyder(q, 1))
            q_root = q_root[0] if q_root>0 or q_root>bins[-1] else np.inf

            # Goodness score
            phi = np.arctan(abs(min_value))/(np.pi/2)
            score = (1-phi)*p_root/q_root

            spikes_threshold = self.__settings["Exclude"]["Spikes threshold"]
            if np.exp(p(p_root))<spikes_threshold*self.__points:
                score = 0

            self.__distributions[cell]["hist"] = cumulative_hist
            self.__distributions[cell]["bins"] = x
            self.__distributions[cell]["p"] = p
            self.__distributions[cell]["q"] = q
            self.__distributions[cell]["p_root"] = p_root
            self.__distributions[cell]["q_root"] = q_root
            self.__distributions[cell]["score"] = score

    def plot_distributions(self, i):
        if self.__distributions is False:
            raise ValueError("No distribution data.")
        hist = self.__distributions[i]["hist"]
        x = self.__distributions[i]["bins"]
        p = self.__distributions[i]["p"]
        q = self.__distributions[i]["q"]
        p_root = self.__distributions[i]["p_root"]
        q_root = self.__distributions[i]["q_root"]
        score = self.__distributions[i]["score"]

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle("Distribution (score = {0:.2f})".format(score), x=0.01, y=.98, ha="left")

        ax1.bar(x, hist, max(x)/len(x)*0.8, log=True, color="dimgrey", alpha=0.5)
        ax1.plot(x, np.exp(p(x)), color="red")
        ax1.plot(p_root, np.exp(p(p_root)), marker="o", color="k")
        if q_root != np.inf:
            ax1.plot(np.linspace(0,q_root,10), np.exp(q(np.linspace(0,q_root,10))), color=EXCLUDE_COLOR)
            ax1.plot(q_root, np.exp(q(q_root)), marker="o", color=EXCLUDE_COLOR)
        ax1.set_xlabel("Signal height h")
        ax1.set_ylabel("Data points N")

        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')

        self.plot(ax2, i, plots=("raw, fast"), protocol=False)
        scaling = np.max(self.__filtered_fast[i])
        ax2.axhline(p_root/scaling, c="k")
        if q_root is not np.inf:
            ax2.fill_between(self.__time, -q_root/scaling, q_root/scaling, color=EXCLUDE_COLOR, zorder=1, alpha=0.5)

        return fig

# ---------- Exclude ---------- #
    def autoexclude(self):
        if self.__distributions is False:
            raise ValueError("No distributions.")
        # Excluding thresholds
        score_threshold = self.__settings["Exclude"]["Score threshold"]

        for cell in range(self.__cells):
            p = self.__distributions[cell]["p"]
            p_root = self.__distributions[cell]["p_root"]
            score = self.__distributions[cell]["score"]
            if score<score_threshold:
                self.__good_cells[cell] = False

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

        # Create a 2D array of sliding indices across the entire length of input array.
        # Match up with the input sequence & get the matching starting indices.
        M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

        # Get the range of those indices as final output
        if M.any() > 0:
            return np.where(M == True)[0]
        else:
            return np.array([], dtype="int") # No match found

    def binarize_fast(self):
        if self.__distributions is False or self.__filtered_fast is False:
            raise ValueError("No distribution or filtered data.")
        self.__binarized_fast = np.zeros((self.__cells, self.__points))
        for cell in range(self.__cells):
            threshold = self.__distributions[cell]["p_root"]
            self.__binarized_fast[cell] = np.where(self.__filtered_fast[cell]>threshold, 1, 0)
            self.__binarized_fast = self.__binarized_fast.astype(int)

    def binarize_slow(self):
        if self.__filtered_slow is False:
            raise ValueError("No filtered data.")
        self.__binarized_slow = np.zeros((self.__cells, self.__points))
        for cell in range(self.__cells):
            signal = self.__filtered_slow[cell]
            heavisided_gradient = np.heaviside(np.gradient(signal), 0)
            minima = self.__search_sequence(heavisided_gradient, [0,1])
            maxima = self.__search_sequence(heavisided_gradient, [1,0])
            extremes = np.sort(np.concatenate((minima, maxima)))

            reverse_mode = False if minima[0] < maxima[0] else True

            self.__binarized_slow[cell, 0:extremes[0]] = 0
            for i in range(len(extremes)-1):
                e1, e2 = extremes[i], extremes[i+1]
                if i%2 == int(reverse_mode):
                    self.__binarized_slow[cell, e1:e2] = np.floor(np.linspace(1,7,e2-e1, endpoint=False))
                else:
                    self.__binarized_slow[cell, e1:e2] = np.floor(np.linspace(7,13,e2-e1, endpoint=False))
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
            lower_limit = cumsum[cumsum<0.1*cumsum[-1]].size/sampling
            upper_limit = (cumsum.size - cumsum[cumsum>0.9*cumsum[-1]].size)/sampling

            box = lambda t, a, t_start, t_end: a*(np.heaviside(t-t_start, 0)-np.heaviside(t-t_end, 0))
            res = differential_evolution(lambda p: np.sum((box(self.__time, *p) - data)**2),  # quadratic cost function
                                 # [[0, 100], [0, t_half], [t_half, 2*t_half]])  # parameter bounds
                                 [[0, 100], [0, lower_limit], [upper_limit, self.__time[-1]]])  # parameter bounds
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
                if bin_fast[i,j] == 1:
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
