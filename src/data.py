import numpy as np

from scipy.signal import butter, sosfiltfilt, ricker, cwt, correlate, savgol_filter
from scipy.optimize import curve_fit, differential_evolution

import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter

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
        }
    }

class Data(object):
    """
    A class for signal analysis.
    """
# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self):
        self.__signal = False
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
            raise ValueError("Signal shape not 2x2.")
        self.__signal = signal[:,1:].transpose()
        if self.__settings is False:
            self.__settings = SAMPLE_SETTINGS
        sampling = self.__settings["Sampling [Hz]"]
        self.__time = np.arange(len(self.__signal[0]))*(1/sampling)

        self.__points = len(self.__time)
        self.__cells = len(self.__signal)

        self.__good_cells = np.ones(self.__cells, dtype="bool")

    def import_settings(self, settings=SAMPLE_SETTINGS):
        if not "Sampling [Hz]" in settings and not "Stimulation [frame]" in settings and not "Filter" in settings and not "Exclude" in settings:
            raise ValueError("Bad keys in settings.")
        if not "Slow [Hz]" in settings["Filter"] and not "Fast [Hz]" in settings["Filter"] and not "Plot [s]" in settings["Filter"]:
            raise ValueError("Bad keys in settings[filter].")
        if not "Score threshold" in settings["Exclude"] and not "Spikes threshold" in settings["Exclude"]:
            raise ValueError("Bad keys in settings[exclude].")
        self.__settings = settings

    def import_excluded(self, cells):
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
    def get_points(self): return self.__points
    def get_cells(self): return self.__cells
    def get_filtered_slow(self): return self.__filtered_slow
    def get_filtered_fast(self): return self.__filtered_fast
    def get_distributions(self): return self.__distributions
    def get_binarized_slow(self): return self.__binarized_slow
    def get_binarized_fast(self): return self.__binarized_fast
    def get_activity(self): return self.__activity
    def get_good_cells(self): return self.__good_cells

# ----------------------------- ANALYSIS METHODS ----------------------------- #
    def plot(self, i):
        if self.__signal is False:
            raise ValueError("No imported data!")
        if i not in range(self.__cells):
            raise ValueError("Cell index not in range.")
        mean = np.mean(self.__signal[i])

        fig, ax = plt.subplots()
        ax.plot(self.__time, self.__signal[i]-mean, linewidth=0.5, color='dimgrey')
        ax.set_xlim(self.__settings["Filter"]["Plot [s]"])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")

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
        mean = np.mean(self.__signal[i])

        fig, (ax1, ax2) = plt.subplots(2)
        if not self.__good_cells[i]:
            ax1.set_facecolor(EXCLUDE_COLOR)
        fig.suptitle("Filtered data ({0})".format("keep" if self.__good_cells[i] else "exclude"))

        ax1.plot(self.__time, self.__signal[i]-mean, linewidth=0.5, color='dimgrey', alpha=0.5)
        ax1.plot(self.__time, self.__filtered_fast[i], linewidth=0.5, color='red')
        ax1.set_xlim(self.__settings["Filter"]["Plot [s]"])
        ax1.set_ylabel("Amplitude (fast)")

        ax2.plot(self.__time, self.__signal[i]-mean, linewidth=0.5, color='dimgrey', alpha=0.5)
        ax2.plot(self.__time, self.__filtered_slow[i], linewidth=2, color='blue')
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Amplitude (slow)")

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
        if self.__good_cells[i] == False:
            ax1.set_facecolor(EXCLUDE_COLOR)
            fig.suptitle("Distribution (score = {0:.2f})".format(score), x=0.01, y=.98, ha="left")
        else:
            fig.suptitle("Distribution (score = {0:.2f})".format(score), x=0.01, y=.98, ha="left")

        mean = np.mean(self.__signal[i])
        ax1.plot(self.__time, self.__signal[i]-mean,linewidth=0.5,color='dimgrey', zorder=0, alpha=0.5)
        ax1.plot(self.__time, self.__filtered_fast[i],linewidth=0.5,color='red', zorder=2)
        ax1.plot(self.__time, [p_root for i in self.__time], color="k", zorder=2)
        if q_root is not np.inf:
            ax1.fill_between(self.__time, -q_root, q_root, color=EXCLUDE_COLOR, zorder=1, alpha=0.5)

        ax1.xaxis.tick_top()
        ax1.set_xlabel("Time [s]")
        ax1.xaxis.set_label_position('top')
        ax1.set_ylabel("Amplitude")

        ax2.bar(x, hist, max(x)/len(x)*0.8, log=True, color="dimgrey", alpha=0.5)
        ax2.plot(x, np.exp(p(x)), color="red")
        ax2.plot(p_root, np.exp(p(p_root)), marker="o", color="k")
        if q_root != np.inf:
            ax2.plot(np.linspace(0,q_root,10), np.exp(q(np.linspace(0,q_root,10))), color=EXCLUDE_COLOR)
            ax2.plot(q_root, np.exp(q(q_root)), marker="o", color=EXCLUDE_COLOR)
        ax2.set_xlabel("Signal height h")
        ax2.set_ylabel("Data points N")

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
            self.__binarized_slow[cell] = np.heaviside(np.gradient(signal), 0)
            extremes = []
            for i in range(1, self.__points):
                if self.__binarized_slow[cell,i]!=self.__binarized_slow[cell,i-1]:
                    extremes.append(i)

            up = self.__binarized_slow[0,cell]
            counter = 0
            for e in extremes:
                interval = e-counter
                for phi in range(6):
                    lower = counter+int(np.floor(interval/6*phi))
                    higher = counter+int(np.floor(interval/6*(phi+1)))
                    add = 1 if up else 7
                    self.__binarized_slow[cell,lower:higher] = phi+add
                up = (up+1)%2
                counter = e

            # Erase values (from 0 to first extreme) and (last extreme to end)
            self.__binarized_slow[cell, 0:extremes[0]] = 0
            self.__binarized_slow[cell, extremes[-1]:] = 0
            self.__binarized_slow = self.__binarized_slow.astype(int)

    def autolimit(self):
        if self.__binarized_fast is False:
            raise ValueError("No binarized data.")
        print("Computing activity...")
        self.__activity = []
        for cell in range(self.__cells):
            data = self.__binarized_fast[cell]
            box = lambda t, a, t_start, t_end: a*(np.heaviside(t-t_start, 0)-np.heaviside(t-t_end, 0))
            t_half = self.__time[-1]/2
            res = differential_evolution(lambda p: np.sum((box(self.__time, *p) - data)**2),  # quadratic cost function
                                 # [[0, 100], [0, t_half], [t_half, 2*t_half]])  # parameter bounds
                                 [[0, 100], [0, 2*t_half], [0, 2*t_half]])  # parameter bounds
            self.__activity.append(res.x[1:])

            if self.__activity[cell][0] < self.__settings["Stimulation [frame]"][0]/self.__settings["Sampling [Hz]"]:
                self.__good_cells[cell] = False


    def plot_binarized(self, cell):
        if self.__binarized_slow is False or self.__binarized_fast is False:
            raise ValueError("No binarized data!")

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        if not self.__good_cells[cell]:
            ax1.set_facecolor(EXCLUDE_COLOR)
        fig.suptitle("Binarized data ({0})".format("keep" if self.__good_cells[cell] else "exclude"))

        mean = np.mean(self.__signal[cell])
        max_signal = max(abs(self.__signal[cell]-mean))
        norm_signal = (self.__signal[cell]-mean)/max_signal
        max_fast = max(abs(self.__filtered_fast[cell]))

        filtered_fast = self.__filtered_fast[cell]
        threshold = self.__distributions[cell]["p_root"]
        ax1.plot(self.__time, norm_signal*max_fast, linewidth=0.5, color='dimgrey', alpha=0.5)
        ax1.plot(self.__time, filtered_fast, linewidth=0.5, color='red')
        ax1.plot(self.__time, self.__binarized_fast[cell]*threshold, linewidth=0.75, color='black')
        ax1.axvline(self.__settings["Stimulation [frame]"][0]/self.__settings["Sampling [Hz]"], c="grey")
        ax1.axvline(self.__settings["Stimulation [frame]"][1]/self.__settings["Sampling [Hz]"], c="grey")
        ax1.set_ylabel("Amplitude")

        if self.__activity is not False:
            border = self.__activity[cell]
            ax1.axvspan(0, border[0], alpha=0.5, color=EXCLUDE_COLOR)
            ax1.axvspan(border[1], self.__time[-1], alpha=0.5, color=EXCLUDE_COLOR)

        ax3 = ax1.twinx()
        ax3.set_ylabel("Action potentials")

        filtered_slow = self.__filtered_slow[cell]
        ax2.plot(self.__time, 12/2*(norm_signal+1), linewidth=0.5, color='dimgrey', alpha=0.5)
        ax2.plot(self.__time, (filtered_slow/max(abs(filtered_slow))*6)+6, color='blue')
        ax2.plot(self.__time, self.__binarized_slow[cell], color='black')
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Amplitude")

        ax4 = ax2.twinx()
        ax4.set_ylabel("Phase")

        return fig

    def save_plots(self, type, directory):
        if type not in ("imported", "filtered", "distributions", "binarized"):
            raise ValueError("Unknown 'type' argument.")
        for i in range(self.__cells):
            if type == "imported":
                fig = self.plot(i)
            elif type == "filtered":
                fig = self.plot_filtered(i)
            elif type == "distributions":
                fig = self.plot_distributions(i)
            elif type == "binarized":
                fig = self.plot_binarized(i)
            plt.savefig("{0}/{1}.pdf".format(directory, i), dpi=200, bbox_inches='tight')
            plt.close()

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
