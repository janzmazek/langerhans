import numpy as np

from scipy import stats
from scipy.signal import butter, sosfiltfilt, ricker, cwt, correlate, savgol_filter
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import colorConverter

import yaml

class Data(object):
    """docstring for Analysis."""
# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self, data_name, sampling_name, settings):
        with open(settings, 'r') as stream:
            try:
                self.__settings = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise(ValueError("Could not open settings file."))

        data = np.loadtxt(data_name)
        self.__time = data[:-1,0]
        self.__signal = data[:-1,1:]
        # self.__signal = np.loadtxt(data_name)
        self.__sampling=int(np.loadtxt(sampling_name))
        # self.__time = np.arange(0,self.__signal.shape[0])*1/self.__sampling

        self.__data_points = len(self.__time)
        self.__number_of_cells = len(self.__signal[0])

        self.__filtered_slow = False
        self.__filtered_fast = False

        self.__distributions = False

        self.__binarized_slow = False
        self.__binarized_fast = False

# -------------------------------- IMPORTERS --------------------------------- #
    def import_filtered_slow(self, path):
        self.__filtered_slow = np.loadtxt(path)

    def import_filtered_fast(self, path):
        self.__filtered_fast = np.loadtxt(path)

    def import_binarized_slow(self, path):
        self.__binarized_slow = np.loadtxt(path).astype(int)

    def import_binarized_fast(self, path):
        self.__binarized_fast = np.loadtxt(path).astype(int)

# ---------------------------------- SAVERS ---------------------------------- #
    def save_filtered_slow(self, path):
        if self.__filtered_slow is False:
            raise ValueError("No filtered data!")
        np.savetxt(path, self.__filtered_slow)

    def save_filtered_fast(self, path):
        if self.__filtered_fast is False:
            raise ValueError("No filtered data!")
        np.savetxt(path, self.__filtered_fast)

    def save_binarized_slow(self, path):
        np.savetxt(path, self.__binarized_slow, fmt="%d")

    def save_binarized_fast(self, path):
        np.savetxt(path, self.__binarized_fast, fmt="%d")

# --------------------------------- GETTERS ---------------------------------- #
    def get_settings(self): return self.__settings
    def get_time(self): return self.__time
    def get_signal(self): return self.__signal
    def get_data_points(self): return self.__data_points
    def get_number_of_cells(self): return self.__number_of_cells
    def get_filtered_slow(self): return self.__filtered_slow
    def get_filtered_fast(self): return self.__filtered_fast
    def get_distributions(self): return self.__distributions
    def get_binarized_slow(self): return self.__binarized_slow
    def get_binarized_fast(self): return self.__binarized_fast

# ----------------------------- ANALYSIS METHODS ----------------------------- #
# ---------- Filter + smooth ---------- #
    def filter(self):
        slow, fast = self.__settings["filter"]["slow"], self.__settings["filter"]["fast"]
        self.__filtered_slow = np.zeros((self.__data_points, self.__number_of_cells))
        self.__filtered_fast = np.zeros((self.__data_points, self.__number_of_cells))

        for i in range(self.__number_of_cells):
            self.__filtered_slow[:,i] = self.__bandpass(self.__signal[:,i], (*slow))
            self.__filtered_fast[:,i] = self.__bandpass(self.__signal[:,i], (*fast))

    def __bandpass(self, data, lowcut, highcut, order=5):
        nyq = 0.5*self.__sampling
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        y = sosfiltfilt(sos, data)
        return y

    def smooth_fast(self):
        points = self.__settings["smooth"]["points"]
        order = self.__settings["smooth"]["order"]
        if self.__filtered_fast is False:
            raise ValueError("No filtered data")
        self.__smoothed_fast = np.zeros((self.__data_points, self.__number_of_cells))
        for i in range(self.__number_of_cells):
            self.__filtered_fast[:,i] = savgol_filter(self.__filtered_fast[:,i], points, order)

    def plot_filtered(self, directory):
        if self.__filtered_slow is False or self.__filtered_fast is False:
            raise ValueError("No filtered data!")
        for i in range(self.__number_of_cells):
            print(i)
            mean = np.mean(self.__signal[:,i])

            plt.subplot(211)
            plt.plot(self.__time,self.__signal[:,i]-mean,linewidth=0.5,color='dimgrey')
            plt.plot(self.__time,self.__filtered_slow[:,i],linewidth=2,color='blue')

            plt.subplot(212)
            plt.plot(self.__time,self.__signal[:,i]-mean,linewidth=0.5,color='dimgrey')
            plt.plot(self.__time,self.__filtered_fast[:,i],linewidth=0.5,color='red')
            plt.xlim(self.__settings["filter"]["image"])

            plt.savefig("{0}/{1}.pdf".format(directory, i), dpi=200, bbox_inches='tight')
            plt.close()

# ---------- Binarize ---------- #

    def compute_distributions(self):
        if self.__filtered_slow is False:
            raise ValueError("No filtered data.")
        self.__distributions = [dict() for i in range(self.__number_of_cells)]
        for cell in range(self.__number_of_cells):
            # Compute cumulative histogram and bins
            signal = np.clip(self.__filtered_fast[:,cell], 0, None)
            hist = np.histogram(signal, 50)
            cumulative_hist = np.flip(np.cumsum(np.flip(hist[0])))
            bins = hist[1]
            x = (bins[1:] + bins[:-1])/2 # middle points of bins

            # Fit polynomial of nth order
            order = self.__settings["distribution_order"]
            z = np.polyfit(x, np.log(cumulative_hist), order)
            p = np.poly1d(z)

            # Transitions of polynomial
            first_derivative = np.polyder(p, 1)
            second_derivative = np.polyder(p, 2)
            roots = np.roots(second_derivative)
            real_roots = roots.real[abs(roots.imag) < 1e-5]
            final_roots = real_roots[np.logical_and(real_roots > 0, real_roots < bins[-1])]
            min_root = min(final_roots)
            min_value = first_derivative(min_root)

            # Quadratic function around 0
            q = np.poly1d([second_derivative(0)/2, first_derivative(0), p(0)])
            q_root = np.roots(np.polyder(q, 1))
            q_root = q_root[0] if q_root>0 or q_root>bins[-1] else np.inf

            # Objective function (grade)
            phi = np.arctan(abs(min_value))/(np.pi/2)
            objective_function = (1-phi)*min_root/q_root

            # Excluding thresholds
            objective_threshold = self.__settings["exclude"]["objective_threshold"]
            spikes_threshold = self.__settings["exclude"]["spikes_threshold"]

            if objective_function<objective_threshold or np.exp(p(min_root))<spikes_threshold*self.__data_points:
                exclude = True
            else:
                exclude = False

            self.__distributions[cell]["hist"] = cumulative_hist
            self.__distributions[cell]["bins"] = x
            self.__distributions[cell]["p"] = p
            self.__distributions[cell]["q"] = q
            self.__distributions[cell]["p_root"] = min_root
            self.__distributions[cell]["p_value"] = min_value
            self.__distributions[cell]["q_root"] = q_root
            self.__distributions[cell]["exclude"] = exclude
            self.__distributions[cell]["objective_function"] = objective_function

    def plot_distributions(self, directory):
        if self.__distributions is False:
            raise ValueError("No distribution data.")
        for cell in range(self.__number_of_cells):

            hist = self.__distributions[cell]["hist"]
            x = self.__distributions[cell]["bins"]
            p = self.__distributions[cell]["p"]
            q = self.__distributions[cell]["q"]
            p_root = self.__distributions[cell]["p_root"]
            p_value = self.__distributions[cell]["p_value"]
            q_root = self.__distributions[cell]["q_root"]
            exclude = self.__distributions[cell]["exclude"]
            objective_function = self.__distributions[cell]["objective_function"]

            fig, (ax1, ax2) = plt.subplots(2, 1)
            if exclude:
                ax1.set_facecolor('xkcd:salmon')
                fig.suptitle("Score = {0:.2f} ({1})".format(objective_function, "EXCLUDE"))
            else:
                fig.suptitle("Score = {0:.2f} ({1})".format(objective_function, "KEEP"))

            mean = np.mean(self.__signal[:,cell])
            ax1.plot(self.__time,self.__signal[:,cell]-mean,linewidth=0.5,color='dimgrey', zorder=0)
            ax1.plot(self.__time,self.__filtered_fast[:,cell],linewidth=0.5,color='red', zorder=2)
            ax1.plot(self.__time, [p_root for i in self.__time], color="green", zorder=2)
            if q_root is not np.inf:
                ax1.fill_between(self.__time, -q_root, q_root, color='orange', zorder=1)

            ax2.bar(x, hist, max(x)/len(x), log=True)
            ax2.plot(x, np.exp(p(x)), color="k")
            ax2.plot(p_root, np.exp(p(p_root)), marker="o", color="green")
            if q_root != np.inf:
                ax2.plot(np.linspace(0,q_root,10), np.exp(q(np.linspace(0,q_root,10))), color="orange")
                ax2.plot(q_root, np.exp(q(q_root)), marker="o", color="orange")

            plt.savefig("{0}/{1}.pdf".format(directory, cell))
            plt.close()

    def binarize_fast(self):
        if self.__distributions is False or self.__filtered_fast is False:
            raise ValueError("No distribution or filtered data.")
        self.__binarized_fast = np.zeros((self.__data_points, self.__number_of_cells))
        for cell in range(self.__number_of_cells):
            if self.__distributions[cell]["exclude"] is True:
                pass
            threshold = self.__distributions[cell]["p_root"]
            self.__binarized_fast[:,cell] = np.where(self.__filtered_fast[:,cell]>threshold, 1, 0)

    def binarize_slow(self):
        if self.__filtered_slow is False:
            raise ValueError("No filtered data.")
        self.__binarized_slow = np.zeros((self.__data_points, self.__number_of_cells))
        for cell in range(self.__number_of_cells):
            signal = self.__filtered_slow[:,cell]
            self.__binarized_slow[:,cell] = np.heaviside(np.gradient(signal), 0)
            extremes = []
            for i in range(1, self.__data_points):
                if self.__binarized_slow[i,cell]!=self.__binarized_slow[i-1,cell]:
                    extremes.append(i)

            up = self.__binarized_slow[0,cell]
            counter = 0
            for e in extremes:
                interval = e-counter
                for phi in range(6):
                    lower = counter+int(np.floor(interval/6*phi))
                    higher = counter+int(np.floor(interval/6*(phi+1)))
                    add = 1 if up else 7
                    self.__binarized_slow[lower:higher,cell] = phi+add
                up = (up+1)%2
                counter = e

            # Erase values (from 0 to first extreme) and (last extreme to end)
            self.__binarized_slow[0:extremes[0],cell] = 0
            self.__binarized_slow[extremes[-1]:,cell] = 0
            self.__binarized_slow.astype(int)

    def plot_binarized(self, directory):
        if self.__binarized_slow is False or self.__binarized_fast is False:
            raise ValueError("No filtered data!")

        for cell in range(self.__number_of_cells):
            if self.__distributions[cell]["exclude"] is True:
                continue

            fig, (ax1, ax2) = plt.subplots(2, 1)
            mean = np.mean(self.__signal[:,cell])
            filtered_fast = self.__filtered_fast[:,cell]
            threshold = self.__distributions[cell]["p_root"]
            ax1.plot(self.__time, (self.__signal[:,cell]-mean)/2, linewidth=0.5, color='dimgrey', alpha=0.5)
            ax1.plot(self.__time, filtered_fast, linewidth=0.5, color='red')
            ax1.plot(self.__time, self.__binarized_fast[:,cell]*threshold, linewidth=0.75, color='black')

            filtered_slow = self.__filtered_slow[:,cell]
            ax2.plot(self.__time, (self.__signal[:,cell]-mean)/2, linewidth=0.5, color='dimgrey', alpha=0.5)
            ax2.plot(self.__time, (filtered_slow/max(abs(filtered_slow))*6)+6, color='blue')
            ax2.plot(self.__time, self.__binarized_slow[:,cell], color='black')

            plt.savefig("{0}/{1}.pdf".format(directory, cell), dpi=200, bbox_inches='tight')
            plt.close()

            print(cell)
