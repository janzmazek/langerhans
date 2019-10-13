import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import colorConverter
from scipy.signal import butter, sosfiltfilt, ricker, cwt, correlate, savgol_filter
from scipy import stats
import networkx as nx
import yaml

class Analysis(object):
    """docstring for Analysis."""
# ------------------------------- INITIALIZER -------------------------------- #
    def __init__(self, data_name, sampling_name, positions_name, settings):
        with open(settings, 'r') as stream:
            try:
                self.settings = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise(ValueError("Could not open settings file."))

        data = np.loadtxt(data_name)
        self.__time = data[:-1,0]
        self.__signal = data[:-1,1:]
        # self.__signal = np.loadtxt(data_name)
        self.__sampling=int(np.loadtxt(sampling_name))
        # self.__time = np.arange(0,self.__signal.shape[0])*1/self.__sampling
        self.__positions = np.loadtxt(positions_name)

        self.__data_points = len(self.__time)
        self.__number_of_cells = len(self.__signal[0])

        self.__filtered_slow = False
        self.__filtered_fast = False

        self.__binarized_slow = False
        self.__binarized_fast = False

        self.__excluded = False

        self.__G_slow = False
        self.__G_fast = False

# -------------------------------- IMPORTERS --------------------------------- #
    def import_filtered_slow(self, path):
        self.__filtered_slow = np.loadtxt(path)

    def import_filtered_fast(self, path):
        self.__filtered_fast = np.loadtxt(path)

    def import_binarized_slow(self, path):
        self.__binarized_slow = np.loadtxt(path).astype(int)

    def import_binarized_fast(self, path):
        self.__binarized_fast = np.loadtxt(path).astype(int)

    def import_excluded(self, path):
        self.__excluded = np.loadtxt(path, dtype=int)

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

# ----------------------------- ANALYSIS METHODS ----------------------------- #
# ---------- Filter + smooth ---------- #
    def filter(self):
        slow, fast = self.settings["filter"]["slow"], self.settings["filter"]["fast"]
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

    def plot_filtered(self, directory):
        if self.__filtered_slow is False or self.__filtered_fast is False:
            raise ValueError("No filtered data!")
        for i in range(self.__number_of_cells):
            print(i)
            mean = np.mean(self.__signal[:,i])

            plt.figure(figsize=(12.5,5.0))
            plt.subplot(211)
            plt.plot(self.__time,self.__signal[:,i]-mean,linewidth=0.5,color='dimgrey')
            plt.plot(self.__time,self.__filtered_slow[:,i],linewidth=2,color='blue')

            plt.subplot(212)
            plt.plot(self.__time,self.__signal[:,i]-mean,linewidth=0.5,color='dimgrey')
            plt.plot(self.__time,self.__filtered_fast[:,i],linewidth=0.5,color='red')

            plt.savefig("{0}/{1}.png".format(directory, i), dpi=200, bbox_inches='tight')
            plt.close()

    def smooth_fast(self):
        points = self.settings["smooth"]["points"]
        order = self.settings["smooth"]["order"]
        if self.__filtered_fast is False:
            raise ValueError("No filtered data")
        self.__smoothed_fast = np.zeros((self.__data_points, self.__number_of_cells))
        for i in range(self.__number_of_cells):
            self.__filtered_fast[:,i] = savgol_filter(self.__filtered_fast[:,i], points, order)

# ---------- Binarize ---------- #


# ---------- Analysis ---------- #
