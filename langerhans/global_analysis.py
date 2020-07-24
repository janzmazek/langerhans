import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.colors as mc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

import pickle
from pathlib import Path
import colorsys

from .analysis import Analysis

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

LOW_LIST = ["white", "C0", lighten_color("C0", 1.5)]
HIGH_LIST = ["white", "C3", lighten_color("C3", 1.5)]
LOW_CMAP = mc.LinearSegmentedColormap.from_list("", LOW_LIST)
HIGH_CMAP = mc.LinearSegmentedColormap.from_list("", HIGH_LIST)
DARK_BLUE = lighten_color("C0", 1.3)
LIGHT_BLUE = lighten_color("C0", 0.8)
DARK_RED = lighten_color("C3", 1.3)
LIGHT_RED = lighten_color("C3", 0.8)

class GlobalAnalysis(object):
    """docstring for GlobalAnalysis."""

    def __init__(self, path):
        data_path = Path(path)
        if not data_path.exists() or not data_path.is_dir():
            raise ValueError("Directory does not exist.")

        pkl_file_list = [x for x in data_path.glob("*.pkl")]
        pickle_file_list = [x for x in data_path.glob("*.pickle")]
        data_list = pkl_file_list + pickle_file_list
        if len(data_list) == 0:
            raise ValueError("No pickle files found.")

        positions_list = []
        for s in data_list:
            positions = data_path / Path("{0}.txt".format(s.stem))
            if not positions.exists():
                raise ValueError("Positions file for {0} not found.".format(s.stem))
            positions_list.append(positions)

        self.__data_dict = {}
        self.__low_glucose = []
        self.__high_glucose = []
        self.__pars_network = {}
        self.__pars_cell = {}
        self.__spikes_v_phases = {}
        self.__spikes_v_phases_sep = {}
        self.__corr_v_dist = {}
        self.__networks = {}

        for d, p in zip(data_list, positions_list):
            s = d.stem
            self.__data_dict[s] = d
            with d.open("rb") as f:
                data = pickle.load(f)
            with p.open() as f:
                positions = np.loadtxt(f)

            glc = data.get_settings()["Glucose [mM]"]
            if glc == 8:
                self.__low_glucose.append(s)
            elif glc == 12:
                self.__high_glucose.append(s)

            analysis = Analysis()
            analysis.import_data(data, positions)
            analysis.build_networks()
            network = analysis.get_networks()

            self.__pars_cell[s], self.__pars_network[s] = analysis.compute_parameters()
            self.__spikes_v_phases[s] = analysis.spikes_vs_phase()
            self.__spikes_v_phases_sep[s] = analysis.spikes_vs_phase(mode="separate")
            self.__corr_v_dist[s] = analysis.correlation_vs_distance()
            self.__networks[s] = (network.get_G_slow(), network.get_G_fast())

            del data
            del analysis

    def get_pars_network(self): return self.__pars_network
    def get_pars_cell(self): return self.__pars_cell

    def get_data(self, series):
        path = self.__data_dict[series]
        with path.open("rb") as f:
            data = pickle.load(f)
        return data

    def mean_std_local(self, parameter):
        means_low = [np.nanmean([self.__pars_cell[s][c][parameter] for c in range(len(self.__pars_cell[s]))]) for s in self.__low_glucose]
        means_high = [np.nanmean([self.__pars_cell[s][c][parameter] for c in range(len(self.__pars_cell[s]))]) for s in self.__high_glucose]

        mean_low = np.mean(means_low)
        mean_high = np.mean(means_high)
        std_low = np.std(means_low)
        std_high = np.std(means_high)

        return ((means_low, means_high),(mean_low, mean_high),(std_low, std_high))

    def mean_std_global(self, parameter):
        values_low = [self.__pars_network[s][parameter] for s in self.__low_glucose]
        values_high = [self.__pars_network[s][parameter] for s in self.__high_glucose]
        mean_low = np.mean(values_low)
        mean_high = np.mean(values_high)
        std_low = np.std(values_low)
        std_high = np.std(values_high)

        return ((values_low, values_high), (mean_low, mean_high), (std_low, std_high))

    def plot_avg_stds_local(self, ax, parameter):
        values, means, stds = self.mean_std_local(parameter)

        g_low = [1 for i in range(len(values[0]))]
        g_high = [2 for i in range(len(values[1]))]

        ax.bar([1,2], means, 0.5, yerr=stds, color=(DARK_BLUE, DARK_RED))
        ax.scatter(g_low+g_high, values[0]+values[1], c="k", zorder=10, s=5)
        ax.set_xticks([1,2])
        ax.set_xticklabels(("8 mM", "12 mM"))
        ax.set_xlabel("Glucose concentration (mM)")
        ax.set_ylabel(parameter)

    def plot_avg_stds_global(self, ax, parameter, glucose=False):
        if glucose is not False:
            values0, means0, stds0 = self.mean_std_global(parameter[0])
            values1, means1, stds1 = self.mean_std_global(parameter[1])
            if glucose == 8:
                values = (values0[0], values1[0])
                means = (means0[0], means1[0])
                stds = (stds0[0], stds1[0])
                color = (LIGHT_BLUE, DARK_BLUE)
            elif glucose == 12:
                values = (values0[1], values1[1])
                means = (means0[1], means1[1])
                stds = (stds0[1], stds1[1])
                color = (LIGHT_RED, DARK_RED)
            ax.set_xticklabels(parameter)
            ax.set_xlabel("Parameter")

        else:
            values, means, stds = self.mean_std_global(parameter) # Low/high glucose values
            color = (DARK_BLUE, DARK_RED)
            ax.set_xticklabels(("8 mM", "12 mM"))
            ax.set_xlabel("Glucose concentration (mM)")

        x_left = [1 for i in range(len(values[0]))]
        x_right = [2 for i in range(len(values[1]))]

        ax.bar([1,2], means, 0.5, yerr=stds, color=color)
        ax.scatter(x_left+x_right, values[0]+values[1], c="k", zorder=10, s=5)
        ax.set_xticks([1,2])

    def plot_spikes_vs_phases(self, ax, series):
        if series == "low":
            phases = self.__spikes_v_phases[self.__low_glucose[0]][0]
            spikes = [self.__spikes_v_phases[s][1] for s in self.__low_glucose]
            cmap = LOW_CMAP
            spikes = np.sum(spikes, axis=0)
        elif series == "high":
            phases = self.__spikes_v_phases[self.__high_glucose[0]][0]
            spikes = [self.__spikes_v_phases[s][1] for s in self.__high_glucose]
            cmap = HIGH_CMAP
            spikes = np.sum(spikes, axis=0)
        elif series == "both":
            phases = self.__spikes_v_phases[self.__low_glucose[0]][0]
            spikes = [self.__spikes_v_phases[s][1] for s in self.__low_glucose+self.__high_glucose]
            cmap = HIGH_CMAP
            spikes = np.sum(spikes, axis=0)
        elif series in self.__low_glucose or series in self.__high_glucose:
            phases, spikes = self.__spikes_v_phases[series]
            cmap = LOW_CMAP if series in self.__low_glucose else HIGH_CMAP
        norm_spikes = spikes/np.max(spikes)
        colors = cmap(norm_spikes)
        ax.bar(phases, norm_spikes, width=2*np.pi/12, bottom=0.0, color=colors)
        ax.set_thetagrids(angles=range(0,360,30))

    def plot_corr_vs_dist(self, ax, bin_number=15, max_distance=210, mode="both"):
        distances_left = []
        distances_right = []
        correlations_left = []
        correlations_right = []
        if mode in ("low", "high", "both"):
            labels = ("slow", "fast")
            if mode == "low":
                series = self.__low_glucose
                color = (LIGHT_BLUE, DARK_BLUE)
            elif mode == "high":
                series = self.__high_glucose
                color = (LIGHT_RED, DARK_RED)
            elif mode == "both":
                series = self.__low_glucose + self.__high_glucose
                color = (LIGHT_RED, DARK_RED)
            for s in series:
                distances_left.extend(self.__corr_v_dist[s][0])
                distances_right.extend(self.__corr_v_dist[s][0])
                correlations_left.extend(self.__corr_v_dist[s][1])
                correlations_right.extend(self.__corr_v_dist[s][2])
        elif mode in ("slow", "fast"):
            labels = ("low", "high")
            if mode == "slow":
                color = (LIGHT_BLUE, LIGHT_RED)
                for s in self.__low_glucose:
                    distances_left.extend(self.__corr_v_dist[s][0])
                    correlations_left.extend(self.__corr_v_dist[s][1])
                for s in self.__high_glucose:
                    distances_right.extend(self.__corr_v_dist[s][0])
                    correlations_right.extend(self.__corr_v_dist[s][1])
            elif mode == "fast":
                color = (DARK_BLUE, DARK_RED)
                for s in self.__low_glucose:
                    distances_left.extend(self.__corr_v_dist[s][0])
                    correlations_left.extend(self.__corr_v_dist[s][2])
                for s in self.__high_glucose:
                    distances_right.extend(self.__corr_v_dist[s][0])
                    correlations_right.extend(self.__corr_v_dist[s][2])


        bins_left = np.linspace(0, max_distance, bin_number)
        inds_left = np.digitize(distances_left, bins_left)

        bins_right = np.linspace(0, max_distance, bin_number)
        inds_right = np.digitize(distances_right, bins_right)

        left_mean = np.mean(correlations_left)
        right_mean = np.mean(correlations_right)

        left_binarized = [[] for b in range(bin_number)]
        right_binarized = [[] for b in range(bin_number)]


        for i in range(len(inds_left)):
            left_binarized[inds_left[i]-1].append(correlations_left[i])
        for i in range(len(inds_right)):
            right_binarized[inds_right[i]-1].append(correlations_right[i])

        left = []
        right = []
        for i in range(bin_number):
            left.append(np.mean(left_binarized[i]))
            right.append(np.mean(right_binarized[i]))

        x = np.arange(bin_number)
        width = 0.35  # the width of the bars

        rects1 = ax.bar(x[:-1] - 0.5 - width/2, left[:-1], width, color=color[0], label=labels[0])
        rects2 = ax.bar(x[:-1] - 0.5 + width/2, right[:-1], width, color=color[1], label=labels[1])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Average correlation coefficient')
        ax.set_xlabel("Distance [μm]")
        ax.set_xticks(x)
        ax.set_xticklabels(np.around(bins_left[1:],0).astype(int), rotation=45)
        ax.set_ylim(0.2,0.9)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

        # For the minor ticks, use no labels; default NullFormatter.
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.legend()
