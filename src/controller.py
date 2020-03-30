import os
import numpy as np
import yaml
import pickle
import matplotlib.pyplot as plt

class Controller(object):
    """docstring for Controller."""

    def __init__(self, data, view):
        self.data = data
        self.view = view
        self.analysis = False

        self.view.register(self)

        self.current_number = 0
        self.current_stage = 0

# ---------------------------- Menu click methods ---------------------------- #
    def import_data(self):
        filename = self.view.open_file()
        if filename is None:
            return
        try:
            series = np.loadtxt(filename)[:-1,:]
            self.data.import_data(series)
            self.current_stage = "imported"
            self.draw_fig()
        except ValueError as e:
            print(e)

    def import_settings(self):
        filename = self.view.open_file()
        if filename is None:
            return
        with open(filename, 'r') as stream:
            try:
                settings = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise(ValueError("Could not open settings file."))
        try:
            self.data.import_settings(settings)
            self.data.reset_computations()
            if self.current_stage is not 0:
                self.current_stage = "imported"
            self.draw_fig()
        except ValueError as e:
            print(e)

    def import_excluded(self):
        if self.current_stage == 0:
            return
        filename = self.view.open_file()
        if filename is None:
            return
        try:
            good_cells = np.loadtxt(filename, dtype=bool)
            self.data.import_excluded(good_cells)
            self.draw_fig()
        except ValueError as e:
            print(e)

    def import_object(self):
        filename = self.view.open_file()
        if filename is None:
            return
        try:
            with open(filename, 'rb') as input:
                self.data = pickle.load(input)
        except:
            print("Unsuccessful.")
        self.current_stage = "imported"
        self.draw_fig()

    def edit_settings(self):
        if self.current_stage == 0:
            return
        settings = self.data.get_settings()
        self.view.open_settings_window(settings)

    def save_settings(self):
        if self.data.get_settings() is False:
            return
        filename = self.view.save_as("yaml")
        if filename is None:
            return
        with open(filename, "w") as outfile:
            yaml.dump(self.data.get_settings(), outfile, default_flow_style=False)

    def save_image(self):
        if self.current_stage == 0:
            return
        file = self.view.save_as("pdf")
        if file is None:
            return
        fig = self.__get_fig()
        plt.savefig(file)

    def save_images(self):
        if self.current_stage == 0:
            return
        directory = self.view.open_directory()
        if directory is None:
            return
        for cell in range(self.data.get_cells()):
            fig = self.data.save_plots(self.current_stage, directory)

    def save_excluded(self):
        if self.current_stage == 0:
            return
        if self.data.get_excluded() is False:
            return
        filename = self.view.save_as("dat")
        if filename is None:
            return
        np.savetxt(filename, self.data.get_good_cells(), fmt="%i")

    def save_object(self):
        if not self.data.is_analyzed():
            return
        filename = self.view.save_as("pkl")
        if filename is None:
            return
        with open(filename, "wb") as output:
            pickle.dump(self.data, output, pickle.HIGHEST_PROTOCOL)

# --------------------------- Button click methods --------------------------- #

    def filter_click(self):
        if self.current_stage == 0:
            return
        if self.data.get_filtered_slow() is not False or self.data.get_filtered_fast() is not False:
            self.current_stage = "filtered"
            self.draw_fig()
        else:
            try:
                self.data.filter()
                self.current_stage = "filtered"
                self.draw_fig()
            except ValueError as e:
                print(e)

    def distributions_click(self):
        if self.current_stage == 0:
            return
        elif self.data.get_distributions() is not False:
            self.current_stage = "distributions"
            self.draw_fig()
        else:
            try:
                self.data.compute_distributions()
                self.current_stage = "distributions"
                self.draw_fig()
            except ValueError as e:
                print(e)

    def binarize_click(self):
        if self.current_stage == 0:
            return
        if self.data.get_binarized_slow() is not False or self.data.get_binarized_fast() is not False:
            self.current_stage = "binarized"
            self.draw_fig()
        else:
            try:
                self.data.binarize_fast()
                self.data.binarize_slow()
                self.current_stage = "binarized"
                self.draw_fig()
            except ValueError as e:
                print(e)

    def previous_click(self):
        if self.current_stage == 0:
            return
        if self.current_number > 0:
            self.current_number -= 1
            self.view.cell_number_text.config(text=self.current_number)
            self.draw_fig()

    def next_click(self):
        if self.current_stage == 0:
            return
        if self.current_number < self.data.get_cells()-1:
            self.current_number += 1
            self.view.cell_number_text.config(text=self.current_number)
            self.draw_fig()

    def exclude_click(self):
        if self.current_stage == 0:
            return
        try:
            self.data.exclude(self.current_number)
        except ValueError as e:
            print(e)
        self.draw_fig()

    def unexclude_click(self):
        if self.current_stage == 0:
            return
        try:
            self.data.unexclude(self.current_number)
        except ValueError as e:
            print(e)
        self.draw_fig()

    def autoexclude_click(self):
        if self.current_stage == 0:
            return
        try:
            self.data.autoexclude()
        except ValueError as e:
            print(e)
        self.draw_fig()

    def autolimit_click(self):
        if self.current_stage == 0:
            return
        if self.data.get_activity() is not False:
            return
        try:
            self.data.autolimit()
        except ValueError as e:
            print(e)
        self.draw_fig()

    def __get_fig(self):
        if self.current_stage == "imported":
            return self.data.plot(self.current_number)
        elif self.current_stage == "filtered":
            return self.data.plot_filtered(self.current_number)
        elif self.current_stage == "distributions":
            return self.data.plot_distributions(self.current_number)
        elif self.current_stage == "binarized":
            return self.data.plot_binarized(self.current_number)

    def draw_fig(self):
        if self.current_stage == 0:
            return
        plt.close()
        self.view.draw_fig(self.__get_fig())

    def apply_parameters_click(self):
        new_settings = self.__get_values(self.view.entries)
        self.data.import_settings(new_settings)
        self.data.reset_computations()
        self.current_stage = "imported"
        self.draw_fig()

    def __get_values(self, parameter):
        if type(parameter) not in (dict, list):
            return float(parameter.get())
        elif type(parameter) is dict:
            dictionary = {}
            for key in parameter:
                dictionary[key] = self.__get_values(parameter[key])
            return dictionary
        elif type(parameter) is list:
            array = []
            for key in parameter:
                array.append(self.__get_values(key))
            return array
