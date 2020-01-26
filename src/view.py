import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import copy


# Window parameters
WIDTH = 1000
HEIGHT = 600

class View(tk.Tk):
    """docstring for View."""

    def __init__(self, *args, **kwargs):
        super(View, self).__init__(*args, **kwargs)
        self.title("Analysis of Calcium Signals")

        menubar = tk.Menu(self)

        importmenu = tk.Menu(menubar, tearoff=0)
        exportmenu = tk.Menu(menubar, tearoff=0)
        editmenu = tk.Menu(menubar, tearoff=0)

        menubar.add_cascade(label="Import", menu=importmenu)
        importmenu.add_command(label="Import data", command=lambda: self.controller.import_data())
        importmenu.add_command(label="Import settings", command=lambda: self.controller.import_settings())
        importmenu.add_command(label="Import excluded", command=lambda: self.controller.import_excluded())
        menubar.add_cascade(label="Export", menu=exportmenu)
        exportmenu.add_command(label="Export settings", command=lambda: self.controller.save_settings())
        exportmenu.add_command(label="Export image", command=lambda: self.controller.save_image())
        exportmenu.add_command(label="Export all images", command=lambda: self.controller.save_images())
        exportmenu.add_command(label="Export excluded", command=lambda: self.controller.save_excluded())
        exportmenu.add_command(label="Export object (pickle)", command=lambda: self.controller.save_object())
        menubar.add_cascade(label="Edit", menu=editmenu)
        editmenu.add_command(label="Settings", command=lambda: self.controller.edit_settings())

        self.config(menu=menubar)
        self.canvas = False

        self.toolbar = tk.LabelFrame(self, text="Tools", padx=5, pady=5)
        self.toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.NO)

        filter_button = tk.Button(self.toolbar, text="Filter", command=lambda: self.controller.filter_click())
        filter_button.pack(side=tk.LEFT)

        distributions_button = tk.Button(self.toolbar, text="Compute distributions", command=lambda: self.controller.distributions_click())
        distributions_button.pack(side=tk.LEFT)

        binarize_button = tk.Button(self.toolbar, text="Binarize", command=lambda: self.controller.binarize_click())
        binarize_button.pack(side=tk.LEFT)

        autoexclude_button = tk.Button(self.toolbar, text="Autoexclude", command=lambda: self.controller.autoexclude_click())
        autoexclude_button.pack(side=tk.LEFT)

        autolimit_button = tk.Button(self.toolbar, text="Autolimit", command=lambda: self.controller.autolimit_click())
        autolimit_button.pack(side=tk.LEFT)

        next_button = tk.Button(self.toolbar, text="Next", command=lambda: self.controller.next_click())
        next_button.pack(side=tk.RIGHT)

        self.cell_number_text = tk.Label(self.toolbar, text="0")
        self.cell_number_text.pack(side=tk.RIGHT)

        previous_button = tk.Button(self.toolbar, text="Previous", command=lambda: self.controller.previous_click())
        previous_button.pack(side=tk.RIGHT)

        self.bind("<Left>", lambda e: self.controller.previous_click())
        self.bind("<Right>", lambda e: self.controller.next_click())

        exclude_button = tk.Button(self.toolbar, text="exclude", command=lambda: self.controller.exclude_click())
        exclude_button.pack(side=tk.RIGHT)

        unexclude_button = tk.Button(self.toolbar, text="unexclude", command=lambda: self.controller.unexclude_click())
        unexclude_button.pack(side=tk.RIGHT)

        self.minsize(width=WIDTH, height=HEIGHT)
        self.controller = None

    def register(self, controller):
        self.controller = controller

    def open_file(self):
        filename = filedialog.askopenfilename(title="Select file", filetypes=(("dat files", "*.dat"), ("YAML files", "*.yaml")))
        if filename == '':
            return None
        return filename

    def open_directory(self):
        """
        This method displays the file dialog box to open file and returns the
        file name.
        """
        directory = filedialog.askdirectory()
        if directory == '':
            return None
        return directory

    def save_as(self, extension):
        filename = filedialog.asksaveasfile(mode='w', defaultextension=extension)
        if filename is None:
            return None
        return filename.name

    def draw_fig(self, fig):
        if self.canvas is not False:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def open_settings_window(self, settings):
        # Open window
        self.settings_window = tk.Toplevel()
        self.settings_window.title("Settings")

        # Add upper frame
        main_frame = tk.Frame(self.settings_window)
        main_frame.pack(fill=tk.BOTH, expand=tk.YES)

        self.entries = self.__add_frame(settings, main_frame)

        apply_parameters_button = tk.Button(main_frame, text="Apply parameters", command=lambda: self.controller.apply_parameters_click())
        apply_parameters_button.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=tk.YES)

    def __add_frame(self, parameter, container):
        if type(parameter) in (int, float):
            e = tk.Entry(container)
            e.pack(side=tk.LEFT)
            e.delete(0, tk.END)
            e.insert(0, parameter)
            return e
        elif type(parameter) is dict:
            dictionary = {}
            for key in parameter:
                parameter_frame = tk.LabelFrame(container, text=key)
                parameter_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
                dictionary[key] = self.__add_frame(parameter[key], parameter_frame)
            return dictionary
        elif type(parameter) is list:
            array = []
            for key in range(len(parameter)):
                array.append(self.__add_frame(parameter[key], container))
            return array
