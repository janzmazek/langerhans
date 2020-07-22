import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import copy
import webbrowser


# Window parameters
WIDTH = 1000
HEIGHT = 600

WHITE = "white"
TEXT = "white"
# BG = "#2d2f37"
BG = "#3E4149"

WELCOME_TEXT = "WELCOME\n \
To start analyzing, load a data file (a 2-D matrix readable by np.loadtxt).\n \
Cells should be rows of the matrix, time should be columns of the matrix.\n \
You can save or load current state in the form of a pickle object."

class View(tk.Tk):
    """docstring for View."""

    def __init__(self, *args, **kwargs):
        super(View, self).__init__(*args, **kwargs)
        self.title("Analysis of Calcium Signals")

        menubar = tk.Menu(self)

        importmenu = tk.Menu(menubar, tearoff=0)
        exportmenu = tk.Menu(menubar, tearoff=0)
        editmenu = tk.Menu(menubar, tearoff=0)
        aboutmenu = tk.Menu(menubar, tearoff=0)

        menubar.add_cascade(label="Import", menu=importmenu)
        importmenu.add_command(label="Import data", command=lambda: self.controller.import_data())
        importmenu.add_command(label="Import settings", command=lambda: self.controller.import_settings())
        importmenu.add_command(label="Import excluded", command=lambda: self.controller.import_excluded())
        importmenu.add_command(label="Import object (pickle)", command=lambda: self.controller.import_object())
        menubar.add_cascade(label="Export", menu=exportmenu)
        exportmenu.add_command(label="Export settings", command=lambda: self.controller.save_settings())
        exportmenu.add_command(label="Export image", command=lambda: self.controller.save_image())
        exportmenu.add_command(label="Export event plot", command=lambda: self.controller.save_eventplot())
        exportmenu.add_command(label="Export excluded", command=lambda: self.controller.save_excluded())
        exportmenu.add_command(label="Export object (pickle)", command=lambda: self.controller.save_object())
        menubar.add_cascade(label="Edit", menu=editmenu)
        editmenu.add_command(label="Settings", command=lambda: self.controller.edit_settings())
        menubar.add_cascade(label="About", menu=aboutmenu)
        aboutmenu.add_command(label="Info", command=lambda: webbrowser.open("https://github.com/janzmazek/cell-networks"))

        self.config(menu=menubar)

        self.toolbar = tk.LabelFrame(self, text="Preprocessing Tools", padx=5, pady=5, bg=BG, fg=TEXT)
        self.toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.NO)

        self.topframe = tk.Frame(self.toolbar, bg=BG)
        self.topframe.pack(anchor="center")

        self.navbar = tk.LabelFrame(self, text="Navigation", padx=5, pady=5, bg=BG, fg=TEXT)
        self.navbar.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=tk.NO)

        filter_button = tk.Button(self.topframe, highlightbackground=BG, text="Filter", command=lambda: self.controller.filter_click())
        filter_button.pack(side=tk.LEFT)

        distributions_button = tk.Button(self.topframe, highlightbackground=BG, text="Compute distributions", command=lambda: self.controller.distributions_click())
        distributions_button.pack(side=tk.LEFT)

        binarize_button = tk.Button(self.topframe, highlightbackground=BG, text="Binarize", command=lambda: self.controller.binarize_click())
        binarize_button.pack(side=tk.LEFT)

        autoexclude_button = tk.Button(self.topframe, highlightbackground=BG, text="Autoexclude", command=lambda: self.controller.autoexclude_click())
        autoexclude_button.pack(side=tk.LEFT)

        autolimit_button = tk.Button(self.topframe, highlightbackground=BG, text="Autolimit", command=lambda: self.controller.autolimit_click())
        autolimit_button.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, bg="red")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        text = tk.Label(self.canvas, text=WELCOME_TEXT)
        text.pack(anchor="center", fill=tk.BOTH, expand=1)

        exclude_button = tk.Button(self.navbar, highlightbackground=BG, text="(↓) exclude", command=lambda: self.controller.exclude_click())
        exclude_button.pack(side=tk.LEFT)

        self.bottomframe = tk.Frame(self.navbar, bg=BG)
        self.bottomframe.pack(side=tk.LEFT, fill="none", expand=True)

        previous_button = tk.Button(self.bottomframe, highlightbackground=BG, text="(←) Prev", command=lambda: self.controller.previous_click())
        previous_button.pack(side=tk.LEFT)

        self.cell_number_text = tk.Label(self.bottomframe, bg=BG, fg=TEXT, text="0")
        self.cell_number_text.pack(side=tk.LEFT)

        next_button = tk.Button(self.bottomframe, highlightbackground=BG, text="Next (→)", command=lambda: self.controller.next_click())
        next_button.pack(side=tk.LEFT)

        unexclude_button = tk.Button(self.navbar, highlightbackground=BG, text="unexclude (↑)", command=lambda: self.controller.unexclude_click())
        unexclude_button.pack(side=tk.RIGHT)

        self.bind("<Left>", lambda e: self.controller.previous_click())
        self.bind("<Right>", lambda e: self.controller.next_click())
        self.bind("<Up>", lambda e: self.controller.unexclude_click())
        self.bind("<Down>", lambda e: self.controller.exclude_click())

        self.minsize(width=WIDTH, height=HEIGHT)
        self.controller = None

    def register(self, controller):
        self.controller = controller

    def open_file(self):
        filename = filedialog.askopenfilename(title="Select file", filetypes=(("dat files", "*.dat"), ("YAML files", "*.yaml"), ("pickle files", "*.pkl")))
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
        if type(self.canvas) == tk.Canvas:
            self.canvas.destroy()
        else:
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
