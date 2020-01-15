import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Window parameters
WIDTH = 1000
HEIGHT = 600

class View(tk.Tk):
    """docstring for View."""

    def __init__(self, *args, **kwargs):
        super(View, self).__init__(*args, **kwargs)
        self.title("Analysis of Calcium Signals")

        menubar = tk.Menu(self)

        filemenu = tk.Menu(menubar, tearoff=0)

        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Import data", command=lambda: self.controller.import_data())
        filemenu.add_command(label="Save data", command=lambda: 0)

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

        next_button = tk.Button(self.toolbar, text="Next", command=lambda: self.controller.next_click())
        next_button.pack(side=tk.RIGHT)

        self.cell_number_text = tk.Label(self.toolbar, text="0")
        self.cell_number_text.pack(side=tk.RIGHT)

        previous_button = tk.Button(self.toolbar, text="Previous", command=lambda: self.controller.previous_click())
        previous_button.pack(side=tk.RIGHT)

        exclude_button = tk.Button(self.toolbar, text="exclude", command=lambda: self.controller.exclude_click())
        exclude_button.pack(side=tk.RIGHT)

        self.minsize(width=WIDTH, height=HEIGHT)
        self.controller = None

    def register(self, controller):
        self.controller = controller

    def open_directory(self):
        """
        This method displays the file dialog box to open file and returns the
        file name.
        """
        filename = filedialog.askdirectory()
        if filename == '':
            return None
        return filename

    def draw_fig(self, fig):
        if self.canvas is not False:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
