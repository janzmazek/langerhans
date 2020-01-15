from src.data import Data
from src.analysis import Analysis

from src.view import View
from src.controller import Controller

if __name__ == '__main__':
    data = Data()
    analysis = Analysis()
    view = View()
    controller = Controller(data, analysis, view)

    view.mainloop()
