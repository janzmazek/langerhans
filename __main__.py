from src.data import Data
from src.networks import Networks
from src.analysis import Analysis
import os
import yaml
import numpy as np

LOCATION = "../DATA"
DATA = input("Series number: ")

def f(name):
    abs = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(abs, name)

series_location = f("{0}/{1}/series{1}.dat".format(LOCATION, DATA))
sampling_location = f("{0}/{1}/sampling{1}.dat".format(LOCATION, DATA))
settings_location = f("{0}/{1}/settings.yaml".format(LOCATION, DATA))

filtered_location = f("{0}/{1}/filtered".format(LOCATION, DATA))
distributions_location = f("{0}/{1}/distributions".format(LOCATION, DATA))
binarized_location = f("{0}/{1}/binarized".format(LOCATION, DATA))

positions_location = f("{0}/{1}/positions{1}.dat".format(LOCATION, DATA))

# ----------------------------------  OPEN  ---------------------------------- #
series = np.loadtxt(series_location)[:-1,:]
sampling = int(np.loadtxt(sampling_location))
positions = np.loadtxt(positions_location)
with open(settings_location, 'r') as stream:
    try:
        settings = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise(ValueError("Could not open settings file."))

# ----------------------------------  MAIN  ---------------------------------- #

data = Data(series, sampling, positions, settings)

data.filter()
data.compute_distributions()
# data.plot_distributions(distributions_location)
data.binarize_fast()
data.binarize_slow()
# data.plot_binarized(binarized_location)
data.exclude_bad_cells()

networks = Networks(data)
networks.build_network()
networks.draw_network()
# networks.draw_network()

# analysis = Analysis(data, networks)
# analysis.compare_slow_fast()
# print(analysis.compare_correlation_distance())
