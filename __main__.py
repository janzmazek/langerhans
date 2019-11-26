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

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

series_location = f("{0}/{1}/series{1}.dat".format(LOCATION, DATA))
settings_location = f("{0}/{1}/settings.yaml".format(LOCATION, DATA))

filtered_location = f("{0}/{1}/filtered".format(LOCATION, DATA))
distributions_location = f("{0}/{1}/distributions".format(LOCATION, DATA))
binarized_location = f("{0}/{1}/binarized".format(LOCATION, DATA))
networks_location = f("{0}/{1}/networks".format(LOCATION, DATA))
analysis_location = f("{0}/{1}/analysis".format(LOCATION, DATA))

positions_location = f("{0}/{1}/positions{1}.dat".format(LOCATION, DATA))

# ----------------------------------  OPEN  ---------------------------------- #
series = np.loadtxt(series_location)[:-1,:]
positions = np.loadtxt(positions_location)
with open(settings_location, 'r') as stream:
    try:
        settings = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise(ValueError("Could not open settings file."))

# ----------------------------------  MAIN  ---------------------------------- #

# Initial call to print 0% progress
l = 14
print_progress_bar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

data = Data(series, positions, settings)
print_progress_bar(1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

data.filter()
print_progress_bar(2, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

# data.plot_filtered(filtered_location)
print_progress_bar(3, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

data.compute_distributions()
print_progress_bar(4, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

# data.plot_distributions(distributions_location)
data.binarize_fast()
print_progress_bar(5, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

data.binarize_slow()
print_progress_bar(6, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

# data.plot_binarized(binarized_location)
print_progress_bar(7, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

data.exclude_bad_cells()
print_progress_bar(8, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

analysis = Analysis(data)
print_progress_bar(9, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

# analysis.draw_networks(networks_location)
print_progress_bar(10, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

# analysis.print_parameters()
print_progress_bar(11, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

analysis.compare_slow_fast(False)
print_progress_bar(12, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

analysis.compare_correlation_distance(False)
print_progress_bar(13, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

analysis.plot_analysis(analysis_location)
print_progress_bar(14, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
