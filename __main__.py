from src.data import Data
from src.networks import Networks
import os

LOCATION = "../DATA"
DATA = input("Series number: ")

def f(name):
    abs = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(abs, name)

series = f("{0}/{1}/series{1}.dat".format(LOCATION, DATA))
sampling = f("{0}/{1}/sampling{1}.dat".format(LOCATION, DATA))
settings = f("{0}/{1}/settings.yaml".format(LOCATION, DATA))

filtered = f("{0}/{1}/filtered".format(LOCATION, DATA))
distributions = f("{0}/{1}/distributions".format(LOCATION, DATA))
binarized = f("{0}/{1}/binarized".format(LOCATION, DATA))

positions = f("{0}/{1}/positions{1}.dat".format(LOCATION, DATA))

# ----------------------------------  MAIN  ---------------------------------- #
data = Data(series, sampling, settings)

data.filter()
data.compute_distributions()
data.binarize_fast()
data.binarize_slow()

networks = Networks(data, positions)
networks.build_network()
