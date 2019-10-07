from src.analysis import Analysis
import os

LOCATION = "../DATA"
DATA = "022"

def f(name):
    abs = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(abs, name)

series = f("{0}/{1}/series{1}.dat".format(LOCATION, DATA))
sampling = f("{0}/{1}/sampling{1}.dat".format(LOCATION, DATA))
positions = f("{0}/{1}/positions{1}.dat".format(LOCATION, DATA))
excluded = f("{0}/{1}/excluded{1}.dat".format(LOCATION, DATA))

filtered = f("{0}/{1}/filtered".format(LOCATION, DATA))
binarized = f("{0}/{1}/binarized".format(LOCATION, DATA))

analysis = Analysis(series, sampling, positions)

# ---------------------------------- FIlTER ---------------------------------- #
if os.path.isfile(filtered + "/slow") and os.path.isfile(filtered + "/fast"):
    analysis.import_slow(filtered + "/slow")
    analysis.import_fast(filtered + "/fast")
else:
    analysis.filter()
    analysis.smooth_fast()
    analysis.save_slow(filtered + "/slow")
    analysis.save_fast(filtered + "/fast")
    analysis.plot_filtered(filtered)

# --------------------------------- BINARIZE --------------------------------- #
if os.path.isfile(binarized + "/slow") and os.path.isfile(binarized + "/fast"):
    analysis.import_binarized_slow(binarized + "/slow")
    analysis.import_binarized_fast(binarized + "/fast")
else:
    analysis.binarize_slow()
    analysis.smooth_fast()
    analysis.binarize_fast()
    analysis.save_binarized_slow(binarized + "/slow")
    analysis.save_binarized_fast(binarized + "/fast")
    analysis.plot_binarized(binarized)

# ---------------------------------- EXCLUDE --------------------------------- #
if os.path.isfile(excluded):
    analysis.import_excluded(f("data/exclude.dat"))
    analysis.exclude()

# --------------------------------- ANALYSIS --------------------------------- #

# print(analysis.compare_slow_fast())
# (distances, correlations_slow, correlations_fast) = analysis.compare_correlation_distance()
# print(analysis.build_network(0.85,0.76))
# (correlation_matrix_slow, correlation_matrix_fast) = analysis.build_network(0.85, 0.76)
# topology = analysis.topology_parameters()
# dynamics = analysis.dynamics_parameters()
# slow_ND = []
# fast_ND = []
# slow_Ton = []
# fast_Ton = []
# for i in range(len(topology)):
#     slow_ND.append(topology[i]["NDs"])
#     fast_ND.append(topology[i]["NDf"])
# print(slow_ND, fast_ND)
# positions = analysis.get_positions()
# (adjacency_slow, adjacency_fast) = analysis.get_adjacency()
# analysis.get_excluded_positions(f("data/excluded_positions.dat"))
# analysis.correlations(f("data/correlations.txt"))
