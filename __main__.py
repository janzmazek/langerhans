from src.analysis import Analysis
import os

def f(name):
    abs = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(abs, name)

# Smoothing parameters
repF=3
NF=3

analysis = Analysis(f("data/series064.dat"), f("data/sampling.txt"), f("data/lege064.dat"))

# ---------------------------------- FIlTER ---------------------------------- #
analysis.filter()
analysis.save_slow(f("data/filtered/slow"))
analysis.save_fast(f("data/filtered/fast"))
analysis.plot_filtered(f("data/filtered"))
#
# analysis.import_slow(f("data/filtered/slow"))
# analysis.import_fast(f("data/filtered/fast"))

# --------------------------------- BINARIZE --------------------------------- #

analysis.binarize_slow()
analysis.smooth_fast(repF, NF)
analysis.binarize_fast()
analysis.save_binarized_fast(f("data/binarized/fast"))
analysis.save_binarized_slow(f("data/binarized/slow"))
analysis.plot_binarized(f("data/binarized"))

# analysis.import_binarized_slow(f("data/binarized/slow"))
# analysis.import_binarized_fast(f("data/binarized/fast"))

# ---------------------------------- EXCLUDE --------------------------------- #

# analysis.import_excluded(f("data/exclude.dat"))
# analysis.exclude()
#
# densities = analysis.compare_slow_fast()
# for i in range(len(densities)):
#     print(densities[i])
# analysis.compare_correlation_distance()
#
# analysis.build_network()
#
# dynamics_parameters = analysis.dynamics_parameters()
# topology_parameters = analysis.topology_parameters()
#
# for cell in range(len(dynamics_parameters)):
#     print(dynamics_parameters[cell])
#     print(topology_parameters[cell])
#
# analysis.correlations(f("data/correlations.txt"))
# analysis.animation()
