# Hardcoded constants used throughout the paper
import numpy as np

nonlinearity = np.tanh

# Ranges for weight and bias standard deviations
nw = 41
nb = 41
wmax = 5
bmax = 4
weight_sigmas =  np.linspace(1, wmax, nw)
bias_sigmas = np.linspace(0, bmax, nb)

# Chosen indices for plotting.
widxs = [3, 15, 30]
bidxs = [3, 3, 3]
