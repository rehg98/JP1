from proxmin import nmf
from proxmin.utils import Traceback
from scipy.optimize import linear_sum_assignment
from scipy.stats import binned_statistic
import numpy as np
import matplotlib.pyplot as plt
import time
from proxmin import operators as po
from functools import partial

data = np.load("hsc_stacked.npy")

redshifts = np.array([item[0] for item in data])

cluster1 = data[(redshifts > 0.1) & (redshifts < 0.2)]
print(cluster1)
cluster2 = data[(redshifts > 0.2) & (redshifts < 0.35)]
cluster3 = data[(redshifts > 0.35) & (redshifts < 0.5)]
cluster4 = data[(redshifts > 0.5) & (redshifts < 0.75)]
cluster5 = data[(redshifts > 0.75) & (redshifts < 1.2)]

radii_1 = np.array([item[1] for item in cluster1])
g_1 = np.array([item[2] for item in cluster1])
#r_1 = np.array([item[3] for item in cluster1])

#gr_1 = g_1 - r_1

#for i in range(g_1.size):
#	if(not np.isfinite(g_1[i])): print(g_1[i])

#rad_bin1 = gr_1[(radii_1 > 0.) & (radii_1 < 0.5)]
#print(np.nansum(rad_bin1)


#radial_bins_1 = np.array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
#mean_gr_1 = binned_statistic(radii_1, gr_1, np.nanmean, radial_bins_1)[0]
#print(mean_gr_1)
