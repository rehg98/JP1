from proxmin import nmf
from proxmin.utils import Traceback
from scipy.optimize import linear_sum_assignment
from scipy.stats import binned_statistic
import numpy as np
import matplotlib.pyplot as plt
import time
from proxmin import operators as po
from functools import partial

##################################################################################################################

data = np.load("hsc_stacked.npy")

redshifts = np.array([item[0] for item in data])

# Bin the cluster data by redshift
cluster1 = data[(redshifts > 0.1) & (redshifts < 0.2)]
cluster2 = data[(redshifts > 0.2) & (redshifts < 0.35)]
cluster3 = data[(redshifts > 0.35) & (redshifts < 0.5)]
cluster4 = data[(redshifts > 0.5) & (redshifts < 0.75)]
cluster5 = data[(redshifts > 0.75) & (redshifts < 1.2)]

##################################################################################################################

# Compute the average colors for the first redshift bin

radii_1 = np.array([item[1] for item in cluster1])
num_bins_1 = 20
radial_bins_1 = np.logspace(np.log10(0.0319), np.log10(5.), num_bins_1)
widths_1 = np.diff(radial_bins_1)

# Arrays containing magnitude data in four different filters
g_1 = np.array([item[2] for item in cluster1])
r_1 = np.array([item[3] for item in cluster1])
i_1 = np.array([item[4] for item in cluster1])
z_1 = np.array([item[5] for item in cluster1])

# For each color g-r, r-i, and i-z: first, throw out data points that have magnitudes of "nan" or "inf" (pruning the
# arrays of radii and magnitudes accordingly), then use scipy.binned_statistic to sort the data into radial bins and
# compute the mean and variance (noise^2) of the color values in those bins

radii_1_gr_pruned = radii_1[np.isfinite(g_1) & np.isfinite(r_1)]
gr_1 = g_1[np.isfinite(g_1) & np.isfinite(r_1)] - r_1[np.isfinite(g_1) & np.isfinite(r_1)]
mean_gr_1 = binned_statistic(radii_1_gr_pruned, gr_1, 'mean', radial_bins_1)[0]
var_gr_1 = binned_statistic(radii_1_gr_pruned, gr_1, np.var, radial_bins_1)[0]

radii_1_ri_pruned = radii_1[np.isfinite(r_1) & np.isfinite(i_1)]
ri_1 = r_1[np.isfinite(r_1) & np.isfinite(i_1)] - i_1[np.isfinite(r_1) & np.isfinite(i_1)]
mean_ri_1 = binned_statistic(radii_1_ri_pruned, ri_1, 'mean', radial_bins_1)[0]
var_ri_1 = binned_statistic(radii_1_ri_pruned, ri_1, np.var, radial_bins_1)[0]

radii_1_iz_pruned = radii_1[np.isfinite(i_1) & np.isfinite(z_1)]
iz_1 = i_1[np.isfinite(i_1) & np.isfinite(z_1)] - z_1[np.isfinite(i_1) & np.isfinite(z_1)]
mean_iz_1 = binned_statistic(radii_1_iz_pruned, iz_1, 'mean', radial_bins_1)[0]
var_iz_1 = binned_statistic(radii_1_iz_pruned, iz_1, np.var, radial_bins_1)[0]

# Compute the mean and variance of the i magnitudes over the same radial bins as above
radii_1_i_pruned = radii_1[np.isfinite(i_1)]
mean_i_1 = binned_statistic(radii_1_i_pruned, i_1, 'mean', radial_bins_1)[0]
var_i_1 = binned_statistic(radii_1_i_pruned, i_1, np.var, radial_bins_1)[0]

##################################################################################################################

#Plot bar graphs for the cluster1 data

plt.bar(radial_bins_1[:-1], mean_gr_1, width=widths_1, fill=False, align="edge", yerr=np.sqrt(var_gr_1))
plt.title("Average g-r Color vs. Radius for Redshift Bin 1 (0.1 < z < 0.2)")
plt.xlabel("Radius from cluster center [Mpc]")
plt.ylabel ("Average g-r [magnitudes]")
plt.show()

plt.bar(radial_bins_1[:-1], mean_ri_1, width=widths_1, fill=False, align="edge", yerr=np.sqrt(var_ri_1))
plt.title("Average r-i Color vs. Radius for Redshift Bin 1 (0.1 < z < 0.2)")
plt.xlabel("Radius from cluster center [Mpc]")
plt.ylabel ("Average r-i [magnitudes]")
plt.show()

plt.bar(radial_bins_1[:-1], mean_iz_1, width=widths_1, fill=False, align="edge", yerr=np.sqrt(var_iz_1))
plt.title("Average i-z Color vs. Radius for Redshift Bin 1 (0.1 < z < 0.2)")
plt.xlabel("Radius from cluster center [Mpc]")
plt.ylabel ("Average i-z [magnitudes]")
plt.show()

plt.bar(radial_bins_1[:-1], mean_i_1, width=widths_1, fill=False, align="edge", yerr=np.sqrt(var_i_1))
plt.title("Average i-filter Magnitude vs. Radius for Redshift Bin 1 (0.1 < z < 0.2)")
plt.xlabel("Radius from cluster center [Mpc]")
plt.ylabel ("Average i [magnitudes]")
plt.show()

##################################################################################################################