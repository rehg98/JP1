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

redshifts = data['z_cl']

# Bin the cluster data by redshift
clusters1 = data[(redshifts > 0.1) & (redshifts < 0.2)]
clusters2 = data[(redshifts > 0.2) & (redshifts < 0.35)]
clusters3 = data[(redshifts > 0.35) & (redshifts < 0.5)]
clusters4 = data[(redshifts > 0.5) & (redshifts < 0.75)]
clusters5 = data[(redshifts > 0.75) & (redshifts < 1.2)]

num_radial_bins = 15

# Compute the average colors (and their variances) for a given redshift bin

def avg_colors(clusters):
    
    clusters = clusters[np.where(clusters['R'] > 0.015)]
    radii = np.array(clusters['R'])
    radial_bins = np.logspace(np.log10(np.amin(radii)), np.log10(np.amax(radii)), num_radial_bins + 1)
    widths = np.diff(radial_bins)
    annular_areas = np.pi * np.diff(radial_bins ** 2.)

    # Arrays containing magnitude data in four different filters
    g = np.array(clusters['gmag_forced_cmodel'])
    r = np.array(clusters['rmag_forced_cmodel'])
    i = np.array(clusters['imag_forced_cmodel'])
    z = np.array(clusters['zmag_forced_cmodel'])

    # For each color g-r, r-i, and i-z: first, throw out data points that have magnitudes of "nan" or "inf" (pruning the
    # arrays of radii and magnitudes accordingly), then use scipy.binned_statistic to sort the data into radial bins and
    # compute the number of objects and the sum and variance (noise^2) of the color values in those bins

    radii_gr_pruned = radii[np.isfinite(g) & np.isfinite(r)]
    gr = g[np.isfinite(g) & np.isfinite(r)] - r[np.isfinite(g) & np.isfinite(r)]
    count_gr = binned_statistic(radii_gr_pruned, gr, 'count', radial_bins)[0]
    sum_gr = binned_statistic(radii_gr_pruned, gr, 'sum', radial_bins)[0]
    var_gr = binned_statistic(radii_gr_pruned, gr, np.var, radial_bins)[0]
    var_gr /= count_gr

    radii_ri_pruned = radii[np.isfinite(r) & np.isfinite(i)]
    ri = r[np.isfinite(r) & np.isfinite(i)] - i[np.isfinite(r) & np.isfinite(i)]
    count_ri = binned_statistic(radii_ri_pruned, ri, 'count', radial_bins)[0]
    mean_ri = binned_statistic(radii_ri_pruned, ri, 'mean', radial_bins)[0]
    var_ri = binned_statistic(radii_ri_pruned, ri, np.var, radial_bins)[0]
    var_ri /= count_ri

    radii_iz_pruned = radii[np.isfinite(i) & np.isfinite(z)]
    iz = i[np.isfinite(i) & np.isfinite(z)] - z[np.isfinite(i) & np.isfinite(z)]
    count_iz = binned_statistic(radii_iz_pruned, iz, 'count', radial_bins)[0]
    mean_iz = binned_statistic(radii_iz_pruned, iz, 'mean', radial_bins)[0]
    var_iz = binned_statistic(radii_iz_pruned, iz, np.var, radial_bins)[0]
    var_iz /= count_iz

    # Compute the mean and variance of the i magnitudes over the same radial bins as above
    radii_i_pruned = radii[np.isfinite(i)]
    count_i = binned_statistic(radii_i_pruned, i, 'count', radial_bins)[0]
    mean_i = binned_statistic(radii_i_pruned, i, 'mean', radial_bins)[0]
    var_i = binned_statistic(radii_i_pruned, i, np.var, radial_bins)[0]
    var_i /= count_i
    
    return (radial_bins, widths, (mean_gr, var_gr), (mean_ri, var_ri), (mean_iz, var_iz), (mean_i, var_i))



    # Plot bar graphs for the average color data in a given redshift bin

def plot_profiles(clusters, z_bin_num):
    
    clusters_data = avg_colors(clusters)

    radial_bins = clusters_data[0]
    widths = clusters_data[1]
    
    # gr, ri, iz, and i are 2-tuples of the form (mean_color, var_color)
    gr = clusters_data[2]
    ri = clusters_data[3]
    iz = clusters_data[4]
    i = clusters_data[5]
    
    z_bins = ("1 (0.1 < z < 0.2)", "2 (0.2 < z < 0.35)", "3 (0.35 < z < 0.5)", "4 (0.5 < z < 0.75)", 
             "5 (0.75 < z < 1.2)")

    plt.bar(radial_bins[:-1], gr[0], width=widths, fill=False, align="edge", yerr=np.sqrt(gr[1]))
    plt.xscale('log')
    plt.title("Average g-r Color vs. Radius for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average g-r [magnitudes]")
    plt.show()

    plt.bar(radial_bins[:-1], ri[0], width=widths, fill=False, align="edge", yerr=np.sqrt(ri[1]))
    plt.xscale('log')
    plt.title("Average r-i Color vs. Radius for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average r-i [magnitudes]")
    plt.show()

    plt.bar(radial_bins[:-1], iz[0], width=widths, fill=False, align="edge", yerr=np.sqrt(iz[1]))
    plt.xscale('log')
    plt.title("Average i-z Color vs. Radius for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average i-z [magnitudes]")
    plt.show()

    plt.bar(radial_bins[:-1], i[0], width=widths, fill=False, align="edge", yerr=np.sqrt(i[1]))
    plt.xscale('log')
    plt.title("Average i-filter Magnitude vs. Radius for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average i [magnitudes]")
    plt.show()
    
    return clusters_data



    # Use NMF to unmix the color profiles generated by plot_profiles and avg_colors
# The arguments of "unmix" are 2-tuples of the form (mean_color, var_color). For example, gr is really (mean_gr, var_gr)

def prox_field(S, step, bins=-4):
    S[0, bins:] = 1
    S[1:, bins:] = 0
    return S

def unmix(gr, ri, iz, i):
    n = num_radial_bins     # component resolution
    k = 2                   # number of components
    b = 4                   # number of observations (b=4 for g-r, r-i, i-z, and i)

    # Data matrix to be unmixed
    Y = np.array([gr[0], ri[0], iz[0], i[0]])
    
    # if noise is variable, specify variance matrix of the same shape as Y
    W = 1. / np.array([gr[1], ri[1], iz[1], i[1]])

    # initialize and run NMF
    A = np.random.uniform(size = (b, k))
    S = np.random.uniform(size = (k, n))
    pA = po.prox_id
    
    psum = partial(po.prox_unity_plus, axis=0)
    pfield = partial(prox_field, bins=-4)
    pS = po.AlternatingProjections([pfield, psum])
    
    nmf(Y, A, S, W=W, prox_A=pA, prox_S=pS, e_rel=1e-6, e_abs=0)
    
    return A, S




    def plot_factorized_profiles(clusters_data, z_bin_num):
    

    radial_bins = clusters_data[0]
    bin_midpoints = (radial_bins[1:] * radial_bins[:-1]) ** 0.5
    widths = clusters_data[1]
    
    # gr, ri, iz, and i are 2-tuples of the form (mean_color, var_color)
    A, S = unmix(gr = clusters_data[2], ri = clusters_data[3], iz = clusters_data[4], i = clusters_data[5])
    print("A:\n", A, "\n")
    print("S:\n", S, "\n")
    
    z_bins = ("1 (0.1 < z < 0.2)", "2 (0.2 < z < 0.35)", "3 (0.35 < z < 0.5)", "4 (0.5 < z < 0.75)", 
             "5 (0.75 < z < 1.2)")
    
    # Plot S_k (the relative abundance of the kth component) vs. radius for each k
    
    for k in range(S.shape[0]):
        plt.plot(bin_midpoints, S[k], label = "Component %s" % (k+1))
    plt.xscale('log')
    plt.title("Relative Component Abundance vs. Radius for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel("Relative abundance")
    plt.legend()
    plt.show()
    

    Y_remixed = np.dot(A, S)
    
    # Plot the reconstructed color profiles following NMF 
    # (that is, plot Y_remixed = A x S, where A and S are the results of factorizing the initial data matrix Y)
    
    plt.bar(radial_bins[:-1], clusters_data[2][0], width=widths, align="edge", alpha=0.5, edgecolor='black', 
            yerr=np.sqrt(clusters_data[2][1]), label="Data")
    plt.bar(radial_bins[:-1], Y_remixed[0], width=widths, align="edge", alpha=0.5, edgecolor='black', label="Predicted Profile")
    plt.xscale('log')
    plt.title("Average g-r Color vs. Radius for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average g-r [magnitudes]")
    plt.legend()
    plt.show()

    plt.bar(radial_bins[:-1], clusters_data[3][0], width=widths, align="edge", alpha=0.5, edgecolor='black', 
            yerr=np.sqrt(clusters_data[3][1]), label="Data")
    plt.bar(radial_bins[:-1], Y_remixed[1], width=widths, align="edge", alpha=0.5, edgecolor='black', label="Predicted Profile")
    plt.xscale('log')
    plt.title("Average r-i Color vs. Radius for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average r-i [magnitudes]")
    plt.legend()
    plt.show()

    plt.bar(radial_bins[:-1], clusters_data[4][0], width=widths, align="edge", alpha=0.5, edgecolor='black', 
            yerr=np.sqrt(clusters_data[4][1]), label="Data")
    plt.bar(radial_bins[:-1], Y_remixed[2], width=widths, align="edge", alpha=0.5, edgecolor='black', label="Predicted Profile")
    plt.xscale('log')
    plt.title("Average i-z Color vs. Radius for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average i-z [magnitudes]")
    plt.legend()
    plt.show()

    plt.bar(radial_bins[:-1], clusters_data[5][0], width=widths, align="edge", alpha=0.5, edgecolor='black', 
            yerr=np.sqrt(clusters_data[5][1]), label="Data")
    plt.bar(radial_bins[:-1], Y_remixed[3], width=widths, align="edge", alpha=0.5, edgecolor='black', label="Predicted Profile")
    plt.xscale('log')
    plt.title("Average i-filter Magnitude vs. Radius for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average i [magnitudes]")
    plt.legend()
    plt.show()
    
    '''
    # Plot the components (that is, the rows of S) for a given redshift bin
    
    plt.bar(radial_bins[:-1], S[0], width=widths, fill=False, align="edge")#, yerr=np.sqrt(i[1]))
    plt.xscale('log')
    plt.title("Component 1 for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average color [magnitudes]")
    plt.show()
    
    plt.bar(radial_bins[:-1], S[1], width=widths, fill=False, align="edge")#, yerr=np.sqrt(i[1]))
    plt.xscale('log')
    plt.title("Component 2 for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average color [magnitudes]") 
    plt.show()
    
    plt.bar(radial_bins[:-1], S[2], width=widths, fill=False, align="edge")#, yerr=np.sqrt(i[1]))
    plt.xscale('log')
    plt.title("Component 3 for Redshift Bin %s" % z_bins[z_bin_num - 1])
    plt.xlabel("Radius from cluster center [Mpc]")
    plt.ylabel ("Average color [magnitudes]")
    plt.show()
    '''
    
#Plot bar graphs for the clusters1 data

clusters1_data = plot_profiles(clusters1, 1)

#Plot bar graphs for the post-NMF clusters1 profiles

plot_factorized_profiles(clusters1_data, 1)