#!/usr/bin/env python
# coding: utf-8

"""
Created on  14.06.2023
Last update 14.06.2023

author: Ludwig Wolfgruber
e-mail: ludwig.wolfgruber@gmx.at
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import xarray as xr


# defining styles
# discrete colormaps
cmap = plt.cm.viridis
vir3 = cmap(np.linspace(0.8,0.2,3))
vir4 = cmap(np.linspace(0.8,0.2,4))
vir1 = cmap(0.5)

mgm3 = plt.cm.magma(np.linspace(0.8,0.2,3))
mgm4 = plt.cm.magma(np.linspace(0.8,0.2,4))

blues3 = plt.cm.Blues(np.linspace(0.4, 1, 3))
blues4 = plt.cm.Blues(np.linspace(0.4, 1, 4))

greens3 = plt.cm.Greens(np.linspace(0.4, 1, 3))
greens4 = plt.cm.Greens(np.linspace(0.4, 1, 4))

# linestyles
stylelist = ['solid', 'dashdot', 'dashed', 'dotted']

# for determing font and ticks size in figures:
# the smaller SCALE, the larger fonts are
# compared to the figure
SCALE = 0.6

plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = True

# load Rain Rate data for all functions
thrsh_perc = xr.load_dataset('../data/percentile_threshold.nc')
all_perc = thrsh_perc.percentile.values
all_thrsh = thrsh_perc.thrsh.values
all_occ = 100 - all_perc
occ = np.array([0.25, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 34])#100])
thrsh = np.array([15.358784165382353, 6.546857919692981, 4.558161976337443,
                  3.4287931823730475, 2.6827078282833097, 2.1295418691635,
                  1.3871746397018434, 0.9479302614927281, 0.669832711219789,
                  0.35095661997795163, 0.19140626490116194, 0.11011233389377595,
                  0.0668206149339677, 0.042710336297750465, 0.02781107142567632,
                  0.016654083877801935, 0.009048837069422012, 0.0])


# functions to switch from thrsh to foo and back
def match_pos(pos, where, to):
    shape = pos.shape
    pos = np.array(pos).astype(float)
    n = len(pos)
    out = np.empty(n)
    
    for i in range(n):
        idx = np.argmin(np.abs(pos[i]-where))
        out[i] = to[idx]
    out = np.reshape(out, shape)
    return out


def to_thrsh(pos):
    neg = pos >= 100
    out = match_pos(pos, 100-all_perc, all_thrsh)
    out[neg] = 0
    return out


def to_occ(pos):
    neg = pos == 0
    out = match_pos(pos, all_thrsh, 100-all_perc)
    out[neg] = 100
    return out

def times_three(x):
    return 3*x

def dev_by_three(x):
    return x/3



def plot_precep_statistic():
    # # figure 2, precipitation statistics
    # load and average the data
    thrsh_perc = xr.load_dataset('../data/percentile_threshold.nc')
    all_perc = thrsh_perc.percentile.values
    all_thrsh = thrsh_perc.thrsh.values
    all_occ = 100 - all_perc
    occ = np.array([0.25, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 34])#100])
    thrsh = np.array([15.358784165382353, 6.546857919692981, 4.558161976337443,
                      3.4287931823730475, 2.6827078282833097, 2.1295418691635,
                      1.3871746397018434, 0.9479302614927281, 0.669832711219789,
                      0.35095661997795163, 0.19140626490116194, 0.11011233389377595,
                      0.0668206149339677, 0.042710336297750465, 0.02781107142567632,
                      0.016654083877801935, 0.009048837069422012, 0.0])
    
    above_thrsh_ds = xr.load_dataset('../data/above_thrsh.nc')
    above_thrsh = above_thrsh_ds.above_thrsh.values
    above_thrsh = np.array([above_thrsh[:,:,i].flatten() for i in range(len(thrsh))])
    mean_above = np.mean(above_thrsh, axis=1)
    min_above = np.min(above_thrsh, axis=1)
    max_above = np.max(above_thrsh, axis=1)

    thrsh_idx = [1, 7, 15]

    fig, ax = plt.subplots(1,2,figsize=(2*SCALE*6.4,SCALE*4.8))

    ax[0].plot(all_occ, all_thrsh, color='k', linewidth=1)
    ax[0].scatter(occ, match_pos(occ, all_occ, all_thrsh), color='k',
              zorder=5, label='percentiles', marker='+')
    for idx in thrsh_idx:
        ax[0].text(occ[idx]*1.1, thrsh[idx]*1.2, '{:.0f}'.format(100-occ[idx]))
    ax[0].semilogy()
    ax[0].semilogx()
    ax[0].set_ylim(10e-5)
    ax[0].legend()
    ax[0].set_xlabel('Frequency of Occurence [%]')
    ax[0].set_ylabel('Rain Rate [mm/h]')
        
    ax[1].plot(occ[:], mean_above[:], linestyle=stylelist[0], color=vir4[0], label='mean')
    ax[1].plot(occ[:], min_above[:], linestyle=stylelist[3], color=vir4[3], label='min & max')
    ax[1].plot(occ[:], max_above[:], linestyle=stylelist[3], color=vir4[3])
    ax[1].vlines(ymin=0, ymax=60, x=occ[thrsh_idx], linestyle='dashed', color='k', linewidth=1)
    for idx in thrsh_idx:
        ax[1].text(occ[idx]+0.25, 34, '{:.0f}'.format(100-occ[idx]))
            
    ax[1].set_xlim(0, 33)
    ax[1].legend()
    #ax[1].semilogx()
    ax[1].set_xlabel('Frequency of Occurence [%]')
    ax[1].set_ylabel('Spatial Coverage [%]')

    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    secax = ax[1].secondary_xaxis('top', functions=(to_thrsh, to_occ))
    secax.set_xticks([10, 1, 0.1, 0.01], ['10', '1', '0.1', '0.01'])
    secax.set_xlabel('Rain Rate [mm/h]')
            
    fig.tight_layout()
    
    plt.savefig('plots/figure2.png')
    plt.savefig('plots/figure2.pdf')
    plt.close()


def plot_FSS_variants():
    # # figure 4, fss variants dependence on thrsh and neighbourhood size
    # load data
    fss_collection = xr.open_dataset('../data/fss_comparison.nc')

    thrsh = fss_collection.thrsh.values
    window = fss_collection.window.values
    ens_size = fss_collection.ens_size.values
    
    fss_fbs = fss_collection.FSSsum.mean('timestamp').values
    fss_mean = fss_collection.FSSmean.mean('timestamp').values
    fss_single = fss_collection.FSSprob.sel(ens_size=1).mean('timestamp').values
    fss_prob = fss_collection.FSSprob.mean('timestamp').values

    occ = np.array([0.25, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 34])#100])

    single_thrsh = 7
    single_ens = 5
    single_window = 1
        
    fig, ax = plt.subplots(1,2,figsize=(2*SCALE*6.4,SCALE*4.8))

    ax[0].plot(window*3, fss_single[single_thrsh,:], label='FSSsingle', color=mgm4[0], linestyle=stylelist[0])
    ax[0].plot(window*3, fss_prob[single_ens,single_thrsh,:], label='FSSprob', color=mgm4[1], linestyle=stylelist[1])
    ax[0].plot(window*3, fss_mean[single_ens,single_thrsh,:], label='FSSmean', color=mgm4[2], linestyle=stylelist[2])
    ax[0].plot(window*3, fss_fbs[single_ens,single_thrsh,:], label='FSSsum', color=mgm4[3], linestyle=stylelist[3])
    ax[0].legend()
    ax[0].set_xlabel('Neighborhood [km]')
    ax[0].set_ylabel('FSS')
    ax[0].set_xticks(np.arange(0, 250, 50)*3)
    ax[0].set_title('n = {:}, c = {:.1f} mm/h, f = {:.0f} %'.format(ens_size[single_ens], thrsh[single_thrsh], occ[single_thrsh]))

    secax = ax[0].secondary_xaxis('top', functions=(dev_by_three, times_three))
    secax.set_xticks(ax[0].get_xticks()/3)
    secax.set_xlabel('Grid Points')


    ax[0].sharey(ax[1])
    ax[1].plot(occ[:], fss_single[:,single_window], label='FSSsingle', color=mgm4[0], linestyle=stylelist[0])
    ax[1].plot(occ[:], fss_prob[single_ens,:,single_window], label='FSSprob', color=mgm4[1], linestyle=stylelist[1])
    ax[1].plot(occ[:], fss_mean[single_ens,:,single_window], label='FSSmean', color=mgm4[2], linestyle=stylelist[2])
    ax[1].plot(occ[:], fss_fbs[single_ens,:,single_window], label='FSSsum', color=mgm4[3], linestyle=stylelist[3])
    #plt.legend()
    ax[1].set_xlabel('Frequency of Occurence [%]')
    ax[1].set_ylabel('FSS')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].set_xlim(0,22)
    ax[1].set_ylim(0,1)
    ax[1].set_title('n = {:}, g = {:}, l = {:} km'.format(ens_size[single_ens], window[single_window], window[single_window]*3))

    secax = ax[1].secondary_xaxis('top', functions=(to_thrsh, to_occ))
    secax.set_xticks([10, 1, 0.1, 0.01], ['10', '1', '0.1', '0.01'])
    secax.set_xlabel('Rain Rate [mm/h]')
    
    fig.tight_layout()
    plt.savefig('plots/figure4.png')
    plt.savefig('plots/figure4.pdf')
    plt.close()



    # # figure 5, fss variants dependence on ensemble size
    # for mediate window size and Rain Rate


    fig, ax = plt.subplots(1,2,figsize=(2*SCALE*6.4,SCALE*4.8))

    thr_idx = 7
    win_idx = 3

    ax[0].hlines(xmin=1, xmax=np.max(ens_size), y=fss_single[thr_idx,win_idx], label='FSSsingle', color=mgm4[0], linestyle=stylelist[0])
    ax[0].plot(ens_size, fss_prob[:,thr_idx,win_idx], label='FSSprob', color=mgm4[1], linestyle=stylelist[1])
    ax[0].plot(ens_size, fss_mean[:,thr_idx,win_idx], label='FSSmean', color=mgm4[2], linestyle=stylelist[2])
    ax[0].plot(ens_size, fss_fbs[:,thr_idx,win_idx], label='FSSsum', color=mgm4[3], linestyle=stylelist[3])
    ax[0].set_ylim(0)
    ax[0].legend()
    ax[0].set_xlabel('Ensemble Size')
    ax[0].set_ylabel('FSS')
    ax[0].grid(True)
    ax[0].yaxis.tick_left()
    ax[0].yaxis.set_label_position("left")
    ax[0].semilogx()
    ax[0].set_title('g={:}, l={:} km, c={:.1f} mm/h, f={:.0f} %'.format(window[win_idx], window[win_idx]*3, thrsh[thr_idx], occ[thr_idx]))


    thr_idx = 1
    win_idx = 5


    ax[0].sharey(ax[1])
    ax[1].hlines(xmin=1, xmax=np.max(ens_size), y=fss_single[thr_idx,win_idx], label='FSSsingle', color=mgm4[0], linestyle=stylelist[0])
    ax[1].plot(ens_size, fss_prob[:,thr_idx,win_idx], label='FSSprob', color=mgm4[1], linestyle=stylelist[1])
    ax[1].plot(ens_size, fss_mean[:,thr_idx,win_idx], label='FSSmean', color=mgm4[2], linestyle=stylelist[2])
    ax[1].plot(ens_size, fss_fbs[:,thr_idx,win_idx], label='FSSsum', color=mgm4[3], linestyle=stylelist[3])
    #ax[1].legend()
    ax[1].set_xlabel('Ensemble Size')
    ax[1].set_ylabel('FSS')
    ax[1].grid(True)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].set_ylim(0,1)
    ax[1].semilogx()
    ax[1].set_title('g={:}, l={:} km, c={:.1f} mm/h, f={:.0f} %'.format(window[win_idx], window[win_idx]*3, thrsh[thr_idx], occ[thr_idx]))

    plt.tight_layout()
    plt.savefig('plots/figure5.png')
    plt.savefig('plots/figure5.pdf')
    plt.close()

def plot_FSSprob():
    # # figure 6, fss prob dependence on ensemble size
    # for window sizes and Rain Rate

    fss_collection = xr.open_dataset('../data/fss_no_thrsh_ens_mean.nc')

    thrsh = fss_collection.thrsh.values
    window = fss_collection.window.values
    ens_size = fss_collection.ens_size.values

    mean_fss = fss_collection.mean_fss.mean('timestamp').values

    thrsh_perc = xr.load_dataset('../data/percentile_threshold.nc')
    all_perc = thrsh_perc.percentile.values
    all_thrsh = thrsh_perc.thrsh.values
    perc = np.empty(len(thrsh))

    for i in range(len(thrsh)):
        free_idx = np.argmin(np.abs(thrsh[i]-all_thrsh))
        perc[i] = all_perc[free_idx]

    occ = 100 - perc
    occ[-1] = 34

    thrsh_idx = [1, 7, 15]
    win_idx = [0, 10, -1]
    single_thrsh = 7
    single_win = 10

    fig, ax = plt.subplots(1, 2, figsize=(SCALE*2*6.4, SCALE*4.8))

    for i, t in enumerate(thrsh_idx):
        label = 'c={:.2f} mm/h, f={:.0f} %, FSS$_{:}$={:.2f}'.format(thrsh[t], occ[t], '{960}', mean_fss[-1,t,single_win])
        ax[0].plot(ens_size, mean_fss[:,t,single_win]/mean_fss[-1,t,single_win], color=blues3[i], linestyle=stylelist[i], label=label)
    ax[0].legend(fontsize=9)
    ax[0].set_ylim(0)
    ax[0].grid(True)
    ax[0].set_xlabel('Ensemble Size')
    ax[0].set_ylabel('FSS$_\mathrm{n}$ / FSS$_{960}$')
    ax[0].set_title('g = {:d}, l = {:} km'.format(window[single_win], window[single_win]*3))
    ax[0].semilogx()

    ax[0].sharey(ax[1])
    for i, w in enumerate(win_idx):
        label = 'g={:d}, l={:.0f} km, FSS$_{:}$={:.2f}'.format(window[w], window[w]*3, '{960}', mean_fss[-1,single_thrsh,w])
        ax[1].plot(ens_size, mean_fss[:,single_thrsh,w]/mean_fss[-1,single_thrsh,w], color=greens3[i], linestyle=stylelist[i], label=label)
    ax[1].legend(fontsize=9)
    ax[1].set_ylim(0.7, 1.01)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].set_xlabel('Ensemble Size')
    ax[1].set_ylabel('FSS$_\mathrm{n}$ / FSS$_{960}$')
    ax[1].set_title('c = {:.2f} mm/h, f = {:} %'.format(thrsh[single_thrsh], occ[single_thrsh]))
    ax[1].semilogx()

    plt.tight_layout()
    plt.savefig('plots/figure6.png')
    plt.savefig('plots/figure6.pdf')
    plt.close()




def plot_LFSSprob():
    # # figure 7, belivable scale prob dependence on ensemble size
    # for window sizes and Rain Rate
    
    fss_collection = xr.open_dataset('../data/fss_no_thrsh_ens_mean.nc')

    thrsh = fss_collection.thrsh.values
    window = fss_collection.window.values
    ens_size = fss_collection.ens_size.values

    
    mean_fss = fss_collection.mean_fss.mean('timestamp').values

    #all_thrsh, all_perc = np.load("../data/percentile_no_thrsh_random.npy")
    thrsh_perc = xr.load_dataset('../data/percentile_threshold.nc')
    all_perc = thrsh_perc.percentile.values
    all_thrsh = thrsh_perc.thrsh.values
    perc = np.empty(len(thrsh))

    for i in range(len(thrsh)):
        free_idx = np.argmin(np.abs(thrsh[i]-all_thrsh))
        perc[i] = all_perc[free_idx]

    occ = 100 - perc
    occ[-1] = 34
    

    bel_scale = xr.open_dataset('../data/bel_scale.nc')
    bel_scale = bel_scale.bel_scale

    thrsh_idx = [1, 7, 15]
    single_thrsh = 7

    fig = plt.figure(figsize=(SCALE*2*6.4, SCALE*4.8))
    gs = GridSpec(1, 4, width_ratios=(1, 0.05, 0.1, 1), figure=fig)
    gs.update(top=.95, bottom=0.05, left=0.05, right=.95, hspace=0, wspace=0.0)

    ax0 = fig.add_subplot(gs[0])
    colbar = ax0.contourf(ens_size, window, mean_fss[:,single_thrsh,:].T, cmap=cmap,
                          levels=np.arange(0,1.01,0.05))
    ax0.plot(ens_size, bel_scale[:,single_thrsh], label='LFSSprob', color='k')
    ax0.hlines(xmin=np.min(ens_size), xmax=np.max(ens_size), y=bel_scale[0,single_thrsh],
               color='k', linestyle='dashed', label='LFSS')
    ax0.legend()
    ax0.set_ylim(3,100)
    ax0.semilogx()
    ax0.set_xlabel('Ensemble Size')
    ax0.set_ylabel('Spatial Scale [km]')

    cax = fig.add_subplot(gs[1])
    fig.colorbar(colbar, cax=cax, label='FSS', ticks=np.arange(0,1.1,0.1), fraction=1)

    ax0.set_title('c = {:.2f} mm/h, f = {:} %'.format(thrsh[single_thrsh], occ[single_thrsh]))

    ax1 = fig.add_subplot(gs[3])
    #ax0.sharey(ax1)
    for i, t in enumerate(thrsh_idx):
        lbl = 'c = {:.2f} mm/h, f = {:.0f} %'.format(thrsh[t], occ[t])
        ax1.plot(ens_size, bel_scale[:,t], label=lbl, linestyle=stylelist[i], color=blues3[i])
    ax1.hlines(xmin=1, xmax=960, y=[bel_scale[-1,thrsh_idx]], ls=':', linewidth=1, color='black', zorder=0)

    ax1.legend()
    ax1.semilogx()
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_xlabel('Ensemble Size')
    ax1.set_ylabel('Skillful Scale [km]')
    #ax1.set_title('xxx')

    gs.tight_layout(fig)
    plt.savefig('plots/figure7.png')
    plt.savefig('plots/figure7.pdf')
    plt.close()

    


if __name__ == "__main__":
    plot_precep_statistic()
    plot_FSS_variants()
    plot_FSSprob()
    plot_LFSSprob()