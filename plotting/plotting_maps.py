#!/usr/bin/env python
# coding: utf-8

"""
Created on  14.06.2023
Last update 14.06.2023

author: Ludwig Wolfgruber
e-mail: ludwig.wolfgruber@gmx.at
"""

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import xarray as xr

import cartopy
import cartopy.crs as crs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
from scipy import signal

# defining styles
# discrete colormaps
cmap = plt.cm.viridis
vir3 = cmap(np.linspace(0.8,0.2,3))
vir4 = cmap(np.linspace(0.8,0.2,4))
vir1 = cmap(0.5)

# linestyles
stylelist = ['solid', 'dashdot', 'dashed', 'dotted']

# for determing font and ticks size in figures:
# the smaller SCALE, the larger fonts are
# compared to the figure
SCALE = 0.6

plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = True

base = plt.cm.get_cmap('Blues')
dBlues = [base(i/10) for i in range(10)]
dBlues[0] = (1.0, 1.0, 1.0, 1.0)

def split_map(h=0.1, base_cmap=plt.cm.Blues):
    base = plt.cm.get_cmap(base_cmap)
    N = 100
    color_list = np.append(np.linspace(0, 0.5-h, N//2), np.linspace(0.5+h, 1, N//2))
    color_list = base(color_list)
    cmap_name = 'split' + base.name
    return base.from_list(cmap_name, color_list, N)

splitB = split_map(h=0.1, base_cmap=plt.cm.Blues)


def plot_precep_map():
    # Plot precipitaion with special colorbar
    clevs = [0.1,0.2,0.5,1,2,5,10,20,50]
    colors = ("#ffffff","#F0EABB","#C0E69F","#72E1A2","#00D8BC","#00C7DC","#00AAF2","#847CF2","#BA3CD0","#B40087")
    projection = crs.Mercator()
    transform = crs.PlateCarree()
    
    fig = plt.figure(figsize=(SCALE*4*4.8*0.9, SCALE*1*6*0.9))
    gs = GridSpec(4, 5, height_ratios=(0.025,1,1,0.025), width_ratios=(1, 1, 1, 1, 0.1), figure=fig)
    gs.update(top=.95, bottom=0.05, left=0.05, right=.95, hspace=0, wspace=0)

    # plot minimum ensemble Member
    ax0 = fig.add_subplot(gs[:,0], projection=projection)
    ax0.set_facecolor('lightgrey')
    data = xens['prec'].sel(ens=imin+1, time='2016-05-29T18:00:00.000000000') * 3600.
    ax0.contourf(s_lon, s_lat, data, clevs, colors=colors, extend='both', transform=transform)
    
    # Customize fig, axis and labels
    ax0.coastlines(resolution='50m')
    ax0.add_feature(cfeature.BORDERS)
    ax0.set_extent([4, 16, 45.4, 55.4])
    gl = ax0.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.bottom_labels = True
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([46, 49, 52, 55])
    ax0.set_title('Minimum Member')

    # plot maximum ensemble Member
    ax1 = fig.add_subplot(gs[:,1], projection=projection)
    ax1.set_facecolor('lightgrey')
    data = xens['prec'].sel(ens=imax+1, time='2016-05-29T18:00:00.000000000') * 3600.
    ax1.contourf(s_lon, s_lat, data, clevs, colors=colors, extend='both', transform=transform)
    
    # Customize fig, axis and labels
    ax1.coastlines(resolution='50m')
    ax1.add_feature(cfeature.BORDERS)
    ax1.set_extent([4, 16, 45.4, 55.4])
    gl = ax1.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([46, 49, 52, 55])
    ax1.set_title('Maximum Member')

    # plot 40-ensemble mean
    ax2 = fig.add_subplot(gs[:,2], projection=projection)
    ax2.set_facecolor('lightgrey')
    data = np.mean(xens['prec'].sel(ens=slice(1,40), time='2016-05-29T18:00:00.000000000') * 3600., axis=0)
    ax2.contourf(s_lon, s_lat, data, clevs, colors=colors, extend='both', transform=transform)
    
    # Customize fig, axis and labels
    ax2.coastlines(resolution='50m')
    ax2.add_feature(cfeature.BORDERS)
    ax2.set_extent([4, 16, 45.4, 55.4])
    gl = ax2.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([46, 49, 52, 55])
    ax2.set_title('40 Member Mean')
    
    # plot 960-ensemble mean
    ax3 = fig.add_subplot(gs[:,3], projection=projection)
    ax3.set_facecolor('lightgrey')
    data = np.mean(xens['prec'].sel(ens=slice(1,960), time='2016-05-29T18:00:00.000000000') * 3600., axis=0)
    colbar = ax3.contourf(s_lon, s_lat, data, clevs, colors=colors, extend='both', transform=transform)
    
    # Customize fig, axis and labels
    ax3.coastlines(resolution='50m')
    ax3.add_feature(cfeature.BORDERS)
    ax3.set_extent([4, 16, 45.4, 55.4])
    gl = ax3.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([46, 49, 52, 55])
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([46, 49, 52, 55])
    ax3.set_title('960 Member Mean')
    
    cax = fig.add_subplot(gs[1:3,4])
    fig.colorbar(colbar, cax=cax, label='[mm/h]')
    gs.tight_layout(fig)

    plt.savefig('plots/figure3a_row.png')
    plt.savefig('plots/figure3a_row.pdf')
    plt.clear()


def plot_probabilities():
    # plotting 
    thrsh = 0.9479302614927281 # 5% frequency of ocuurence
    projection = crs.Mercator()
    transform = crs.PlateCarree()
    
    fig = plt.figure(figsize=(SCALE*4*4.8*0.9, SCALE*1*6*0.9))
    gs = GridSpec(4, 5, height_ratios=(0.025,1,1,0.025), width_ratios=(1, 1, 1, 1, 0.1), figure=fig)
    gs.update(top=.95, bottom=0.05, left=0.05, right=.95, hspace=0, wspace=0)
    
    
    # plot one ensemble Member with thrsh
    ax0 = fig.add_subplot(gs[:,0], projection=projection)
    ax0.set_facecolor('lightgrey')
    data = xens['prec'].sel(ens=1, time='2016-05-29T18:00:00.000000000') * 3600. > thrsh
    ax0.contourf(s_lon, s_lat, data, colors=dBlues, levels=np.arange(0,1.1,0.1), transform=transform)
    
    # Customize fig, axis and labels
    ax0.coastlines(resolution='50m')
    ax0.add_feature(cfeature.BORDERS)
    ax0.set_extent([4, 16, 45.4, 55.4])
    gl = ax0.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.bottom_labels = True
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([46, 49, 52, 55])
    ax0.set_title('BP, n=1')


    # plot 10-ensemble mean with thrsh
    ax1 = fig.add_subplot(gs[:,1], projection=projection)
    ax1.set_facecolor('lightgrey')
    data = np.mean(xens['prec'].sel(ens=slice(1,40), time='2016-05-29T18:00:00.000000000') * 3600. > thrsh, axis=0)
    colbar = ax1.contourf(s_lon, s_lat, data, colors=dBlues, levels=np.arange(0,1.1,0.1), transform=transform)
    
    # Customize fig, axis and labels
    ax1.coastlines(resolution='50m')
    ax1.add_feature(cfeature.BORDERS)
    ax1.set_extent([4, 16, 45.4, 55.4])
    gl = ax1.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([46, 49, 52, 55])
    ax1.set_title('EP, n=40')
    
    
    # plot 40-ensemble mean after thrsh
    ax2 = fig.add_subplot(gs[:,2], projection=projection)
    ax2.set_facecolor('lightgrey')
    data = xens['prec'].sel(ens=1, time='2016-05-29T18:00:00.000000000') * 3600. > thrsh
    data = signal.convolve2d(data, np.ones((9,9)), mode='same') / 81
    ax2.contourf(s_lon, s_lat, data, colors=dBlues, levels=np.arange(0,1.1,0.1), transform=transform)

    # Customize fig, axis and labels
    ax2.coastlines(resolution='50m')
    ax2.add_feature(cfeature.BORDERS)
    ax2.set_extent([4, 16, 45.4, 55.4])
    gl = ax2.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([46, 49, 52, 55])
    ax2.set_title('NP, n=1, g=9, l=27 km')
    

    # plot 40-ensemble mean after thrsh
    ax3 = fig.add_subplot(gs[:,3], projection=projection)
    ax3.set_facecolor('lightgrey')
    data = np.mean(xens['prec'].sel(ens=slice(1,40), time='2016-05-29T18:00:00.000000000') * 3600. > thrsh, axis=0)
    data = signal.convolve2d(data, np.ones((9,9)), mode='same') / 81
    ax3.contourf(s_lon, s_lat, data, colors=dBlues, levels=np.arange(0,1.1,0.1), transform=transform)
    
    # Customize fig, axis and labels
    ax3.coastlines(resolution='50m')
    ax3.add_feature(cfeature.BORDERS)
    ax3.set_extent([4, 16, 45.4, 55.4])
    gl = ax3.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([46, 49, 52, 55])
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([46, 49, 52, 55])
    ax3.set_title('NEP, n=40, g=9, l=27 km')

    cax = fig.add_subplot(gs[1:3,4])
    fig.colorbar(colbar, cax=cax, label='Probability []', ticks=np.arange(0,1.1,0.1))
    gs.tight_layout(fig)
    
    plt.savefig('plots/figure3b_row.png')
    plt.savefig('plots/figure3b_row.pdf')
    plt.clear()


if __name__ == "__main__":
    # load data
    ens_path='/path/to/ensemble/data/20160529120000/fcstprec/member*.nc'
    print ('Read from forecast-dir:', ens_path)
    xens = xr.load_dataset(ens_path, member_by_filename="member(.*?).nc", in_memory=False)

    var_list = xens.data_vars.keys()
    print ('Variables:', var_list)

    s_lat=np.load('/path/to/data/lat2d_high_resolution.npy')
    s_lon=np.load('/path/to/data/lon2d_high_resolution.npy')

    xens['prec'].sel(time='2016-05-29T18:00:00.000000000').max().values * 3600

    summed_precep = xens['prec'].sel(time='2016-05-29T18:00:00.000000000').sum('lat').sum('lon').values * 3600
    imin = np.argmin(summed_precep)
    imax = np.argmax(summed_precep)
    
    plot_precep_map()
    plot_probabilities()