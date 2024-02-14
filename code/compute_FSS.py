#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  10.03.2022
Last update 26.06.2023

author: Ludwig Wolfgruber
e-mail: ludwig.wolfgruber@gmx.at
"""


import numpy as np


# defining functions:
def compute_integral_table(field):
    '''
    Compute the integral table with cumulative sums.

    Parameters
    ----------
    field : np.array
        input field.

    Returns
    -------
    output : np.array
        cumulative sum of input in both dimensions.

    '''
    return field.cumsum(1).cumsum(0).astype(float)


def integral_filter(field, n, table=None):
    '''
    Fast summed area table version of the sliding accumulator. Corrected
    version of Faggian (2015)

    Parameters
    ----------
    field : np.array
        field on which the filter is applied.
    n : int
        kernel size.
    table : np.array, optional
        precomputed integral table. The default is None.

    Returns
    -------
    integral_table : np.array
        filtered input field.

    '''
    w = n // 2
    if w < 1:
        return field.astype(float)
    
    if table is None: # compute integral table if not provided
        table = compute_integral_table(field)
        
    r, c = np.mgrid[0:field.shape[0], 0:field.shape[1]]
    r = r.astype(int)
    c = c.astype(int)
    w = int(w)
    r0, c0 = (np.clip(r - w, 0, field.shape[0] - 1),
              np.clip(c - w, 0, field.shape[1] - 1))
    
    if n % 2 == 0: # even sized kernel
        r1, c1 = (np.clip(r + w, 0, field.shape[0] - 1),
                  np.clip(c + w, 0, field.shape[1] - 1))
        
    else: # odd sized kernel
        r1, c1 = (np.clip(r + w + 1, 0, field.shape[0] - 1),
                  np.clip(c + w + 1, 0, field.shape[1] - 1))
        
    integral_table = np.zeros(field.shape).astype(np.float64)
    integral_table += np.take(table, np.ravel_multi_index((r1, c1), field.shape))
    integral_table += np.take(table, np.ravel_multi_index((r0, c0), field.shape))
    integral_table -= np.take(table, np.ravel_multi_index((r0, c1), field.shape))
    integral_table -= np.take(table, np.ravel_multi_index((r1, c0), field.shape))
    return integral_table


def compute_fss(fcst, obs, window, fcst_cache=None, obs_cache=None):
    '''
    Compute the fraction skill score using summed area tables. Attention: input
    fields are already probabilities/frequencies, use fss_prob() for physical 
    quantities.

    Parameters
    ----------
    fcst : np.array
        forecast field, frequency of ensemble members exceeding the threshold
        for each grid point.
    obs : np.arry
        observation field, binary information if observation exceeds threshold
        on each grid point.
    window : int
        window (neighbourhood, kernel size) with wich the FSS is computed.
    fcst_cache : np.array, optional
        precomputed table of fcst, independent of window size, computed with
        compute_integral_table(). The default is None.
    obs_cache : np.array, optional
        precomputed table of obs, independent of window size, computed with
        compute_integral_table(). The default is None.

    Returns
    -------
    output : float
        FSS for given kernel size
        
    '''
    fhat = integral_filter(fcst, window, fcst_cache)
    ohat = integral_filter(obs, window, obs_cache)
    num = np.power(fhat - ohat, 2).sum()
    denom = (np.power(fhat, 2) + np.power(ohat, 2)).sum()
    
    if num == 0 and denom == 0:
        fss = 1
    else:
        fss = 1. - num / denom
    
    return num, denom, fss


def fss_prob(fcst, obs, thrsh, window):
    '''
    Apply probabilistic FSS on a pair of forecast - observation data, with
    different thresholds given in thrsh and different windows (neighbourhoods,
    kernel sizes) given in window.

    Parameters
    ----------
    fcst : numpy.array or xarray
        forecast to verify. In each forecast the ensemble members are arranged
        on the first axis.
    obs : numpy.array or xarray
        observation to verify the forecast.
    thrsh : np.array
        thresholds for which the FSS is computed.
    window : np.array
        windows (neighbourhoods, kernel sizes) for which the FSS is computed.

    Returns
    -------
    fss : np.array
        FSS aranged like [threshold, window].

    '''
    n_thrsh = len(thrsh)
    n_window = len(window)
    
    fss = np.empty((n_thrsh, n_window))

    for l in range(n_thrsh):
        thr_fcst = np.mean(fcst >= thrsh[l], axis=0)
        thr_obs = obs >= thrsh[l]
                    
        fcst_cache = compute_integral_table(thr_fcst)
        obs_cache = compute_integral_table(thr_obs)
                    
        for m in range(n_window):
            _, _, fss[l,m] = compute_fss(
                fcst=thr_fcst, obs=thr_obs, window=window[m],
                fcst_cache=fcst_cache, obs_cache=obs_cache
                )
    
    return fss


def fss_det(fcst, obs, thrsh, window):
    '''
    Apply FSS on a pair of forecast - observation data, with different
    thresholds given in thrsh and different windows (neighbourhoods, kernel
    sizes) given in window.

    Parameters
    ----------
    fcst : numpy.array or xarray
        deterministic forecast to verify.
    obs : numpy.array or xarray
        observation to verify the forecast.
    thrsh : np.array
        thresholds for which the FSS is computed.
    window : np.array
        windows (neighbourhoods, kernel sizes) for which the FSS is computed.

    Returns
    -------
    fss : np.array
        FSS aranged like [threshold, window].

    '''
    n_thrsh = len(thrsh)
    n_window = len(window)
    
    fss = np.empty((n_thrsh, n_window))

    for l in range(n_thrsh):
        thr_fcst = fcst >= thrsh[l]
        thr_obs = obs >= thrsh[l]
                    
        fcst_cache = compute_integral_table(thr_fcst)
        obs_cache = compute_integral_table(thr_obs)
                    
        for m in range(n_window):
            _, _, fss[l,m] = compute_fss(
                fcst=thr_fcst, obs=thr_obs, window=window[m],
                fcst_cache=fcst_cache, obs_cache=obs_cache
                )
    
    return fss


def apply_fss_mean_in_ens(fcst, obs, ens_size, thrsh, window):
    '''
    Apply FSS on a pair of forecast - observation data, with ensemble size
    given in ens_size and different thresholds given in thrsh. The ensebles 
    are prepared that before taking the average, the members are compared with
    the threshold. A setup like this was used to produce results for the study
    "The fractions skill score for ensemble forecastverification". Do not use
    for regular verification!

    Parameters
    ----------
    fcst : xarray
        forecast to verify. In each forecast the ensemble members are arranged
        on the first axis.
    obs : xarray
        observation to verify the forecast.
    ens_size : np.array
        Ensemble sizes for which the FSS is computed. Note that the FSS is
        averaged over samples with the same ensemble size.
    thrsh : np.array
        thresholds for which the FSS is computed.
    window : np.array
        windows (neighbourhoods, kernel sizes) for which the FSS is computed.

    Returns
    -------
    fss : np.array
        FSS aranged like [ensemble size, threshold, window].

    '''
    n_ens = len(ens_size)
    n_thrsh = len(thrsh)
    n_window = len(window)
    sample = ens_size[-1] // ens_size
    
    fss = np.empty((n_ens, n_thrsh, n_window))
                    
    for j in range(n_ens):
        fss_mean = np.empty((sample[j],n_thrsh,n_window))
        ens_start = 1
                                
        for k in range(sample[j]): # loop through ensembles of size ens_size[j]
            #temp_fcst = fcst.sel(ens=slice(ens_start, ens_start+ens_size[j]-1)).prec.values*3600.
            temp_fcst = fcst[(ens_start-1):(ens_start+ens_size[j]),:,:]
                
            for l in range(n_thrsh):
                thr_fcst = np.mean(temp_fcst >= thrsh[l], axis=0)
                thr_obs = obs >= thrsh[l]
                    
                fcst_cache = compute_integral_table(thr_fcst)
                obs_cache = compute_integral_table(thr_obs)
                    
                for m in range(n_window):
                    _, _, fss_mean[k,l,m] = compute_fss(
                        fcst=thr_fcst, obs=thr_obs,window=window[m],
                        fcst_cache=fcst_cache, obs_cache=obs_cache
                        )
                        
            ens_start = ens_start + ens_size[j]
                    
        fss[j,:,:] = np.mean(fss_mean, axis=0)
    
    return fss
