#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:30:39 2023

@author: Ludwig Wolfgruber
e-mail: ludwig.wolfgruber@gmx.at
"""

import numpy as np
import compute_FSS as cf


"""def fss_prob(fcst, obs, thrsh, window):
    '''
    Apply probabilisic FSS on a list of forecast - observation data, with
    ensemble size given in ens_size and different thresholds given in thrsh.
    The ensebles are prepared that before taking the average, the members are
    compared with the threshold.

    Parameters
    ----------
    fcst : xarray
        forecast to verify. In each forecast the ensemble members are arranged
        on the first axis.
    obs : xarray
        observation to verify the forecast.
    thrsh : np.array
        thresholds for which the FSS is computed.
    window : np.array
        windows (neighbourhoods, kernel sizes) for which the FSS is computed.

    Returns
    -------
    fss : np.array
        FSS aranged like [forecast, ensemble size, threshold, window].
    '''
    n_thrsh = len(thrsh)
    n_window = len(window)
    
    fss = np.empty((n_thrsh, n_window))

                
    for l in range(n_thrsh):
        thr_fcst = np.mean(fcst >= thrsh[l], axis=0)
        thr_obs = obs >= thrsh[l]

                            
        fcst_cache = cf.compute_integral_table(thr_fcst)
        obs_cache = cf.compute_integral_table(thr_obs)
                    
        for m in range(n_window):
            _, _, fss[l,m] = cf.compute_fss(
                fcst=thr_fcst, obs=thr_obs, window=window[m],
                fcst_cache=fcst_cache, obs_cache=obs_cache
                )

    return fss"""

fss_prob = cf.fss_prob


def fss_sum(fcst, obs, thrsh, window):
    '''
    Apply FSSsum. This FSS flavor computes the terms num and denom for
    
       FSS = 1 - num / denom
    for each ensemble member, averages them and than computes the FSS.

    Parameters
    ----------
    fcst : xarray
        forecast to verify. In each forecast the ensemble members are arranged
        on the first axis.
    obs : xarray
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
    n_ens = fcst.shape[0]
    
    fss = np.empty((n_thrsh, n_window))
    
    num = np.empty(n_ens)
    denom = np.empty(n_ens)
                        
    for i in range(n_thrsh):
        thr_fcst = fcst >= thrsh[i]
        thr_obs = obs >= thrsh[i]
        
        fcst_cache = cf.compute_integral_table(thr_fcst)
        obs_cache = cf.compute_integral_table(thr_obs)
        
        for j in range(n_window):
            for k in range(n_ens):
                num[k], denom[k], _ = cf.compute_fss(
                    thr_fcst[k,:,:], thr_obs, window[j],
                    fcst_cache[k,:,:], obs_cache
                    )
                

            mean_num = np.mean(num)
            mean_denom = np.mean(denom)
                    
            fss[i,j] = 1 - mean_num / mean_denom

    
    return fss


def fss_mean(fcst, obs, thrsh, window):
    '''
    Compute the FSS of the ensemble averaged forecast.

    Parameters
    ----------
    fcst : xarray
        forecast to verify. In each forecast the ensemble members are arranged
        on the first axis.
    obs : xarray
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
        thr_fcst = np.mean(fcst, axis=0) >= thrsh[l]
        thr_obs = obs >= thrsh[l]

        fcst_cache = cf.compute_integral_table(thr_fcst)
        obs_cache = cf.compute_integral_table(thr_obs)
                    
        for m in range(n_window):
            _, _, fss[l,m] = cf.compute_fss(
                fcst=thr_fcst, obs=thr_obs, window=window[m],
                fcst_cache=fcst_cache, obs_cache=obs_cache
                )

    return fss