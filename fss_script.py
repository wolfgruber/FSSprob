#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:05:55 2022

@author: Ludwig Wolfgruber
"""

import numpy as np
import matplotlib.pyplot as plt
from fss90 import mod_fss
from compute_FSS import apply_fss_mean_in_ens

n1 = 15
n2 = 15
N = 960
ens_size = np.array([10, 20, 40, 80, 160, 320, 940])
thrsh = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
kernel = np.arange(14)#np.array([1, 4, 7, 10])

fcst = np.random.random((N, n1, n2))
obs = np.random.randint(0, 2, (n1, n2))

fss_py = apply_fss_mean_in_ens(fcst, obs, ens_size, thrsh, kernel)
fss_f90 = mod_fss.ensemble_fss_one_lead_time(fcst, obs, ens_size, thrsh, kernel)

plt.plot(thrsh, fss_py[0,:,0], label='python')
plt.plot(thrsh, fss_f90[0,:,0], label='f90', linestyle='dashed')
#plt.plot(kernel, fss_py[1,1,:]-fss_f90[1,1,:], label='diff')
plt.legend()
plt.show()