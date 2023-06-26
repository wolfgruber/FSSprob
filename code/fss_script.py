#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:05:55 2022
Last update 22.06.2023

@author: Ludwig Wolfgruber
"""

import numpy as np
import matplotlib.pyplot as plt
from fss90 import mod_fss
from compute_FSS import fss_prob, fss_det

n1 = 30
n2 = 30
N = 40
thrsh = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
kernel = np.arange(30)

np.random.seed(0)
fcst = np.random.random((N, n1, n2))
obs = np.random.randint(0, 2, (n1, n2))

# test probabilistic FSS
fss_py = fss_prob(fcst, obs, thrsh, kernel)
fss_f90 = mod_fss.fss_prob(fcst, obs, thrsh, kernel)

plt.plot(kernel, fss_py[0,:], label='python')
plt.plot(kernel, fss_f90[0,:], label='f90', linestyle='dashed')
plt.legend()
plt.grid()
plt.show()

print('prob, all close:', np.allclose(fss_py, fss_f90))

# test deterministic FSS
fss_py = fss_det(fcst[0,:,:], obs, thrsh, kernel)
fss_f90 = mod_fss.fss_det(fcst[0,:,:], obs, thrsh, kernel)

print('det,  all close:', np.allclose(fss_py, fss_f90))