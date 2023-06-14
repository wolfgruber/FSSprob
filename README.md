Python3 and Fortran90 code to compute the probabilistic Fractions Skill Score (FSS). This code was used to produce results for the study "The fractions skill score for ensemble forecast verification". The code builds on an algorithm by Faggian et al. (2015) which contained a small bug.

# Usage
## Computing
The folder `code` contains the Python and Fortran scripts which contain functions and subroutines for calculating the FSS for deterministic and probabilistic forecasts using a fast lookup table method instead of convolutions. 

The Fortran code is written so it can be compiled into a python module using numpys `f2py`. 
