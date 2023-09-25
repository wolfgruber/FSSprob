Python3 and Fortran90 code to compute the probabilistic Fractions Skill Score (FSS). This code was used to produce results for the study "The fractions skill score for ensemble forecast verification". The code builds on an algorithm by [Faggian et al. (2015)](https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/555) which contained a small bug.

The branch `main` contains the FSSprob code, the data from the associated article and python scripts to plot the data. Users only interested in the FSSprob code might want to check out the branch `code_only`.

Please cite as [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8082583.svg)](https://doi.org/10.5281/zenodo.8082583)

# Usage
## Computing
The folder `code` contains the Python and Fortran scripts which contain functions and subroutines for calculating the FSS for deterministic and probabilistic forecasts using a fast lookup table method instead of convolutions. The Python code can be imported from `code/compute_FSS.py`.

The Fortran code is written so it can be compiled into a python module using numpys `f2py`. You can compile it by running

    f2py -c -m fss90 mod_fss.f90 --f90flags="-O3"
In python include the module with

    from fss90 import mod_fss
A usecase of the `f2py`-compiled Fortran code can be seen in the script `code/fss_script.py`. To first compile the Fortran module and run the python script you can use the bash script `code/compile_and_run.sh`

The file `code/compare_fss_flavors.py` contains python functions to compute the different variants of the probabilistic FSS.

## Data
The folder `data` contains the results in NetCDF format which are presented in the accompanying research article (yet to be published).

## Plotting
The folder `plotting` contains two python scripts to recreate the figures of the research article. `plotting/plotting_graphs.py` displays the data stored in `data`. `plotting/plotting_maps.py` recreates the precipitation maps. Please note that the raw forecast files are not included in the dataset.
