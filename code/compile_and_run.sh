#!/bin/bash
f2py -c -m fss90 mod_fss.f90 --f90flags="-O3"
python3 fss_script.py
