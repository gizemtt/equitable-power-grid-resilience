# Impact of power outages depends on who loses it: Equity-informed grid resilience planning via stochastic optimization

This repository contains the code and data associated with the publication titled **"[Impact of Power Outages Depends on Who Loses It: Equity-Informed Grid Resilience Planning via Stochastic Optimization](https://doi.org/10.1016/j.seps.2024.102036)"** published in the *Journal of Socio-Economic Planning Sciences*.

## Overview

The code in this repository extends the work presented in **"[Comparisons of Two-Stage Models for Flood Mitigation of Electrical Substations](https://doi.org/10.1287/ijoc.2023.0125)"** by Austgen et al., building on their DC power flow formulation.

If you use this code in your research, please make sure to cite both publications.

## Utilization of the Code

1. **Running the Code**:  
   To execute the main code, open and run `MainCode.ipynb`.

2. **Changing the Objective Function**:  
   To switch between different objective functions (LSO, EIP, EID models), navigate to `powerutils/models/shortterm.py`. You can select the desired objective function by modifying the code between lines 197 and 209.

3. **Adjusting the Mapping Function Parameters**:  
   If you need to change the mapping function parameters, edit the values between lines 41 and 56 in `powerutils/models/shortterm.py`.

