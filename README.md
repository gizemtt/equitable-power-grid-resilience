# equitable-power-grid-resilience
This repository contains code and data for the publication titled "Impact of Power Outages Depends on Who Loses It: Equity-Informed Grid Resilience Planning via Stochastic Optimization" in the Journal of Socio-Economic Planning Sciences.

This code builds upon the work presented in the publication titled "Comparisons of two-stage models for flood mitigation of electrical substations" by Austgen et al., utilizing their DC power flow formulation.

If you use this code, please cite both papers.

# Utilization of the code
Run the MainCode.ipynb
If you would like to change the objective function among the options of LSO, EIP, EID models' objectives, go to powerutils -> models -> shortterm.py and choose one from line 197 to 209.
If you would like to change the mapping function parameters, go to powerutils -> models -> shortterm.py and change them in lines from 41 to 56.
