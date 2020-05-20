Code for experiments in "Post-Estimation Smoothing: a Simple Baseline for Learning with Side Information". 

This repo is in progress, and will continue to be updated (details below). Stay tuned!

1. Simulation experiments.
Code for the simulation results is in the folder pes/simulations. 
	- sim_fxns.py houses the main functions needed to run the simulations
	- simulations_tls.ipynb and simulations_ols.ipynb run the simulations and produce the figures in the paper. 
	The only difference between these notebooks is that simualtions_tls using the total least squares solve to 
	instantiate unsmoothed predictions, whereas simulations_ols using ordinary least squares.

If you'd like to save figures (by setting the flag save_fig = True in the notebooks), make sure you have the following directory: p-es/figs_output/simulations. 

2. Video experiments.
(coming soon, please contact me if you're interested in using the analysis sooner!)


3. Housing experiments.
(coming soon, subject to data availability, please contact me if you're interested in using the analysis sooner!)



