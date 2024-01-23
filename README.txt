This program computes coefficients of the azimuthal distribution of a lepton pair in the case of a photon-photon collision.
The analytical expresion are given in arXiv:1903.10084 eq. (3) to (6).
The from factor expression come from arXiv:2207.03012 eq. (17)
Author: Nicolas (nicolas.crepet@ens-paris-saclay.fr)
Date: January 2023

List of files:
smearing.py: Main file to use, edit a .lhe file given in input to compute the correct azimuthal distribution. This code was originally written
             by Hua-Sheng Sao
grid_1D.py and grid_3D.py: Contain the grid class, used for the integration
coefficient_eval.py: Compute the coefficient with integrals
coefficient.py: Compute the coefficient with the grid if they exist
compute_grid_HTCONDOR.py and construct_grid_HTCONDOR.py: Allow to compute the grids of coefficient using HTCONDOR.


