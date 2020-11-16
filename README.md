# FQH_ED
Exact diagonalization of fractional quantum Hall (FQH) based heterostructure systems. Currently supports a planar annulus geometry.
(compatible with python 3)

## Modules description

### ATLASClusterInterface
This package containts all code relevant for working with the Technion ATLAS cluster computation mainframe.

### AnnulusFQH
This is the heart of the code - creating a many-body basis and operators, as well as finding the low lying excitations and analysis of the spectra.

### DataManaging
This module has two different funtionalities:
1. Handling the bookeeping of all the different data obtained by the simulations (operator matrices, eigenvalues, etc.)
2. Representing and Visualizing the data in comprehensive ways.

### FockSpaceAnnulus
Work in progress

### IQHAnnulus
Simple analysis of the non-interacting integer quantum Hall system, usually used for testing out new ideas and operators.

### clusterScripts
All the scripts that the cluster calls and runs.


