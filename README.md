# ML_2D_Turbulence
LES-ML closures for Kraichnan turbulence

# Build status (TravisCI)
[![Build Status](https://travis-ci.com/Romit-Maulik/ML_2D_Turbulence.svg?token=JqpUnmxjwxTCTVyKXrpy&branch=master)](https://travis-ci.com/Romit-Maulik/ML_2D_Turbulence)

# References
1. Subgrid modelling for two-dimensional turbulence using neural networks, J. Fluid Mech., 858, 122-144, 2019.
2. Sub-grid scale model classification and blending through deep learning, J. Fluid Mech., 870, 784-812, 2019.
3. Data-driven deconvolution for large eddy simulations of Kraichnan turbulence, Phys. Fluids, 30(12), 125109, 2018.
4. A stable and scale-aware dynamic modeling framework for subgrid-scale parameterizations of two-dimensional turbulence, Comput. Fluids, 158, 11-38, 2017.

# Quick start
Set `closure_choice` variable within ML_2D_Turbulence.py file to test performance in a-posteriori for Kraichnan turbulence.

# Turbulence model classifier training
As an example - the Keras code for training the turbulence model classification framework and data (reference 2) is provided in `Model_Classifier_Network`
