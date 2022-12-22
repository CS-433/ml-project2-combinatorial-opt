# ml-project2-combinatorial-opt
Combinatorial Optimization with NNs

In this project we're trying to get approximate solutions to combinatorial optimization problems with autoregressive neural nets.

Authors:
Axel Andersson (axel.andersson@epfl.ch)
Alfred Clemedtson (alfred.clemedtson@epfl.ch)
Eric Dannetun (eric.dannetun@epfl.ch)

Files:

the_main.py:                    File to run experiments with two different Hamiltonians and two different neural net architechtures

conv_net_implementation.py:     File with implementation of an autoregressive convolutional network [1]

linear_autoreg_nn.py:           File with implementation of an autoregressive linear network [1]

graphs.py:                      File to generate graphs

hamiltonians.py:                File containing Hamiltonians for Traveling Salesman Problem and the Next-Nearest Neighbor problem

mean_field_model_2.py:          File with experiments of a mean-field-model



References:

[1] Classes in this file are from the paper "Solving Statistical Mechanics Using Variational Autoregressive Networks" 
(url: https://doi.org/10.48550/arXiv.1809.10606) by Dian Wu, Lei Wang and Pan Zhang. Link to their repo:https://github.com/wdphy16/stat-mech-van
