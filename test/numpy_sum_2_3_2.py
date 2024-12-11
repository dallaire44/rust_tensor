#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:59:25 2023

@author: dd
"""

"""
Usage: analyse_data.py --company=<company>
"""
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from hmmlearn._hmmc import GaussianHMM
#import hmmlearn._hmmc as _hmmc

 
#args = docopt(doc=__doc__, argv=None, help=True,
#              version=None, options_first=False)
 
# Supress warning in hmmlearn

def DD_log_norme_density(means, covars, tr):
    print("-----start")
    #print(tr)
    #print(tr.shape)
    tr_reshaped = tr.reshape(tr.shape[0], 1 , tr.shape[1])
    #print(tr_reshaped)
    #print(means)
    tr_minus_means = tr_reshaped - means
    #print(tr_minus_means)
    #print(tr_minus_means.shape)
    tr_minus_means_sqrt = tr_minus_means ** 2
    #print(tr_minus_means_sqrt)
    
    #print(covars)
    div = tr_minus_means_sqrt / covars
    print(div)

    div_sum = div.sum(2)
    print(div_sum)
def adfas():
    print(div_sum.shape)
    log_covars = np.log(covars).sum(1)
    print(log_covars)
    print(log_covars.shape)
    log_x = div_sum + log_covars
    print(log_x)
    nc, nf = means.shape
    term1 = nf * np.log(2 * np.pi)
    print(term1)
    term1_log_x = log_x + term1
    print(term1_log_x)
    final = term1_log_x * -0.5
    print(final)
    #[[  6.53336746  11.63653869   6.25775325   7.8762887 ]
    # [  7.98789289  11.67420863   6.46739582   7.41056561]
    # [ 11.09300998   8.88768249   7.3913026    5.28846944]
    # [  8.89741599  11.74641634   6.71733817   7.23048704]
    # [  7.9407728   12.27024921   6.80549792   8.28292971]
    # [  6.93658269  12.33158811   6.57050462   8.53863301]
    # [  5.93075781  11.83386494   6.33918805   8.39726311]
    # [  8.0995322   11.87899576   6.54275636   7.59964269]
    # [  7.1505109   12.35741404   6.65582783   8.40769708]
    # [-72.85038826 -79.44852455  -4.1397033   -3.65512832]]
    
def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model."""
    # X: (ns, nf); means: (nc, nf); covars: (nc, nf) -> (ns, nc)
    nc, nf = means.shape
    # Avoid 0 log 0 = nan in degenerate covariance case.
    covars = np.maximum(covars, np.finfo(float).tiny)
    with np.errstate(over="ignore"):
        return -0.5 * (nf * np.log(2 * np.pi)
                       + np.log(covars).sum(axis=-1)
                       + ((X[:, None, :] - means) ** 2 / covars).sum(axis=-1))    
    


"""
[[[1 2]
  [2 3]
  [2 2]]

 [[4 5]
  [6 7]
  [3 3]]]
(2, 3, 2)
(48, 16, 8)
6 2 1
[1 2 2 3 2 2 4 5 6 7 3 3]
--------------- dim 2

[[ 3  5  4]
 [ 9 13  6]]

[1 2 2 3 2 2 4 5 6 7 3 3]
  3   5   4   9  13   6
(2, 3)
(24, 8)
3  1  -- 3 1 0
[ 3  5  4  9 13  6]
---------------dime 1

[1 2 2 3 2 2 4 5 6 7 3 3]
   5     7    15    13


[[ 5  7]
 [13 15]]
(2, 2)
(16, 8)
2 1 -- 2 0 1  
[ 5  7 13 15]
---------------dim 0
[1 2 2 3 2 2 4 5 6 7 3 3]
 \    5      \ 
   \    7      \
     \    8      \
       \    10     \
         \    5      \
           \    5      \  
[[ 5  7]
 [ 8 10]
 [ 5  5]]
(3, 2)
(16, 8)
2 1   -- 0 2 1
[ 5  7  8 10  5  5]
"""  
def test_sum():
    it = np.array([
                    [
                    [1,2],
                    [2,3],
                    [2,2]
                    ],
                    [
                        [4,5],
                        [6,7],
                        [3,3]
                    ]
                    ])
    print(it)
    print(it.shape)
    print(it.strides)
    print(it.ravel())
    print("--------------- dim 2")
    print(it.sum(2))
    print(it.sum(2).shape)
    print(it.sum(2).strides)
    print(it.sum(2).ravel())
    print("---------------dime 1")
    print(it.sum(1))
    print(it.sum(1).shape)
    print(it.sum(1).strides)
    print(it.sum(1).ravel())
    print("---------------dim 0")
    print(it.sum(0))
    print(it.sum(0).shape)
    print(it.sum(0).strides)
    print(it.sum(0).ravel())

    
    
if __name__ == "__main__":
    test_sum()
    print("---------")
    #startprob
    #array([1.60061161e-052, 2.41032748e-115, 5.82981162e-031, 1.00000000e+000])
    startprob = np.array([1.60061161e-052, 2.41032748e-115, 5.82981162e-031, 1.00000000e+000])
    #print(startprob)    
    #print(np.log(startprob))
    #print("----------------")
    means = np.array([[0.01585097,0.02186496,0.00392907],
                    [-0.00151173,0.0064778,0.0086702],
                    [0.02525463,0.04286314,0.01473968],
                    [-0.02334926,0.00540885,0.03255712]])
    
    covars = np.array([[8.27228701e-05, 7.06533771e-05, 3.19888556e-05], 
                       [5.38019282e-05, 2.54458966e-05, 3.60805308e-05],
                       [7.37036162e-04, 4.92512751e-04, 2.55521536e-04],
                       [2.49777141e-04, 5.34307035e-05, 2.56926942e-04]])
    
    #print(np.log(transmat))
    tr = np.array([[-0.00239948,0.00021593,0.00539888],
                                [ 0.0007117,0.00322084,0.00190604],
                                [ 0.01125161,0.01402525,0],
                                [ 0.00386113,0.00459483,0.00279541],
                                [ -0.00490723,0.00830631,0.00573307],
                                [ -0.00543003,0.00479339,0.00713592],
                                [ -0.00397356,0.00063415,0.00899437],
                                [ -0.00079964,0.00479768,0.00196274],
                                [ -0.00164908,0.00329447,0.00932446],
                                [ -0.1,0,0],

        ]) 
    
    #print(_log_multivariate_normal_density_diag(tr, means, covars))
    #tr2 = np.array([[1,1,1],
    #                [ 1,1,1],
    #                [ 1,1,1],
    #                [ 1,1,1]])

    #means2 = np.array([[1,1,1],
    #                [2,2,2],
    #                [3,3,3]])
                    #[4,4,4]])


    #print(tr2.shape)
    #tr2_r = tr2.reshape(4,1,3);
    #print(tr2_r)
    #print("------------")
    #res = tr2_r - means2
    #print(res)
    #print(res.shape)
    
    
    
    #DD_log_norme_density(means, covars, tr)
    #print(dd_log_probij)
    #print(dd_fwd)
    #print('-------')

    #print(log_probij)
    #print(fwdlattice)

    #_hmmc_forward.forward_log()