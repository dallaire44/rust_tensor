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

 
"""
[[[ 0  1]
  [ 2  3]]

 [[ 4  5]
  [ 6  7]]

 [[ 8  9]
  [10 11]]]
(3, 2, 2)
(32, 16, 8)
4, 2, 1
[ 0  1  2  3  4  5  6  7  8  9 10 11]
---------------dim 0

[[12 15]
 [18 21]]

(2, 2)
(16, 8)
2, 1
[12 15 18 21]
---------------dime 1

[[ 2  4]
 [10 12]
 [18 20]]

(3, 2)
(16, 8)
2, 1
[ 2  4 10 12 18 20]
--------------- dim 2

[[ 1  5]
 [ 9 13]
 [17 21]]

(3, 2)
(16, 8)
2,1
[ 1  5  9 13 17 21]
"""

def test_slice():
    print("test slice")
    it = np.array([
                    [
                    [0,1],
                    [2,3]
                    ],
                    [
                        [4,5],
                        [6,7]
                    ],
                    [
                        [8,9],
                        [10,11]                            
                    ]
                    ])
    print(it.shape)
    print([x/8 for x in list(it.strides)])
    print(it[1:3, :, :1])
    print("------")
    print(it[1:3, :])
    print("-----------")
    print(it[2, 0, 0])


def test_sum():
    it = np.array([
                    [
                    [0,1],
                    [2,3]
                    ],
                    [
                        [4,5],
                        [6,7]
                    ],
                    [
                        [8,9],
                        [10,11]                            
                    ]
                    ])
    print(it)
     
    
    print(it.shape)
    print(it.strides)
    print(it.ravel())
    print("---------------dim 0")
    print(it.sum(0))
    print(it.sum(0).shape)
    print(it.sum(0).strides)
    print(it.sum(0).ravel())
    print("---------------dime 1")
    print(it.sum(1))
    print(it.sum(1).shape)
    print(it.sum(1).strides)
    print(it.sum(1).ravel())
    
    print("--------------- dim 2")
    print(it.sum(2))
    print(it.sum(2).shape)
    print(it.sum(2).strides)
    print(it.sum(2).ravel())
    
    
if __name__ == "__main__":
    test_sum()
    #test_slice()
