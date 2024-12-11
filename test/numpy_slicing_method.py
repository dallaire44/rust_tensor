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
import itertools
#from hmmlearn._hmmc import GaussianHMM
import hmmlearn._hmmc as _hmmc

 


def test_slice_dim1():
    
    """
    i: 0 j: 0 x: 0
    i: 1 j: 0 x: 0
    i: 2 j: 0 x: 0
    -----------
    i: 0 j: 0 x: 1
    i: 1 j: 0 x: 1
    i: 2 j: 0 x: 1
    -----------
    i: 0 j: 1 x: 0
    i: 1 j: 1 x: 0
    i: 2 j: 1 x: 0
    -----------
    i: 0 j: 1 x: 1
    i: 1 j: 1 x: 1
    i: 2 j: 1 x: 1
    """
    
    
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
    strides_n = [x/8 for x in list(it.strides)]
    print(strides_n)
    
    rav = it.ravel()
    print(rav)
    
    print(it)
    
    
    print("------")
    print(it[:,0,0])
    print(it[:,0,0].shape)
    print("-----------")
    print(it[2, 0, 0])
    
    res = np.array([[0,0],[0,0]])
    
    for j in range(0, it.shape[1]):
        for x in range(0, it.shape[2]):
            print("-----------")
            val = 0
            for i in range(0, it.shape[0]):
                print("i: " + str(i) + " j: " + str(j) + " x: " + str(x))
                #print("i: " + str(i * strides_n[0]) + " j: " + str(j * strides_n[1]) + " x: " + str(x * strides_n[2]))
                coord = i * strides_n[0] + j * strides_n[1] + x * strides_n[2]
                #print("coord: " + str(int(coord)) + " val " + str(rav[int(coord)]))
                val += rav[int(coord)]
            #print(val)
            res[j, x] = val
            
    print(res)
    print(it.sum(0))
    print(it[:,0,0])
    print(it[:,0,1])
    print(it[:,1,0])
    print(it[:,1,1])

def test_slice_dim2():
    """
    i: 0 j: 0 x: 0
    i: 0 j: 1 x: 0
    --------
    i: 0 j: 0 x: 1
    i: 0 j: 1 x: 1
    --------
    i: 1 j: 0 x: 0
    i: 1 j: 1 x: 0
    --------
    i: 1 j: 0 x: 1
    i: 1 j: 1 x: 1
    --------
    i: 2 j: 0 x: 0
    i: 2 j: 1 x: 0
    --------
    i: 2 j: 0 x: 1
    i: 2 j: 1 x: 1
    """
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
    strides_n = [x/8 for x in list(it.strides)]
    print(strides_n)
    
    rav = it.ravel()
    print(rav)
    
    print(it)
    
    
    print("------")
    print(it[:,0,0])
    print(it[:,0,0].shape)
    print("-----------")
    print(it[2, 0, 0])
    
    res = np.array([[0,0], [0,0] ,[0,0]])


    for i in range(0, it.shape[0]):
   
        for x in range(0, it.shape[2]):
        
            val=0
            for j in range(0, it.shape[1]):
                print("i: " + str(i) + " j: " + str(j) + " x: " + str(x))
                #print("i: " + str(i * strides_n[0]) + " j: " + str(j * strides_n[1]) + " x: " + str(x * strides_n[2]))
                coord = i * strides_n[0] + j * strides_n[1] + x * strides_n[2]
                #print("coord: " + str(int(coord)) + " val " + str(rav[int(coord)]))
                val += rav[int(coord)]
            #print("val: " + str(val))
            print("--------")
            res[i,x] = val
            
    print(res)
    print("------")
    print(it.sum(1))
    print(it.sum(1).shape)
    print(it[0,:,0])
    print(it[0,:,1])
    print(it[1,:,0])
    print(it[1,:,1])

def test_slice_dim3():

    """
    i: 0 j: 0 x: 0
    i: 0 j: 0 x: 1
    --------
    i: 0 j: 1 x: 0
    i: 0 j: 1 x: 1
    --------
    i: 1 j: 0 x: 0
    i: 1 j: 0 x: 1
    --------
    i: 1 j: 1 x: 0
    i: 1 j: 1 x: 1
    --------
    i: 2 j: 0 x: 0
    i: 2 j: 0 x: 1
    --------
    i: 2 j: 1 x: 0
    i: 2 j: 1 x: 1
    """

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
    strides_n = [x/8 for x in list(it.strides)]
    print(strides_n)
    
    rav = it.ravel()
    print(rav)
    
    print(it)
    
    
    print("------")
    print(it[:,0,0])
    print(it[:,0,0].shape)
    print("-----------")
    print(it[2, 0, 0])
    
    res = np.array([[0,0], [0,0] ,[0,0]])


    for i in range(0, it.shape[0]):
   
        for j in range(0, it.shape[1]):
            val=0
            for x in range(0, it.shape[2]):
                print("i: " + str(i) + " j: " + str(j) + " x: " + str(x))
                print("i: " + str(i * strides_n[0]) + " j: " + str(j * strides_n[1]) + " x: " + str(x * strides_n[2]))
                coord = i * strides_n[0] + j * strides_n[1] + x * strides_n[2]
                print("coord: " + str(int(coord)) + " val " + str(rav[int(coord)]))
                val += rav[int(coord)]
            #print("val: " + str(val))
            print("--------")
            res[i, j] = val
            
    print(res)
    print("------")
    print(it.sum(2))
    print(it.sum(2).shape)
    print(it.sum(2).ravel())
    print(it[0,0,:])
    print(it[0,1,:])
    print(it[1,0,:])
    print(it[1,1,:])
    print(it[2,0,:])
    print(it[2,1,:])
    
def combo():
    print("combo")
    
    a_col = [[0, 1, 2],[0,1],[0,1]]
    a_cob_list =  list(itertools.permutations(a_col, 3))
    a_prod = list(itertools.product([2,1,0],[1,0],[1,0]))
    print(a_prod)
    df = pd.DataFrame(a_prod, columns=['a','b','c'])
    print("dim1")
    print(df.sort_values(['b','c','a'],ascending=True))
    print("----- dim2")
    print(df.sort_values(['a','c','b'],ascending=True))
    print('======== dim3')
    print(df.sort_values(['a','b','c'],ascending=True))
    
    
def slice():
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
    it1 = it[0,0,0:2]
    print(it1)
    print(it1.shape)
    
if __name__ == "__main__":
    pass
    #slice()
    #test_sum()
    #test_slice_dim1()
    #test_slice_dim2()
    test_slice_dim3()
    #combo()
    