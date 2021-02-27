# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:29:40 2021

@author: 44743
"""
"""
Networks Project; Ben Amroota 
CID: 01508466
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
#%%
def Increment():
    time[0] += 1
    vertex[0] += 1
    degree.append(0)
    vertices.append(vertices[-1] + 1)

def Edge():
    m = 3 # np.random.randint(0,3)
    edges.append(m)
    print("m=")
    print(m)
    for a in range(m):
        degree[-1] += 1
        # For each node i, define a probability of an end attaching, prob
        prob = np.array(degree[:-1])/(sum(degree[:-1]))
        prob = prob.tolist()
        Exist_ver = np.random.choice(vertices[:-1], p = prob)
        degree[Exist_ver-1] += 1

def A(Iterations):
    for b in range(Iterations):
        Increment()
        Edge()
#%%
# Initialisation
N = 3
vertex = np.array([0])
time = np.array([0])
edges = []

vertices = [1,2,3]
degree = [2,2,2]
Data = [vertices, degree]

A(5)
print((sum(degree)-6)/2)
print(sum(edges))