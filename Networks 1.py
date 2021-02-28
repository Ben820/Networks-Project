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
    vertex_con.append([])
    vertices.append(vertices[-1] + 1)

def Edge():
    m = 3 # np.random.randint(0,3)
    edges.append(m)
#    print("m=")
#    print(m)
    for a in range(m):
        degree[-1] += 1
        # For each node i, define a probability of an end attaching, prob
        prob = np.array(degree[:-1])/(sum(degree[:-1]))
        prob = prob.tolist()
        Exist_ver = np.random.choice(vertices[:-1], p = prob)
        print(prob)
#        """ Probability not working properly - needs to be a separate probability 
#        for each of the nodes bar the newest """
        vertex_con[-1].append(Exist_ver)
        vertex_con[Exist_ver-1].append(vertices[-1])
        #print(Exist_ver)
        degree[Exist_ver-1] += 1
        """ Double connects to vertices - always one after each other so its 
        due to the first edge ( of the three) connecting, and then the subsequent 
        edges do not know NOT to connect to that same vertex, hence the double or 
        triple connection """
#        """ Need to add Exist_ver to the list of vertex connections for the NEW vertex, 
#        and add the NEW vertex to the list of vertex connections for the Exist_ver vertex. """

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
vertex_con = [[2,3], [1,3], [1,2]]
Data = [vertices, degree, vertex_con]
#%%
#A(50)
A(20)
#print((sum(degree)-6)/2)
#print(sum(edges))