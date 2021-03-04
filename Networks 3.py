# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:29:40 2021

@author: 44743
"""
"""
Networks Project; Ben Amroota 
CID: 01508466
"""
#%% WRITE
import pickle

with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\AAAAAAAAAA', 'wb') as dummy:
    pickle.dump(Data, dummy, protocol=pickle.HIGHEST_PROTOCOL)
    
#%% READ
import pickle
# Data100k;6
# Data100k;10
# Data1.5M;2R
with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\AAAAAAAAAAAAA', 'rb') as dummy:
    dataA = pickle.load(dummy)

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from collections import Counter
##%%
def Increment():
    time[0] += 1
    vertex[0] += 1
    degree.append(0)
    vertex_con.append([])
    vertices.append(vertices[-1] + 1)

def Edge():
#    m = 3 # np.random.randint(0,3)
    for a in range(m):
        degree[-1] += 1

        # For each node i, define a probability of an end attaching, prob
        prob = np.array(degree[:-1])/(sum(degree[:-1]))
        prob = prob.tolist()
        Exist_ver = np.random.choice(vertices[:-1], p = prob)

        if a >= 1:
            while np.any(Exist_ver == np.array(vertex_con[-1])) == True:
                Exist_ver = np.random.choice(vertices[:-1], p = prob)

        # Adds the index of the existing vertex to the list of new vertex connections
        vertex_con[-1].append(Exist_ver)  
        # Adds the index of the new vertex to the list of existing vertex connections              
        vertex_con[Exist_ver-1].append(vertices[-1])
        degree[Exist_ver-1] += 1
    
def A(Iterations):
    for b in range(Iterations):
        Increment()
        Edge()
#%%
# Initialisation
N = 4
vertex = np.array([N])
time = np.array([N])
edges = []

vertices = [1,2,3,4]
degree = [2,2,2,2]
vertex_con = [[2,3,4], [1,3,4], [1,2,4],[1,2,3]]
Data = [vertices, degree, vertex_con]
vert_list = [[],[],[],[]]

#%%
data = {}
# R-1 is the number of separate networks simulated  
# I is the number of single grain additions (i.e. total time)
R = 2 
t = 400
g = np.array([1,2,3,4])
for j in range(len(g)):
    for h in range(1,R):
        m = g[j]
        A(t)
        data[g[j],h] = [vertices, degree, vertex_con]
        
        # Condition to handle hitting the end of the list h
        if h < R-1:
            L = g[j]
        if h == R-1:
            if j >= len(g)-1:
                L = g[j]
            else:
                L = g[j+1]
        vertex = np.array([N])
        time = np.array([N])
        edges = []
        
        vertices = [1,2,3,4]
        degree = [2,2,2,2]
        vertex_con = [[2,3,4], [1,3,4], [1,2,4],[1,2,3]]
        Data = [vertices, degree, vertex_con]
        vert_list = [[],[],[],[]]
# 15:43 19:28 1,1 2,1 3,1 complete; 4,1 87214/100000 process complete
#%%
A(100)
#%%
from collections import Counter 

Count = {}

# The number of nodes with total degree k; degree k is the key in the dict
Count = Counter(data[2,1][1])

#for e in range(len(Count)):
#    Count[g[e],0] = dict(Counter(dataTC[g[e]]))
#Prob = []
#Val = [[],[],[],[],[],[],[]]
n = []
k = []

for e in sorted(Count.keys()):
    #for r in sorted(Prob[g[e],0].keys()):
    #print(e)
    #for a in range(Count[e]):
    n.append(Count[e]/sum(data[1,1][1]))
    k.append(e)
        
#        Val[e].append(r)
#        Prob[e].append(Prb[g[e],0][r])


#for r in sorted(Count.values()):
    
#%%
#plt.figure()
plt.plot(k, n)
plt.xscale("log")
plt.yscale("log")
plt.show()
plt.grid()

#%%
# Attachment list
Deg = {}
# The number of nodes with total degree k; degree k is the key in the dict
Deg = Counter(degree)
n = []
for e in sorted(Deg.keys()):
    #for r in sorted(Prob[g[e],0].keys()):
    print(e)
    for a in range(Deg[e]):
        n.append(Deg[e])

#%%
#A(50)
A(1000)
#print((sum(degree)-6)/2)
#print(sum(edges))