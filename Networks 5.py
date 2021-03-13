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

with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\Pref248', 'wb') as dummy:
    pickle.dump(data, dummy, protocol=pickle.HIGHEST_PROTOCOL)
    
#%% READ
import pickle
# Complete3m100k
# Incompletem4100k
# Data1.5M;2R
with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\Pref248', 'rb') as dummy:
    data = pickle.load(dummy)

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from collections import Counter
#%%
""" PREFERENTIAL ATTACHMENT """
# Algorithm/ Network definitions
def Increment():
    degree.append(0) #!
    vertex_con.append([]) #!
    vertices.append(vertices[-1] + 1) #!

def Edge():
#    m = 3 # np.random.randint(0,3)
    for a in range(m):
        degree[-1] += 1

        x = np.random.choice([1,2], p = [q,1-q])
        if x == 1:
            # For each node i, define a probability of an end attaching, prob
            prob = np.array(degree[:-1])/(sum(degree[:-1]))
            #prob = prob.tolist()
            Exist_ver = np.random.choice(vertices[:-1], p = prob)
    
            if a >= 1:
                while np.any(Exist_ver == np.array(vertex_con[-1])) == True:
                    Exist_ver = np.random.choice(vertices[:-1], p = prob)
    
            # Adds the index of the existing vertex to the list of new vertex connections
            vertex_con[-1].append(Exist_ver)  
            # Adds the index of the new vertex to the list of existing vertex connections              
            vertex_con[Exist_ver-1].append(vertices[-1])
            degree[Exist_ver-1] += 1
            
        #if x == 2:
        else:
            Exist_ver = np.random.choice(vertices[:-1])#, p = prob)
    
            if a >= 1:
                while np.any(Exist_ver == np.array(vertex_con[-1])) == True:
                    Exist_ver = np.random.choice(vertices[:-1])#, p = prob)
    
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

vertices = [1,2,3,4]
degree = [3,3,3,3]
vertex_con = [[2,3,4], [1,3,4], [1,2,4], [1,2,3]]
#Data = [vertices, degree, vertex_con]
###%%
Nn = 64 # The number of nodes in the initial graph
vertices = [i for i in range(1,Nn+1)]
degree = [Nn-1]*Nn
degree = [2]*Nn
#vertex_con = [[] for o in range(Nn)]
#vertex_con = [vertices[:x] + vertices[x+1:] for x in range(Nn)]
vertex_con = [[o-1, o+1] for o in vertices]
vertex_con[0] = [Nn, 2]
vertex_con[Nn-1] = [Nn-1, 1]

#for a in range(Nn):
#    Exist_ver = np.random.choice(vertices[:-1])#, p = prob)
#
#    if a >= 1:
#        while np.any(Exist_ver == np.array(vertex_con[-1])) == True:
#            Exist_ver = np.random.choice(vertices[:-1])#, p = prob)
#
#    # Adds the index of the existing vertex to the list of new vertex connections
#    vertex_con[-1].append(Exist_ver)  
#    # Adds the index of the new vertex to the list of existing vertex connections              
#    vertex_con[Exist_ver-1].append(vertices[-1])
#    degree[Exist_ver-1] += 1
#%%
# Data Collection Cell
data = {}
# R-1 is the number of separate networks simulated  
# I is the number of single grain additions (i.e. total time)
R = 2 
t = 10000
g = np.array([2,4,8])#,16,32,64])
#g = np.array([1,2,3,4])
for j in range(len(g)):
    for h in range(1,R):
        m = g[j]
        # q is the probability the edge is joined using preferential attachment 
        q = 1
        A(t)
        data[g[j],h] = [degree, vertex_con]
        
#        # Condition to handle hitting the end of the list h
#        if h < R-1:
#            L = g[j]
#        if h == R-1:
#            if j >= len(g)-1:
#                L = g[j]
#            else:
#                L = g[j+1]
        
        vertices = [1,2,3,4]
        degree = [3,3,3,3]
        vertex_con = [[2,3,4], [1,3,4], [1,2,4],[1,2,3]]
        
        vertices = [i for i in range(1,Nn+1)]
        degree = [Nn-1]*Nn
        vertex_con = [vertices[:z] + vertices[z+1:] for z in range(Nn)]
        
        degree = [2]*Nn
        #vertex_con = [[] for o in range(Nn)]
        #vertex_con = [vertices[:x] + vertices[x+1:] for x in range(Nn)]
        vertex_con = [[o-1, o+1] for o in vertices]
        vertex_con[0] = [Nn, 2]
        vertex_con[Nn-1] = [Nn-1, 1]

        #Data = [degree, vertex_con]

# 15:43 19:28 1,1 2,1 3,1 complete; 4,1 87214/100000 process complete
# 11:11 13:39 1,1 2,1 3,1 complete; 4,1 96206/100000 process complete

# 13:25 
#%%
from collections import Counter 

Count = {}

# The number of nodes with total degree k; degree k is the key in the dict

nlist = []
klist = []
#for i in [1,2,3,4]:
for i in [2,4,8]:
    # Count contains the number of nodes with degree x; key is degree and value is 
    # the number of nodes with that degree 
    Count = Counter(data[i,1][0]) # dataA[i,1][1] is the degree of each node 
    n = []
    k = []
    
    for e in sorted(Count.keys()):
        n.append(Count[e]/sum(data[i,1][0])) # divided by total number of nodes 
        k.append(e)

    nlist.append(n)
    klist.append(k)
#%%
""" Unbinned degree distribution; Degree probability vs degree 
Number of nodes with degree x divided by total number of nodes vs degree """
plt.figure()

plt.plot(klist[0], nlist[0], 'x', label = "m = 1")
plt.plot(klist[1], nlist[1], 'x', label = "m = 2")
plt.plot(klist[2], nlist[2], 'x', label = "m = 3")
#plt.plot(klist[3], nlist[3], 'x', label = "m = 3")

plt.xlabel("Degree, $k$", size = "15")
plt.ylabel("Degree probability, $P_\infty(k)$", size = "15")
plt.xscale("log")
plt.yscale("log")
plt.legend()
#plt.grid()
#plt.savefig("Task 3a unbin.png", dpi = 1000)
plt.show()
#%%
""" Binned degree distribution; Degree probability vs degree 
Number of nodes with degree x divided by total number of nodes vs degree """
plt.figure()

scale = 1.2
s0 = False # whether or not to include s = 0 avalanches 
 
# Need to first run logbin file - Credit: Max Falkenberg McGillivray
#bin_k1 = logbin(data[1,1][0],scale, s0)
#bin_k2 = logbin(data[2,1][0],scale, s0)
#bin_k3 = logbin(data[3,1][0],scale, s0)
#bin_k4 = logbin(data[4,1][0],scale, s0)

bin_k1 = logbin(data[2,1][0],scale, s0)
bin_k2 = logbin(data[4,1][0],scale, s0)
bin_k3 = logbin(data[8,1][0],scale, s0)
#bin_k4 = logbin(data[4,1][0],scale, s0)


plt.plot(bin_k1[0], bin_k1[1], 'x-', label = "m = 1")
plt.plot(bin_k2[0], bin_k2[1], 'x-', label = "m = 2")
plt.plot(bin_k3[0], bin_k3[1], 'x-', label = "m = 3")
#plt.plot(bin_k4[0], bin_k4[1], 'x-', label = "m = 4")


plt.xlabel("Degree, $k$", size = "15")
plt.ylabel("Degree probability, $P_\infty(k)$", size = "15")
plt.xscale("log")
plt.yscale("log")
plt.legend()
#plt.grid()
#plt.savefig("Task 3a unbin.png", dpi = 1000)
plt.show()
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
#%%
q = 0.5
x = np.random.choice([1,2], p = [q,1-q])
if x == 1:
    print('yes')
else:
    print("No")
#else:
#    print("fuck")