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

with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\AAAAAAAAAAA', 'wb') as dummy:
    pickle.dump(data, dummy, protocol=pickle.HIGHEST_PROTOCOL)
    
#%% READ
import pickle
# Complete3m100k
# Incompletem4100k
# Data1.5M;2R

# P10k3R64i2;4;8;16;32i
# P10k5R64
# 32ii
# Pref10k;50R32ii
# Rand4;8;16;32;64

# Prefer100k50runs
with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\32ii', 'rb') as dummy:
    data = pickle.load(dummy)

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from collections import Counter
#%%
""" MIXED ATTACHMENT """
# Algorithm/ Network definitions
def Increment():
    degree.append(0)
    vertex_con.append([])
    vertices.append(vertices[-1] + 1)

def Edge():
    #vert = vertices[:-1]
    for a in range(m):
        degree[-1] += 1

        x = np.random.choice([1,2], p = [q,1-q]) # Chooses between Preferential and Random 
        if x == 1:
            # For each node i, define a probability of an end attaching, prob
            #prob = np.array(degree[:-1])/(sum(degree[:-1]))
            #prob = prob.tolist()
            Exist_ver = wattachlist[np.random.randint(len(wattachlist))]#[:-1]))]
            #print(Exist_ver)
            #Exist_ver = np.random.choice(vertices[:-1], p = prob)
    
            if a >= 1:
                while np.any(Exist_ver == np.array(vertex_con[-1])) == True:
                    #Exist_ver = np.random.choice(vertices[:-1], p = prob)
                    #Exist_ver = vertices[np.random.randint(len(vertices[:-1]))]
                    Exist_ver = wattachlist[np.random.randint(len(wattachlist))]#[:-1]))]
    
            """ Existing vertex cannot be chosen to be the new node 
            Remove eself loops """
            while Exist_ver == vertices[-1]:
                Exist_ver = wattachlist[np.random.randint(len(wattachlist))]
            
            #print(Exist_ver)
            # Adds the index of the existing vertex to the list of new vertex connections
            vertex_con[-1].append(Exist_ver)  
            # Adds the index of the new vertex to the list of existing vertex connections              
            vertex_con[Exist_ver-1].append(vertices[-1]) # Old vertices
            degree[Exist_ver-1] += 1
#            wattachlist.append(Exist_ver)
#            wattachlist.append(vertices[-1])
            
        #if x == 2:
        else:
            #Exist_ver = np.random.choice(vertices[:-1])#, p = prob)
            Exist_ver = vertices[np.random.randint(len(vertices))]
    
            if a >= 1:
                while np.any(Exist_ver == np.array(vertex_con[-1])) == True:
                    #Exist_ver = np.random.choice(vertices[:-1])#, p = prob)
                    Exist_ver = vertices[np.random.randint(len(vertices))]
            
            """ Existing vertex cannot be chosen to be the new node 
            Removes self loops """
            while Exist_ver == vertices[-1]:
                Exist_ver = vertices[np.random.randint(len(vertices))]

            # Adds the index of the existing vertex to the list of new vertex connections
            vertex_con[-1].append(Exist_ver)  
            # Adds the index of the new vertex to the list of existing vertex connections              
            vertex_con[Exist_ver-1].append(vertices[-1])
            degree[Exist_ver-1] += 1
        
        wattachlist.append(Exist_ver)
        wattachlist.append(vertices[-1])
    
def A(Iterations):
    for b in range(Iterations):
        Increment()
        Edge()
#%%
Nng = [3,5,9,17,33]
Nn = Nng[0] # m+1 nodes in system
vertices = [i for i in range(1,Nn+1)]
ver = vertices.copy()
degree = [Nn-1]*Nn
vertex_con = [[] for o in range(Nn)]
for k in range(len(ver)):
    ver.pop(k)
    vertex_con[k] = ver
    ver = vertices.copy()
wattachlist = np.concatenate(vertex_con).tolist()
#%%
# Data Collection Cell
data = {}
# R-1 is the number of separate networks simulated  
# I is the number of single grain additions (i.e. total time)
R = 101
t = 10000
t = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
g = np.array([2,4,8,16,32])#,16,32,64])
#g = np.array([32])
Nng = [3,5,9,17,33]
#Nng = [33]
#g = np.array([1,2,3,4])
# If Task 4, len(t), if other len(g) CHANGE BETWEEN t and g
for j in range(len(t)):
    for h in range(1,R):
        #Nn = Nng[j]
        Nn = 3
        #print(Nn)
        #m = g[j]
        m = 2
        # q is the probability the edge is joined using preferential attachment 
        q = 2/3
        A(t[j]) # A(t[j]) A(t)
        data[t[j],h] = [degree, vertex_con]
        
        # Condition to handle hitting the end of the list h
        # HASH THIS OUT FOR TASK 4
#        if h < R-1:
#            Nn = Nng[j]
#        if h == R-1:
#            if j >= len(t)-1:
#                Nn = Nng[j]
#            else:
#                Nn = Nng[j+1]
                        
        """  New initialisation """
        vertices = [i for i in range(1,Nn+1)]
        ver = vertices.copy()
        degree = [Nn-1]*Nn
        vertex_con = [[] for o in range(Nn)]
        for k in range(len(ver)):
            ver.pop(k)
            vertex_con[k] = ver
            ver = vertices.copy()
        wattachlist = np.concatenate(vertex_con).tolist()
    
        print("Run Complete")
    print("Iteration Complete")
        #Data = [degree, vertex_con]
#%% 
# Need to first run logbin file - Credit: Max Falkenberg McGillivray
#%%
""" USEFUL TOOL """
for j in range(len(g)):
    for d in range(1,2):
        print(min(data[g[j],d][0]))
#ind = data[16,7][0].index(6)
#del(data[16,7][0][ind])
#%%
""" Primitive plotting """
plt.figure()
scale = 1.2
s0 = False # whether or not to include s = 0 avalanches 

#bin_k2 = logbin(data[2,1][0],scale, s0)
#bin_k4 = logbin(data[4,1][0],scale, s0)
#bin_k8 = logbin(data[8,1][0],scale, s0)
#bin_k16 = logbin(data[16,1][0],scale, s0)
bin_k32 = logbin(data[131072,56][0],scale, s0)

#plt.plot(bin_k2[0], bin_k2[1], 'x-', label = "m = 2")
#plt.plot(bin_k4[0], bin_k4[1], 'x-', label = "m = 4")
#plt.plot(bin_k8[0], bin_k8[1], 'x-', label = "m = 8")
#plt.plot(bin_k16[0], bin_k16[1], 'x-', label = "m = 16")
plt.plot(bin_k32[0], bin_k32[1], 'x-', label = "m = 32")

plt.xlabel("Degree, $k$", size = "15")
plt.ylabel("Degree probability, $P_\infty(k)$", size = "15")
plt.xscale("log")
plt.yscale("log")
plt.legend()
#plt.grid()
plt.show()
#%%
""" RELEVANT """
scale = 1.3
s0 = False # whether or not to include s = 0 avalanches 

bin2 = []
bin4 = []
bin8 = []
bin16 = []
bin32 = [] 

A = ['bin2', 'bin4', 'bin8', 'bin16', 'bin32']
binlist = [bin2, bin4, bin8, bin16, bin32]
#binlist = [bin32]

R = 51

for h in range(1,R):
    bin2.append(logbin(data[2,h][0],scale))
    bin4.append(logbin(data[4,h][0],scale))
    bin8.append(logbin(data[8,h][0],scale))
    bin16.append(logbin(data[16,h][0],scale))
    bin32.append(logbin(data[32,h][0],scale))

g = np.array([2,4,8,16,32])
#%%
""" Maximum degree distribution; m = 2, N = 2^10 - 2^17 """
scale = 1.2
s0 = False # whether or not to include s = 0 avalanches 

bin128 = []
bin256 = []
bin512 = []
bin1024 = []
bin2048 = [] 
bin4096 = []
bin8192 = []
bin16384 = []
bin32768 = []
bin65536 = [] 
bin131072 = []

A = ['bin128', 'bin256', 'bin512', 'bin1024', 'bin2048', 'bin4096', 'bin8192', 'bin16384', 'bin32768',
     'bin65536', 'bin131072']
binlist = [bin128, bin256, bin512, bin1024, bin2048, bin4096, bin8192, bin16384, bin32768, bin65536,
           bin131072]

binlist = [bin128, bin512, bin2048, bin8192, bin32768, bin131072]

#binlist = [bin32]

R = 101

for h in range(1,R):
    bin128.append(logbin(data[128,h][0],scale))
    bin256.append(logbin(data[256,h][0],scale))
    bin512.append(logbin(data[512,h][0],scale))
    bin1024.append(logbin(data[1024,h][0],scale))
    bin2048.append(logbin(data[2048,h][0],scale))
    bin4096.append(logbin(data[4096,h][0],scale))
    bin8192.append(logbin(data[8192,h][0],scale))
    bin16384.append(logbin(data[16384,h][0],scale))
    bin32768.append(logbin(data[32768,h][0],scale))
    bin65536.append(logbin(data[65536,h][0],scale))
    bin131072.append(logbin(data[131072,h][0],scale))

g = np.array([2,4,8,16,32])
#%%
""" Maximum degree vs N """
max128 = []
max256 = []
max512 = []
max1024 = []
max2048 = []
max4096 = []
max8192 = []
max16384 = []
max32768 = []
max65536 = []
max131072 = []

maxlist = [max128, max512, max2048, max8192, max32768, max131072]
maxlist = [max128, max256, max512, max1024, max2048, max4096, max8192, max16384, 
           max32768, max65536, max131072]

ts = [128, 512, 2048, 8192, 32768, 131072]
R = 101

for v in range(len(maxlist)):
    for h in range(1,R):
        maxlist[v].append(np.amax(np.array(data[t[v],h][0])))

stdev = []
avg = []

for r in range(len(maxlist)):
    stdev.append(np.std(np.array(maxlist[r])))
    avg.append(np.mean(np.array(maxlist[r])))
#%% 
""" Variable m, Fixed N """
unbin2 = []
unbin4 = []
unbin8 = []
unbin16 = []
unbin32 = [] 

for h in range(1,R):
    unbin2.append(data[2,h][0])
    unbin4.append(data[4,h][0])
    unbin8.append(data[8,h][0])
    unbin16.append(data[16,h][0])
    unbin32.append(data[32,h][0])

# Concatenate unbin lists 
UnBin2 = np.concatenate(unbin2)
UnBin4 = np.concatenate(unbin4)
UnBin8 = np.concatenate(unbin8)
UnBin16 = np.concatenate(unbin16)
UnBin32 = np.concatenate(unbin32)

UnBinlist = [UnBin2, UnBin4, UnBin8, UnBin16, UnBin32]
#%%
""" Fixed m, Variable N """
unbin128 = []
unbin256 = []
unbin512 = []
unbin1024 = []
unbin2048 = [] 
unbin4096 = []
unbin8192 = []
unbin16384 = []
unbin32768 = []
unbin65536 = []
unbin131072 = []

for h in range(1,R):
    unbin128.append(data[128,h][0])
    unbin256.append(data[256,h][0])
    unbin512.append(data[512,h][0])
    unbin1024.append(data[1024,h][0])
    unbin2048.append(data[2048,h][0])
    unbin4096.append(data[4096,h][0])
    unbin8192.append(data[8192,h][0])
    unbin16384.append(data[16384,h][0])
    unbin32768.append(data[32768,h][0])
    unbin65536.append(data[65536,h][0])
    unbin131072.append(data[131072,h][0])

# Concatenate unbin lists 
unbin128 = np.concatenate(unbin128)
unbin256 = np.concatenate(unbin256)
unbin512 = np.concatenate(unbin512)
unbin1024 = np.concatenate(unbin1024)
unbin2048 = np.concatenate(unbin2048)
unbin4096 = np.concatenate(unbin4096)
unbin8192 = np.concatenate(unbin8192)
unbin16384 = np.concatenate(unbin16384)
unbin32768 = np.concatenate(unbin32768)
unbin65536 = np.concatenate(unbin65536)
unbin131072 = np.concatenate(unbin131072)

UnBinlist = [unbin128, unbin256, unbin512, unbin1024, unbin2048, unbin4096, 
             unbin8192, unbin16384, unbin32768, unbin65536, unbin131072]

UnBinlist = [unbin128, unbin512, unbin2048, 
             unbin8192, unbin32768, unbin131072]

#%%
#scale = 1.1 # SCALE NEEDS TO BE SET WITH THE ERRORS (i.e. scale needs to be same for both errors and this)
s0 = False # whether or not to include s = 0 avalanches 
# Fixed N variable m
LogBin = []
for j in range(len(g)):
    LogBin.append(logbin(UnBinlist[j],scale))

# Fixed m variable N
#LogBin = []
#for j in range(len(t)-5):
#    LogBin.append(logbin(UnBinlist[j],scale))
#%%
""" ERROR ANALYSIS - GEORGE ACKNOWLEDGMENT """

""" add probabilities together from same geometric means across iterations, to get
set of values per geometric mean; calculate average and st dev """

errlist = []
Avg_ylist = []
BigXlist = []
for u in range(len(binlist)):
    xlist = []
    ylist = []
    
    for h in range(0,R-1):
        xlist.append(binlist[u][h][0].tolist())
        ylist.append(binlist[u][h][1].tolist())
    Big_xlist = np.unique(np.concatenate(xlist, 0))
    Sampl = []
    
    for a in range(len(Big_xlist)):
        sample = []
        for v in range(len(ylist)):
            if Big_xlist[a] in xlist[v]:
                sample.append(ylist[v][xlist[v].index(Big_xlist[a])])          
        Sampl.append(sample)  
    err_y = []
    avg_y = []
    
    for s in range(len(Sampl)):
        err_y.append(np.std(Sampl[s])/np.sqrt(np.size(Sampl[s])))
        avg_y.append(np.mean(Sampl[s]))
    errlist.append(err_y)
    Avg_ylist.append(avg_y)
    BigXlist.append(Big_xlist.tolist())
##%%
#""" Testing Error Analysis Code """
#x1 = [0,1,2,3]
#x2 = [1,2,3,4]
#x3 = [2,4,5]
#y1 = [10,20,30,40]
#y2 = [11,21,31,41]
#y3 = [12,22,32]

#big_x = [x1, x2, x3]
#big_y = [y1, y2, y3]
#Big_x = np.unique(np.concatenate(big_x))

#Samp = []
#for a in range(len(Big_x)):
#    sample = []
#    for v in range(len(big_y)):
#        if Big_x[a] in big_x[v]:
#            sample.append(big_y[v][big_x[v].index(a)])
#    Samp.append(sample)
#%%
""" Averaged Logbin + Errors """
plt.figure()

""" Combining different runs into a huge dataset which is logbinned to give x and y """

plt.plot(LogBin[0][0], LogBin[0][1], 'x-', label = "m=2")
plt.plot(LogBin[1][0], LogBin[1][1], 'x-', label = "m=4")
plt.plot(LogBin[2][0], LogBin[2][1], 'x-', label = "m=8")
plt.plot(LogBin[3][0], LogBin[3][1], 'x-', label = "m=16")
plt.plot(LogBin[4][0], LogBin[4][1], 'x-', label = "m=32")

plt.errorbar(LogBin[0][0], LogBin[0][1], yerr = errlist[0], color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)
plt.errorbar(LogBin[1][0], LogBin[1][1], yerr = errlist[1], color = "orange", fmt='o', mew=1, ms=0.2, capsize=6)
plt.errorbar(LogBin[2][0], LogBin[2][1], yerr = errlist[2], color = "green", fmt='o', mew=1, ms=0.2, capsize=6)
plt.errorbar(LogBin[3][0], LogBin[3][1], yerr = errlist[3], color = "crimson", fmt='o', mew=1, ms=0.2, capsize=6)
plt.errorbar(LogBin[4][0], LogBin[4][1], yerr = errlist[4], color = "purple", fmt='o', mew=1, ms=0.2, capsize=6)


#import math

def Pref_deg_dist(k, m):
    A = 2*m*(m+1)
    B = (k+2)*(k+1)*k
    y = A/B
    return y

def Rand_deg_dist(k, m):
    A = m**(k-m)
    B = (1+m)**(1+k-m)
#    ai = k-m
#    bi = 1+m
#    bii = 1+k-m
#    A = math.pow(m, ai)
#    B = math.pow(bi, bii)
    y = A/B
    return y

def Mixed_deg_dist_twothirds(k, m):
    A = 3*((2*m)+3)*((2*m)+2)*((2*m)+1)
    B = np.square(k+m+3)
    C = B*(k+m+2)*(k+m+1)
    return A/C

def Mixed_deg_dist_onehalf(k, m):
    A = 3*((2*m)+3)*((2*m)+2)*((2*m)+1)
    B = np.square(k+m+3)
    C = B*(k+m+2)*(k+m+1)
    return A/C

func = Pref_deg_dist
colour = ["navy", "orangered", "forestgreen", "firebrick", "blueviolet"]

theor_func = []

for j in range(len(g)):
    arO = np.arange(g[j], np.amax(LogBin[j][0]+10), 0.01) 
    plt.plot(arO, func(arO, g[j]), '--', zorder=10, color = colour[j]) 
    
    theor_func.append(func(arO, g[j]))


plt.xlabel("Degree, $k$", size = "15")
plt.ylabel("Degree probability, $P_\infty(k)$", size = "15")
plt.xscale("log")
plt.yscale("log")
plt.legend()
#plt.grid()
#plt.savefig("Task 3a unbin.png", dpi = 1000)
plt.show()
#%%
""" Statistical Tests; Kolmogorov Smirnov Test """
from scipy import stats

for j in range(len(g)):
    KS = stats.ks_2samp(theor_func[j], LogBin[j][1])
    print("KS Test m =", g[j])
    print(KS)
#%%
""" Averaged Logbin + Errors """
plt.figure()

""" Combining different runs into a huge dataset which is logbinned to give x and y """
labels = ["N = 128", "N = 256", "N = 512", "N = 1024", "N = 2048", "N = 4096", "N = 8192",
          "N = 16384", "N = 32768", "N = 65536", "N = 131072"]

labels_redu = ["N = 128", "N = 512", "N = 2048", "N = 8192",
          "N = 32768", "N = 131072"]

tlist = np.sqrt(np.array([128, 512, 2048, 8192, 32768, 131072])).tolist()
trlist = np.log10(np.array([128, 512, 2048, 8192, 32768, 131072])).tolist()
colours = ["lightseagreen", "orangered", "forestgreen", "firebrick", "blueviolet", "navy"]

#func = Rand_deg_dist

for z in [5,4,3,2,1,0]:#range(len(t)-5):
    #ycoll = func(LogBin[z][0], 2)
    plt.plot(LogBin[z][0], LogBin[z][1], 'x-', color = colours[z], label = labels_redu[z])
    plt.errorbar(LogBin[z][0], LogBin[z][1], yerr = errlist[z], color = colours[z], fmt='o', mew=1, ms=0.2, capsize=6)
    # Data Collpase Preferential
#    plt.plot(LogBin[z][0]/tlist[z], LogBin[z][1]/ycoll, 'x', color = colours[z], label = labels_redu[z])
#    plt.errorbar(LogBin[z][0]/trlist[z], LogBin[z][1]/ycoll, yerr = errlist[z], color = colours[z], fmt='o', mew=1, ms=0.2, capsize=6)
#
#    plt.plot(LogBin[z][0]/trlist[z], LogBin[z][1]/ycoll, 'x', color = colours[z], label = labels_redu[z])
#    plt.errorbar(LogBin[z][0]/trlist[z], LogBin[z][1]/ycoll, yerr = errlist[z], color = colours[z], fmt='o', mew=1, ms=0.2, capsize=6)
    
    
#plt.plot(LogBin[0][0], LogBin[0][1], 'x-', label = "m=2")
#plt.plot(LogBin[1][0], LogBin[1][1], 'x-', label = "m=4")
#plt.plot(LogBin[2][0], LogBin[2][1], 'x-', label = "m=8")
#plt.plot(LogBin[3][0], LogBin[3][1], 'x-', label = "m=16")
#plt.plot(LogBin[4][0], LogBin[4][1], 'x-', label = "m=32")

#plt.errorbar(LogBin[0][0], LogBin[0][1], yerr = errlist[0], color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)
#plt.errorbar(LogBin[1][0], LogBin[1][1], yerr = errlist[1], color = "orange", fmt='o', mew=1, ms=0.2, capsize=6)
#plt.errorbar(LogBin[2][0], LogBin[2][1], yerr = errlist[2], color = "green", fmt='o', mew=1, ms=0.2, capsize=6)
#plt.errorbar(LogBin[3][0], LogBin[3][1], yerr = errlist[3], color = "crimson", fmt='o', mew=1, ms=0.2, capsize=6)
#plt.errorbar(LogBin[4][0], LogBin[4][1], yerr = errlist[4], color = "purple", fmt='o', mew=1, ms=0.2, capsize=6)


#import math

def Pref_deg_dist(k, m):
    A = 2*m*(m+1)
    B = (k+2)*(k+1)*k
    y = A/B
    return y

def Rand_deg_dist(k, m):
    A = m**(k-m)
    B = (1+m)**(1+k-m)
#    ai = k-m
#    bi = 1+m
#    bii = 1+k-m
#    A = math.pow(m, ai)
#    B = math.pow(bi, bii)
    y = A/B
    return y

def Mixed_deg_dist_twothirds(k, m):
    A = 3*((2*m)+3)*((2*m)+2)*((2*m)+1)
    B = np.square(k+m+3)
    C = B*(k+m+2)*(k+m+1)
    return A/C

def Mixed_deg_dist_onehalf(k, m):
    A = 3*((2*m)+3)*((2*m)+2)*((2*m)+1)
    B = np.square(k+m+3)
    C = B*(k+m+2)*(k+m+1)
    return A/C

#func = Mixed_deg_dist_twothirds
#colour = ["navy", "orangered", "forestgreen", "firebrick", "blueviolet"]
#
#theor_func = []
#
#for j in range(len(g)):
#    arO = np.arange(g[j], np.amax(LogBin[j][0]+10), 0.01) 
#    #plt.plot(arO, func(arO, g[j]), '--', zorder=10, color = colour[j]) 
#    
#    theor_func.append(func(arO, g[j]))


plt.xlabel("Degree, $k$", size = "15")
plt.ylabel("Degree probability, $P_\infty(k)$", size = "15")
plt.xscale("log")
plt.yscale("log")
plt.legend()
#plt.grid()
#plt.savefig("Task 3a unbin.png", dpi = 1000)
plt.show()
#%%
""" Maximum degree vs N for variable N fixed m = 2 """
plt.figure()

plt.plot(t, avg, 'x')
plt.errorbar(t, avg, yerr = stdev, fmt='o', mew=1, ms=0.2, capsize=6)

plt.xlabel("Total Number of Nodes, $N$", size = "15")
plt.ylabel(r"Average Maximum Degree, $\langle k_1 \rangle$", size = "15")
#plt.xscale("log")
#plt.yscale("log")

plt.show()

















