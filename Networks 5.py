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
with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\Pref10k;50R32ii', 'rb') as dummy:
    dataB = pickle.load(dummy)

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from collections import Counter
#%%
""" MIXED ATTACHMENT """
# Algorithm/ Network definitions
def Increment():
    degree.append(0) #!
    vertex_con.append([]) #!
    vertices.append(vertices[-1] + 1) #!
   # attachmentlist.append()

def Edge():
    #vert = vertices[:-1]
#    m = 3 # np.random.randint(0,3)
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
            Exist_ver = vertices[np.random.randint(len(vert))]
    
            if a >= 1:
                while np.any(Exist_ver == np.array(vertex_con[-1])) == True:
                    #Exist_ver = np.random.choice(vertices[:-1])#, p = prob)
                    Exist_ver = vertices[np.random.randint(len(vert))]
    
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
# Initialisation

#vertices = [1,2,3,4]
#degree = [3,3,3,3]
#vertex_con = [[2,3,4], [1,3,4], [1,2,4], [1,2,3]]
#Data = [vertices, degree, vertex_con]
###%%
Nn = 32 # The number of nodes in the initial graph
vertices = [i for i in range(1,Nn+1)]
#degree = [Nn-1]*Nn
degree = [31]*Nn
#vertex_con = [[] for o in range(Nn)]
#vertex_con = [vertices[:x] + vertices[x+1:] for x in range(Nn)]
vertex_con = [[o-1, o+1] for o in vertices]
vertex_con[0] = [Nn, 2]
vertex_con[Nn-1] = [Nn-1, 1]

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

    
    
#vertex_con = [vertices.copy().pop(o) for o in range(len(vertices))] # problem is vertices labelling 




#for j in range(len(vertex_con)):
#    del(vertex_con[j][j])
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
R = 51
t = 100000
g = np.array([2,4,8,16,32])#,16,32,64])
#g = np.array([32])
Nng = [3,5,9,17,33]
#Nng = [33]
#g = np.array([1,2,3,4])
for j in range(len(g)):
    for h in range(1,R):
        Nn = Nng[j]
        #print(Nn)
        m = g[j]
        # q is the probability the edge is joined using preferential attachment 
        q = 1
        A(t)
        data[g[j],h] = [degree, vertex_con]
        
#        # Condition to handle hitting the end of the list h
        if h < R-1:
            Nn = Nng[j]
        if h == R-1:
            if j >= len(g)-1:
                Nn = Nng[j]
            else:
                Nn = Nng[j+1]
        
#        vertices = [1,2,3,4]
#        degree = [3,3,3,3]
#        vertex_con = [[2,3,4], [1,3,4], [1,2,4],[1,2,3]]
        
        """  Old initialisation """
#        vertices = [i for i in range(1,Nn+1)]
##        degree = [Nn-1]*Nn
#        vertex_con = [vertices[:z] + vertices[z+1:] for z in range(Nn)]
#        
#        degree = [2]*Nn
#        #vertex_con = [[] for o in range(Nn)]
#        #vertex_con = [vertices[:x] + vertices[x+1:] for x in range(Nn)]
#        vertex_con = [[o-1, o+1] for o in vertices]
#        vertex_con[0] = [Nn, 2]
#        vertex_con[Nn-1] = [Nn-1, 1]
        
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

# 15:43 19:28 1,1 2,1 3,1 complete; 4,1 87214/100000 process complete
# 11:11 13:39 1,1 2,1 3,1 complete; 4,1 96206/100000 process complete

# 13:25 
#%%
""" IRRELEVANT """
d = []
for i in range(len(data[2,1][0])):
    d.append(np.mean([data[2,1][0][i], data[2,2][0][i], data[2,3][0][i], data[2,4][0][i], data[2,5][0][i]]))#,
# data[2,6][0][i], data[2,7][0][i], data[2,8][0][i], data[2,9][0][i], data[2,10][0][i]]))
#%%
#scale = 1.2
#s0 = False # whether or not to include s = 0 avalanches 
#d = []
#
#for i in range(len(data[2,1][0])):
#    d.append(np.mean([logbin(data[2,1][i],scale, s0), logbin(data[2,2][i],scale, s0),
#                        logbin(data[2,3][i],scale, s0), logbin(data[2,4][i],scale, s0), 
#                        logbin(data[2,5][i],scale, s0), logbin(data[2,6][i],scale, s0), 
#                        logbin(data[2,7][i],scale, s0), logbin(data[2,8][i],scale, s0),
#                        logbin(data[2,9][i],scale, s0), logbin(data[2,10][i],scale, s0)]))
Data = {}
for c in g:
    Data[c] = (data[c,1][0] + data[c,2][0] + data[c,3][0] + data[c,4][0] + data[c,5][0])# + 
           # data[c,6][0] + data[c,7][0] + data[c,8][0] + data[c,9][0] + data[c,10][0])
#bin_k1 = logbin(data[2,1][0],scale, s0)
#%%
""" Primitive plotting tool """
from collections import Counter 

Count = {}

# The number of nodes with total degree k; degree k is the key in the dict

R = 51
t = 100000
g = np.array([2,4,8,16,32])#,16,32,64])
#g = np.array([32])#,16,32,64])
Nn = 33

nlist = []
klist = []
Troubleshoot = []

#for i in [1,2,3,4]:
for i in range(len(g)):
    # Count contains the number of nodes with degree x; key is degree and value is 
    # the number of nodes with that degree 
    Count = Counter(data[g[i],1][0]) # dataA[i,1][1] is the degree of each node 
    n = []
    k = []
    T = []
    
    for e in sorted(Count.keys()):
        n.append(Count[e]/(t+Nng[i])) #sum(data[i,1][0])) # divided by total number of nodes 
        k.append(e)
        T.append(Count[e])
        
    Troubleshoot.append(T)
    nlist.append(n)
    klist.append(k)
#%%
""" Unbinned degree distribution; Degree probability vs degree 
Number of nodes with degree x divided by total number of nodes vs degree """
plt.figure()

plt.plot(klist[0], nlist[0], 'x', label = "m = 2")
plt.plot(klist[1], nlist[1], 'x', label = "m = 4")
plt.plot(klist[2], nlist[2], 'x', label = "m = 8")
plt.plot(klist[3], nlist[3], 'x', label = "m = 16")
plt.plot(klist[4], nlist[4], 'x', label = "m = 32")
#plt.plot(klist[5], nlist[5], 'x', label = "m = 64")

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
#plt.figure()

scale = 1.2
s0 = False # whether or not to include s = 0 avalanches 
 
# Need to first run logbin file - Credit: Max Falkenberg McGillivray
#bin_k1 = logbin(data[1,1][0],scale, s0)
#bin_k2 = logbin(data[2,1][0],scale, s0)
#bin_k3 = logbin(data[3,1][0],scale, s0)
#bin_k4 = logbin(data[4,1][0],scale, s0)
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

logdict = {}
##%%

#for i in range(len(bin16[0][0])):
#    for k in range(len(bin16)):
#        #bin16[k][0][i]
#        logdict[i] = [bin16[k][1][i]]


bin_k2 = logbin(data[2,1][0],scale, s0)
bin_k4 = logbin(data[4,1][0],scale, s0)
bin_k8 = logbin(data[8,1][0],scale, s0)
bin_k16 = logbin(data[16,1][0],scale, s0)
bin_k32 = logbin(data[32,1][0],scale, s0)
#bin_k64 = logbin(data[64,3][0],scale, s0)

#bin16 = [logbin(data[16,1][0],scale), logbin(data[16,2][0],scale, s0), logbin(data[16,3][0],scale, s0)]

Bins = [bin_k2, bin_k4, bin_k8, bin_k16, bin_k32]
#bin_k1 = logbin(Data[2],scale, s0)
#bin_k2 = logbin(Data[4],scale, s0)
#bin_k3 = logbin(Data[8],scale, s0)

#plt.plot(bin_k1[0], bin_k1[1], 'x-', label = "m = 1")
#plt.plot(bin_k2[0], bin_k2[1], 'x-', label = "m = 2")
#plt.plot(bin_k3[0], bin_k3[1], 'x-', label = "m = 3")
#plt.plot(bin_k4[0], bin_k4[1], 'x-', label = "m = 4")

#bin_k2 = logbin(Data[2],scale, s0)
#bin_k4 = logbin(Data[4],scale, s0)
#bin_k8 = logbin(Data[8],scale, s0)
#bin_k16 = logbin(Data[16],scale, s0)
#bin_k32 = logbin(Data[32],scale, s0)
#bin_k64 = logbin(Data[64],scale, s0)

plt.plot(bin_k2[0], bin_k2[1], 'x-', label = "m = 2")
plt.plot(bin_k4[0], bin_k4[1], 'x-', label = "m = 4")
plt.plot(bin_k8[0], bin_k8[1], 'x-', label = "m = 8")
plt.plot(bin_k16[0], bin_k16[1], 'x-', label = "m = 16")
plt.plot(bin_k32[0], bin_k32[1], 'x-', label = "m = 32")
#plt.plot(bin_k64[0], bin_k64[1], 'x-', label = "m = 64")

plt.xlabel("Degree, $k$", size = "15")
plt.ylabel("Degree probability, $P_\infty(k)$", size = "15")
plt.xscale("log")
plt.yscale("log")
plt.legend()
#plt.grid()
#plt.savefig("Task 3a unbin.png", dpi = 1000)
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
binlist = [bin2, bin4, bin8, bin16, bin32] #[bin32]#
#binlist = [bin32]
#logdict = {}

R = 51

for h in range(1,R):
    bin2.append(logbin(data[2,h][0],scale))
    bin4.append(logbin(data[4,h][0],scale))
    bin8.append(logbin(data[8,h][0],scale))
    bin16.append(logbin(data[16,h][0],scale))
    bin32.append(logbin(data[32,h][0],scale))

g = np.array([2,4,8,16,32])#,16,32,64])

#for j in range(len(g)):
#    logdict[A[j]] = binlist[j]
#%%
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
#%%
# Concatenate unbin lists 
UnBin2 = np.concatenate(unbin2)
UnBin4 = np.concatenate(unbin4)
UnBin8 = np.concatenate(unbin8)
UnBin16 = np.concatenate(unbin16)
UnBin32 = np.concatenate(unbin32)
##%%
UnBinlist = [UnBin2, UnBin4, UnBin8, UnBin16, UnBin32]
#%%
#scale = 1.1 # SCALE NEEDS TO BE SET WITH THE ERRORS (i.e. scale needs to be same for both errors and this)
s0 = False # whether or not to include s = 0 avalanches 

LogBin = []
for j in range(len(g)):
    LogBin.append(logbin(UnBinlist[j],scale))

#%%
""" ERROR ANALYSIS - GEORGE ACKNOWLEDGMENT """
errlist = []
Avg_ylist = []
BigXlist = []
for t in range(len(binlist)):
    xlist = []
    ylist = []
    
    for h in range(0,R-1):
        xlist.append(binlist[t][h][0].tolist())
        ylist.append(binlist[t][h][1].tolist())
    
    #np.concatenate(xlist, 0)
    Big_xlist = np.unique(np.concatenate(xlist, 0))
    ##%%
    Sampl = []
    for a in range(len(Big_xlist)):
       # print(a)
        sample = []
        for v in range(len(ylist)):
            #print(v)
            if Big_xlist[a] in xlist[v]:
                sample.append(ylist[v][xlist[v].index(Big_xlist[a])])
                
        Sampl.append(sample)
    
    #logerr = [Big_xlist.tolist(), Sampl]
    ##%%
    err_y = []
    avg_y = []
    for s in range(len(Sampl)):
        err_y.append(np.std(Sampl[s])/np.size(Sampl[s]))
        avg_y.append(np.mean(Sampl[s]))
    errlist.append(err_y)
    Avg_ylist.append(avg_y)
    BigXlist.append(Big_xlist.tolist())

""" add probabilities together from same geometric means across iterations, to get
set of values per geometric mean; calculate average and st dev """
##%%
#""" TESTING """
#x1 = [0,1,2,3]
#x2 = [1,2,3,4]
#x3 = [2,4,5]
#y1 = [10,20,30,40]
#y2 = [11,21,31,41]
#y3 = [12,22,32]
###%%
#
#big_x = [x1, x2, x3]
#big_y = [y1, y2, y3]
#Big_x = np.unique(np.concatenate(big_x))
###%%
#Samp = []
#for a in range(len(Big_x)):
#    sample = []
#    for v in range(len(big_y)):
#        if Big_x[a] in big_x[v]:
#            sample.append(big_y[v][big_x[v].index(a)])
#    Samp.append(sample)
##
###%%
###bin2[0], bin2[1]
##
##S = {}
##
##for j in range(len(g)):
##    for p in range(1,R):
##        for f in range(len(bin2[p])): # bin2[p] __> binlist[j][p]
##            S[j,p] = [[], []]
##
#%%
""" Averaged Logbin + Errors """
plt.figure()
#plt.errorbar(BigXlist[0], Avg_ylist[0], )

""" Plotting averages and geometric means as calculated by George 1st time """
#plt.plot(BigXlist[0], Avg_ylist[0], 'x-', label = "m=2")
#plt.plot(BigXlist[1], Avg_ylist[1], 'x-', label = "m=4")
##plt.plot(BigXlist[2], Avg_ylist[2], 'x-', label = "m=8")
##plt.plot(BigXlist[3], Avg_ylist[3], 'x-', label = "m=16")
##plt.plot(BigXlist[4], Avg_ylist[4], 'x-', label = "m=32")
#
#
#plt.errorbar(BigXlist[0], Avg_ylist[0], yerr = errlist[0], color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)
#plt.errorbar(BigXlist[1], Avg_ylist[1], yerr = errlist[1], color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)
##plt.errorbar(BigXlist[2], Avg_ylist[2], yerr = errlist[2], color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)
##plt.errorbar(BigXlist[3], Avg_ylist[3], yerr = errlist[3], color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)
##plt.errorbar(BigXlist[4], Avg_ylist[4], yerr = errlist[4], color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)

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



def Pref_Deg_dist(k, m):
    A = 2*m*(m+1)
    B = (k+2)*(k+1)*k
    y = A/B
    return y

def Rand_Deg_dist(k, m):
    A = m**(k-m)
    B = (1+m)**(1+k-m)
    y = A/B
    return y

#arO = np.arange(2, 400, 0.01)
#p0 = np.array([20])
#p, cov = opt.curve_fit(Deg_dist, bin_k2[0], bin_k2[1], p0)
#plt.plot(arO, Deg_dist(arO, p[0]), zorder=10,color = 'red')

func = Pref_Deg_dist
colour = ["navy", "orangered", "forestgreen", "firebrick", "blueviolet"]

theor_func = []

for j in range(len(g)):
    arO = np.arange(g[j], np.amax(LogBin[j][0]), 0.01) # Bins[j][0] BigXlist[j]
    p0 = np.array([2])
    #p, cov = opt.curve_fit(func, BigXlist[j], Avg_ylist[j], p0) # Bins[j][0], Bins[j][1]
    p, cov = opt.curve_fit(func, LogBin[j][0][1:], LogBin[j][1][1:], p0) # Bins[j][0], Bins[j][1]
    plt.plot(arO, func(arO, g[j]), '--', zorder=10, color = colour[j]) # PLOTTED
    #plt.plot(arO, func(arO, p[0]), '--', zorder=10, color = colour[j])#, linewidth = 1) # FITTED
    
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


from scipy import stats
 
for j in range(len(g)):
    KS = stats.ks_2samp(theor_func[j], LogBin[j][1])
    print("KS Test", j)
    print(KS)




#plt.figure()
#
#
#def Pref_Deg_dist(k, m):
#    A = 2*m*(m+1)
#    B = (k+2)*(k+1)*k
#    y = A/B
#    return y
#
#xA = np.arange(2, 1000001)
#plt.plot(xA, Pref_Deg_dist(xA, 2), 'x')
#plt.grid()
#plt.xscale("log")
#plt.yscale("log")
#plt.show()






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