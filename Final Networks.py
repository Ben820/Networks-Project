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
from collections import Counter
#%%
""" MIXED ATTACHMENT """
# Algorithm/ Network definitions
def Increment():
    degree.append(0)
    vertex_con.append([])
    vertices.append(vertices[-1] + 1)

def Edge():
    for a in range(m):
        degree[-1] += 1
        
        # Chooses between Preferential and Random 
        x = np.random.choice([1,2], p = [q,1-q]) 
        if x == 1:
            Exist_ver = wattachlist[np.random.randint(len(wattachlist))]#[:-1]))]
    
            if a >= 1:
                while np.any(Exist_ver == np.array(vertex_con[-1])) == True:
                    Exist_ver = wattachlist[np.random.randint(len(wattachlist))]#[:-1]))]
    
            """ Existing vertex cannot be chosen to be the new node 
            Remove eself loops """
            while Exist_ver == vertices[-1]:
                Exist_ver = wattachlist[np.random.randint(len(wattachlist))]
            
            # Adds the index of the existing vertex to the list of new vertex connections
            vertex_con[-1].append(Exist_ver)  
            # Adds the index of the new vertex to the list of existing vertex connections              
            vertex_con[Exist_ver-1].append(vertices[-1]) # Old vertices
            degree[Exist_ver-1] += 1
            
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
""" Initialisation """
Nng = [3,5,9,17,33,65]
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
Data = {}
# R-1 is the number of separate networks simulated  
# I is the number of single grain additions (i.e. total time)
R = 2
#t = 100000
t = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
t_test = np.array([10, 100, 1000, 10000, 100000])
g = np.array([2,4,8,16,32,64])
Nng = [3,5,9,17,33,65]
""" If Task 4 change g for t """
for j in range(len(t)):
    for h in range(1,R):
        #Nn = Nng[j]
        Nn = 3
        #m = g[j]
        m = 2
        # q is the probability the edge is joined using preferential attachment 
        q = 1
        A(t[j]) # A(t[j]) A(t)
        Data[t[j],h] = [degree]#, vertex_con]
                                
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
#%% 
""" Working Code Test """
Test = []
for j in range(len(t_test)):
    A = np.sum(Data[t_test[j],1][0])/len(Data[t_test[j],1][0])
    Test.append(A/2)
#%%
""" Logbin Data """
# Need to first run logbin file - Credit: Max Falkenberg McGillivray
scale = 1.3
s0 = False # whether or not to include s = 0 avalanches 

bin2 = []
bin4 = []
bin8 = []
bin16 = []
bin32 = []
bin64 = [] 

A = ['bin2', 'bin4', 'bin8', 'bin16', 'bin32', 'bin64']
binlist = [bin2, bin4, bin8, bin16, bin32, bin64]

R = 51

for h in range(1,R):
    bin2.append(logbin(data[2,h][0],scale))
    bin4.append(logbin(data[4,h][0],scale))
    bin8.append(logbin(data[8,h][0],scale))
    bin16.append(logbin(data[16,h][0],scale))
    bin32.append(logbin(data[32,h][0],scale))
    bin64.append(logbin(data[64,h][0],scale))
#%%
""" Maximum degree distribution; m = 2, N = 2^10 - 2^17 """
scale = 1.3
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

t = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
ts = [128, 512, 2048, 8192, 32768, 131072]
R = 101

for v in range(len(maxlist)):
    for h in range(1,R):
        maxlist[v].append(np.amax(np.array(data[t[v],h][0])))

stdev = []
avg = []

for r in range(len(maxlist)):
    stdev.append(np.std(np.array(maxlist[r])))#/np.sqrt(len(maxlist[r])))
    avg.append(np.mean(np.array(maxlist[r])))
#%% 
""" Variable m, Fixed N """
unbin2 = []
unbin4 = []
unbin8 = []
unbin16 = []
unbin32 = [] 
unbin64 = []

for h in range(1,R):
    unbin2.append(data[2,h][0])
    unbin4.append(data[4,h][0])
    unbin8.append(data[8,h][0])
    unbin16.append(data[16,h][0])
    unbin32.append(data[32,h][0])
    unbin64.append(data[64,h][0])

# Concatenate unbin lists 
UnBin2 = np.concatenate(unbin2)
UnBin4 = np.concatenate(unbin4)
UnBin8 = np.concatenate(unbin8)
UnBin16 = np.concatenate(unbin16)
UnBin32 = np.concatenate(unbin32)
UnBin64 = np.concatenate(unbin64)

UnBinlist = [UnBin2, UnBin4, UnBin8, UnBin16, UnBin32, UnBin64]
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
s0 = False # whether or not to include s = 0 avalanches 
""" Fixed N variable m """
LogBin = []
for j in range(len(g)):
    LogBin.append(logbin(UnBinlist[j],scale))

"""Fixed m variable N """
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
            else: 
                sample.append(0)
        Sampl.append(sample)  
    """ Need to add zero entries to lists in Sampl until all at same length """
    err_y = []
    avg_y = []
    for s in range(len(Sampl)):
        err_y.append(np.std(Sampl[s])/np.sqrt(np.size(Sampl[s])))
        avg_y.append(np.mean(Sampl[s]))
    errlist.append(err_y)
    Avg_ylist.append(avg_y)
    BigXlist.append(Big_xlist.tolist())
#%%
""" Testing Error Analysis Code """
x1 = [0,1,2,3]
x2 = [1,2,3,4]
x3 = [2,4,5]
y1 = [10,20,30,40]
y2 = [11,21,31,41]
y3 = [12,22,32]

big_x = [x1, x2, x3]
big_y = [y1, y2, y3]
Big_x = np.unique(np.concatenate(big_x))

Samp = []
for a in range(len(Big_x)):
    sample = []
    for v in range(len(big_y)):
        if Big_x[a] in big_x[v]:
            sample.append(big_y[v][big_x[v].index(a)])
    Samp.append(sample)
#%%
def Pref_deg_dist(k, m):
    A = 2*m*(m+1)
    B = (k+2)*(k+1)*k
    y = A/B
    return y

def Rand_deg_dist(k, m):
    return ((m/(m+1))**(k-m))*(1/(1+m))

def Mixed_deg_dist_twothirds(k, m):
    A = 6*m*(2*m+2)*(2*m+1)
    C = (k+m+3)*(k+m+2)*(k+m+1)*(k+m)
    return A/C

def Mixed_deg_dist_onehalf(k, m):
    return (12*m*(3*m+3)*(3*m+2)*(3*m+1))/((k+2*m+4)*(k+2*m+3)*(k+2*m+2)*(k+2*m+1)*(k+2*m))
#%%
""" Averaged Logbin + Errors """
plt.figure()

""" Combining different runs into a huge dataset which is logbinned to give x and y """

func = Mixed_deg_dist_onehalf    #Mixed_deg_dist_twothirds
colours = ["royalblue","orange","green","crimson","purple","steelblue"]
colour = ["navy", "orangered", "forestgreen", "firebrick", "blueviolet", "steelblue"]
labeLs = ["m = 2", "m = 4", "m = 8", "m = 16", "m = 32", "m = 64"]
theor_func = []

extra = [10,20,30,50,100,200]
for j in range(len(g)):
    # Normal Plot
    plt.plot(LogBin[j][0], LogBin[j][1], 'x-', label = labeLs[j])
    plt.errorbar(LogBin[j][0], LogBin[j][1], yerr = errlist[j], color = colours[j], fmt='o', mew=1, ms=0.2, capsize=6)

    arO = np.arange(g[j], np.amax(LogBin[j][0]), 1)  #+extra[j]
    plt.plot(arO, func(arO, g[j]), '--', zorder=10, color = colour[j]) 
        
    theor_func.append(func(LogBin[j][0], g[j]))
    
    # Pn vs Pinf Preferential 
#    plt.plot(theor_func[j], LogBin[j][1], 'x-', label = labeLs[j])
#    plt.errorbar(theor_func[j], LogBin[j][1], yerr = errlist[j], color = colours[j], fmt='o', mew=1, ms=0.2, capsize=6)
#
#plt.plot(np.arange(0.000000003,0.5,0.000001), np.arange(0.000000003,0.5,0.000001), color = "k", label = "y = x")

#for j in range(len(g)):
#    
#    plt.plot(LogBin[j][0], LogBin[j][1], 'x-', label = labeLs[j])
#    plt.errorbar(LogBin[j][0], LogBin[j][1], yerr = errlist[j], color = colours[j], fmt='o', mew=1, ms=0.2, capsize=6)

plt.xlabel("Degree, $k$", size = "15")
#plt.xlabel("Degree probability, $p_N(k)$", size = "15")
plt.ylabel("Degree probability, $p_\infty(k)$", size = "15")
plt.ylabel("Degree probability, $p_N(k)$", size = "15")
plt.xscale("log")
plt.yscale("log")
plt.legend()
#plt.grid()
#plt.savefig("Mixed Deg Dist 12.png", dpi = 1000)
plt.show()

import scipy.stats as st
for j in range(len(g)):
    Pear = st.pearsonr(theor_func[j], LogBin[j][1])
    print("Pearsons Coefficient m =", g[j])
    print(Pear)
#%%
plt.figure()
x = np.arange(0.000000003,0.5,0.000001)
plt.plot(x,x)
plt.xscale("log")
plt.yscale("log")
plt.show()
#%%
""" Statistical Tests; Kolmogorov Smirnov Test """
from scipy import stats

for j in range(len(g)):
    KS = stats.ks_2samp(theor_func[j], LogBin[j][1])
    #KS = stats.kstest(LogBin[j][1], Rand_deg_dist())
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
trlist = np.log(np.array([128, 512, 2048, 8192, 32768, 131072])).tolist()
colours = ["lightseagreen", "orangered", "forestgreen", "firebrick", "blueviolet", "navy"]

#func = Rand_deg_dist

for z in [5,4,3,2,1,0]:
    ycoll = func(LogBin[z][0], 2)
#    plt.plot(LogBin[z][0], LogBin[z][1], 'x-', color = colours[z], label = labels_redu[z])
#    plt.errorbar(LogBin[z][0], LogBin[z][1], yerr = errlist[z], color = colours[z], fmt='o', mew=1, ms=0.2, capsize=6)
    """ Data Collpase Preferential """
#    plt.plot(LogBin[z][0]/tlist[z], LogBin[z][1]/ycoll, 'x', color = colours[z], label = labels_redu[z])
#    plt.errorbar(LogBin[z][0]/tlist[z], LogBin[z][1]/ycoll, yerr = errlist[z]/ycoll, color = colours[z], fmt='o', mew=1, ms=0.2, capsize=6)

    """ Data Collapse Random """
    plt.plot(LogBin[z][0]/trlist[z], LogBin[z][1]/ycoll, 'x', color = colours[z], label = labels_redu[z])
    plt.errorbar(LogBin[z][0]/trlist[z], LogBin[z][1]/ycoll, yerr = errlist[z]/ycoll, color = colours[z], fmt='o', mew=1, ms=0.2, capsize=6)


#func = Pref_deg_dist
#colour = ["navy", "orangered", "forestgreen", "firebrick", "blueviolet", "red"]
#
#theor_func = []
#
#for j in range(len(g)):
#    arO = np.arange(g[j], np.amax(LogBin[j][0]+10), 0.01) 
#    plt.plot(arO, func(arO, g[j]), '--', zorder=10, color = colour[j]) 
#    
#    theor_func.append(func(arO, g[j]))

# Preferential 
#plt.xlabel(r"$k/\sqrt{N}$", size = "15")
#plt.ylabel(r"$p_N(k)/p_\infty(k)$", size = "15")
# Random 
plt.xlabel(r"$k/ln(N)$", size = "15")
plt.ylabel(r"$p_N(k)/p_\infty(k)$", size = "15")

plt.xscale("log")
plt.yscale("log")
plt.ylim(0.04, 1.75)
plt.legend()
#plt.grid()
#plt.savefig("Task 4 Rand Data Collapse 1 zoom.png", dpi = 1000)
plt.show()
#%%
""" Maximum degree vs N for variable N fixed m = 2 """
# Preferential 
plt.figure()

# Pref
#plt.plot((t), avg, 'x')
#plt.errorbar((t), avg, yerr = (stdev), fmt='o', mew=1, ms=0.2, capsize=6)
#
## Rand
#plt.plot(T1, Avg1, 'x')
#plt.errorbar(T1, Avg1, yerr = (Stdev1), fmt='o', mew=1, ms=0.2, capsize=6)

# Mixed 23
plt.plot(T2, Avg2, 'x')
plt.errorbar(T2, Avg2, yerr = (Stdev2), fmt='o', mew=1, ms=0.2, capsize=6)

# Mixed 12
plt.plot(T, Avg, 'x')
plt.errorbar(T, Avg, yerr = (Stdev), fmt='o', mew=1, ms=0.2, capsize=6)

#plt.plot(np.sqrt(t), avg, 'x')
#plt.errorbar(np.sqrt(t), avg, yerr = (stdev), fmt='o', mew=1, ms=0.2, capsize=6)

def Plaw(x, m, k):
    y = m*(x**k)
    return y

def Linear(x,m,c):
    y = m*x+c
    return y

def Quad(x,m):
    A = -1 + np.sqrt(1+4*x*m*(m+1))
    y = A/2
    return y

def RandM(x,m):
    A = np.log(x)
    B = np.log(m)-np.log(m+1)
    return m - x/B

arO = np.arange(100,135000, 1)
#arO = np.arange(10,380, 1)
#arO = np.arange(4.5,12, 0.01)


p0 = np.array([4.745, 0.334])
p1 = np.array([5.1088, 0.2814])# pm 0.003397
#p, cov = opt.curve_fit(Plaw, (t), avg, p0) #np.sqrt
#plt.plot(arO, Linear(arO, p0[0], p0[1]), '--', zorder=10, color = 'navy', label = "Theoretical Power Law")
#plt.plot(arO, Quad(arO, 2), '--', zorder=10, color = 'navy', label = "Theoretical Distribution")
#plt.plot(arO, Plaw(arO, p0[0], p0[1]), '--', zorder=10, color = 'navy', label = "Approximate Theoretical \nPower Law")
#plt.plot((arO), RandM((arO), 2), '--', zorder=10, color = 'navy', label = "Theoretical Distribution")
plt.plot((arO), Plaw((arO), p0[0], p0[1]), '--', zorder=10, color = 'navy', label = "Estimated Theoretical \nDistribution q = 2/3")
plt.plot((arO), Plaw((arO), p1[0], p1[1]), '--', zorder=10, color = 'crimson', label = "Estimated Theoretical \nDistribution q = 1/2")

#plt.xlabel(r"$\sqrt{N}$", size = "15")
plt.xlabel(r"$ln(N)$", size = "15")
plt.xlabel(r"$N$", size = "15")
plt.ylabel(r"$\langle k_1 \rangle$", size = "15")
plt.xscale("log")
plt.yscale("log")
plt.legend()
#plt.savefig("Task 4 Mixed blah.png", dpi = 1000)
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))): #zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1])) #prints value and uncertainty, f is decimal places and G is sig figs       

import scipy.stats as st
st.pearsonr((t)**p0[1], avg)
