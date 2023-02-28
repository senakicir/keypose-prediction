#!/usr/bin/env python3
#-----------------------------------------------------------------------------
#                         
#=============================================================================

#%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------
#%

import numpy as np
import matplotlib.pyplot as plt

#%---------------------------------------------------------------------------
#                                
#-----------------------------------------------------------------------------
#%%
def interpSegm(zs,ys,t_start,t1):
    p  = 0.0
    q  = 1.0
    y0 = ys[t_start,:]
    y1 = ys[t1,:]
    dt = 1.0/(t1-t_start)
    for t in range(t_start,t1):
        zs[t,:]= q*y0+p*y1
        p   += dt
        q   -= dt
  
def argMaxDist(target,t_start,t_end,eps=0.0,robustP=False):
    
    if(t_end<=(t_start+1)):
        return -1
    
    p  = 0.0
    q  = 1.0
    y0 = target[t_start,:]
    y1 = target[t_end,:]
    dt = 1.0/(t_end-t_start)
    
    # maxD is the error threshold (maximum distance)
    maxD = eps
    #if no error passes error threshold this function will return -1 as maxT
    maxT = -1
    
    #try every t
    for t in range(t_start,t_end):
        if(robustP):
            d    = np.median(np.abs(target[t,:]-(q*y0+p*y1)))
        else:
            d    = np.linalg.norm(target[t,:]-(q*y0+p*y1))
        #if an error manages to exceed the previous maxD, save that as the new maxD
        if(d>maxD):
            maxD = d
            maxT = t
        p   += dt
        q   -= dt
        
    return maxT

        
#%%
def linInterpolate(ys,ts):
    
    zs = np.zeros_like(ys)
    t_start = ts[0]
    for t1 in ts[1:]:
        interpSegm(zs,ys,t_start,t1)
        t_start=t1
    zs[t1]=ys[t1]
    return zs
    
    
def recSubdivide(target,eps=1e-3,robustP=False):
    #returns indices of keyposes (their locations)

    #target: N, 66
    num_frames = target.shape[0]
    t_start = 0
    t_end = target.shape[0]-1
    
    t_mask = np.zeros(num_frames)
    #first and last time steps are 1.
    t_mask[t_start]=1
    t_mask[t_end]=1
    
    #call recursive function to find location of keyposes
    auxSubdivide(target,t_start,t_end,t_mask,eps,robustP)
    time_indice_list = []
    
    #if there was a keypose indicated in t_mask, add it to
    #time indice list
    for t,x in enumerate(t_mask):
        if(x>0):
            time_indice_list.append(t)
    return time_indice_list
    

def auxSubdivide(ys,t_start,t_end,t_mask,eps,robustP):
    t_lowerror = argMaxDist(ys,t_start,t_end,eps=eps)
    if (t_lowerror>=0):
        t_mask[t_lowerror]=1
        auxSubdivide(ys,t_start,t_lowerror,t_mask,eps,robustP)
        auxSubdivide(ys,t_lowerror,t_end,t_mask,eps,robustP)

if __name__ == "__main__":  
    ys = np.random.randn(100,2)
    zs = np.random.randn(100,2)
    t_start = 20
    t1 = 30
    interpSegm(zs,ys,t_start,t1)
    plt.clf()
    plt.plot(range(t_start,t1),zs[t_start:t1,0],'-r')
    plt.plot(range(t_start,t1),zs[t_start:t1,1],'-g')
    plt.plot(range(t_start,t1+1),ys[t_start:t1+1,0],'rx')
    plt.plot(range(t_start,t1+1),ys[t_start:t1+1,1],'gx')
    plt.show()

    ns = 100
    ys = np.zeros((ns,2))
    ys[:,0] = np.sin(np.array(np.arange(0,2,2/ns)))
    ys[:,1] = np.cos(np.array(np.arange(0,2,2/ns)))
    ts = recSubdivide(ys,eps=1e-1,robustP=True)
    zs = linInterpolate(ys,ts)
    plt.clf()
   # plt.plot(ys)
    plt.plot(zs)
    plt.show()
    plt.close()
    
