#!/usr/bin/env python3
#-----------------------------------------------------------------------------
#                         
#=============================================================================

#%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------
#%%

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

from scipy.optimize import minimize,least_squares 

from .pline import recSubdivide

#%----------------------------------------------------------------------------
#                                
#-----------------------------------------------------------------------------
#%%
class OptimModule(nn.Module):
    
    def setState(self,x):
        
        self.params.data[:]=torch.FloatTensor(x)
        
    def getState(self):

        state = self.params.detach().cpu().numpy()
        return np.array(state,dtype=np.float64)
    
    def func (self,x,target):
        
        output = self(torch.FloatTensor(x))
        if(target is not None):
            output -= target
        output = output.flatten()
        return output.detach().cpu().numpy()
        
    def objF (self,x,target,lossF=F.mse_loss):
        
        output = self(torch.FloatTensor(x))
        loss   = lossF(output,target)
        
        return loss.item()
    
    def grdF(self,x,target,lossF=F.mse_loss):
        
        inp         = torch.zeros(x.shape[0],dtype=torch.float,requires_grad=True)
        inp.data[:] = torch.FloatTensor(x)   
        
        grad = self.gradient(inp,target,lossF=lossF)
        
        return grad.detach().cpu().numpy()/x.size
    
    def jacF(self,x,target):
        
        inp         = torch.zeros(x.shape[0],dtype=torch.float,requires_grad=True)
        inp.data[:] = torch.FloatTensor(x)
        
        jacT = self.jacobian(inp,target)
        
        return jacT.detach().cpu().numpy()
    
    #------------------------------------
    #          Minimization
    #------------------------------------
    def minimize (self,x,target,method='BFGS',nIt=1000,ftol=1000,eps=1e-5,verbose=0):
        
        tgt = torch.FloatTensor(target)
        F = (lambda x : self.objF(x,tgt))
        G = (lambda x : self.grdF(x,tgt))
        
        opt =  dict([('disp', verbose), ('maxiter', nIt),('eps',eps),('gtol',ftol)])
        
        sol=minimize(F,x,method=method,jac=G,options=opt)
        self.setState(sol.x)
        
        return np.array(sol.x,dtype=np.float64)
    
    def lsqmin   (self,x,target,nIt=1000,verbose=0,ftol=1e-8):
        
        tgt = torch.FloatTensor(target)
        F = (lambda x : self.func(x,tgt))
        J = (lambda x : self.jacF(x,tgt))
        
        sol=least_squares(F,x,jac=J,max_nfev=nIt,verbose=verbose,ftol=ftol,gtol=ftol)
        self.setState(sol.x)
        
        return np.array(sol.x,dtype=np.float64)
    
    
    def sgd (self,target,nIt=100,lossF=F.mse_loss,lr=1.0):
        
        optimizer = optim.SGD(self.parameters(),lr=lr)
        for i in range(nIt):
            optimizer.zero_grad()
            output = self()
            loss   = lossF(output, target)
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print('iter ', i, loss.item())
                
    def gradient (self,x,target,lossF=F.mse_loss):
        
        self.zero_grad()
        output = self(x)
        loss   = lossF(output,target)
        loss.backward()
        
        return x.grad
        
    def jacobian (self,x,target):
       
        x1 = x.view(-1,1)
        y1 = self(x1).view(-1,1)
        n  = x1.size(0)   # Number of variables
        m  = y1.size(0)   # Number of equations        
        
        jacT   = torch.zeros((n,m),dtype=torch.float,requires_grad=False)     
        for i in range(m):                                                                                                                     
            output    = torch.zeros(m,1)                                                                                                          
            output[i] = 1.0
            jacT[:,i:i+1] = torch.autograd.grad(y1,x1,grad_outputs=output,retain_graph=True,allow_unused=True)[0]
        
        return jacT.t()
    #------------------------------------
    #          Weighted averages
    #------------------------------------
    def residuals(self,x,ys):
        
        nb,num_frames,num_curves = ys.shape
        output = self(torch.FloatTensor(x)).detach().numpy()
        output = output.reshape(1,num_frames,num_curves).repeat(nb,0)
        return output-ys
    
    def weigths(self,x,ys,alpha=2.0,eps=1e-10):
        
        resid1     = self.residuals(x,ys)
        resid2     = resid1*resid1
        nb,num_frames,num_curves   = resid1.shape
    
        weigths = np.zeros((nb,num_frames,num_curves),dtype=np.float64)    
        
        for curvN in range(num_curves):
            curvR  = resid2[:,:,curvN]
            curvM  = np.maximum(alpha*np.median(curvR,axis=1,keepdims=True),eps)
            curvW  = np.exp(- (curvR / curvM))
            curvM  = np.mean(curvW,axis=0,keepdims=True)
            curvW /= curvM
            
            weigths[:,:,curvN] = curvW
            
        return weigths
    
    def averageCurves(self,ys,nodes=None,alpha=2.0):
    
        nb,num_frames,num_curves = ys.shape
        zs = np.zeros((num_frames,num_curves),dtype=np.float64)
    
        if(nodes is not None):
            self.setState(nodes)
            ws=self.weigths(nodes,ys,alpha=alpha)
            assert((np.linalg.norm(np.sum(ws,axis=0)-nb))<1e-8)
            ys=ws*ys

        for c in range(num_curves):
            zs[:,c]=np.mean(ys[:,:,c],axis=0)
            
        return zs
    
    def gradDBG(self,x,target,lossF=F.mse_loss,eps=1e-2):
               
        grad   = self.gradient(x,target,lossF)
        output = self(x)
        loss0  = lossF(output,target).item()
        for i in range(x.size(0)):
            xi = x.data[i].item()
            x.data[i]=xi+eps
            output = self(x)
            loss1  = lossF(output,target)
            print('{:2d}: {:3f} {:3f}'.format(i,grad[i].item(),((loss1-loss0)/eps).item()))
            x.data[i]=xi
            
    def jacobDBG(self,x,target,eps=1e-2):
        
        output0 = self(x).flatten()
        jacT    = self.jacF(x,target).T
        for j,xj in enumerate(x):
            x.data[j]=xj+eps
            output1 = self(x).flatten()
            for i,v1 in enumerate(output1):
                v0 = output0[i].item()
                jval = jacT[i,j]
                if(abs(jval)>0.001):
                    print('{:2d} {:2d}: {:3f} {:3f}'.format(i,j,jval,((v1-v0)/eps).item()))
            x.data[j]=xj
    
#gauss = GaussianFit(num_base=args.nk,num_t=args.num_frames,target=zs,sigma=30.0)
#y1 = gauss.averageCurves(ys,nodes=x1)

#%% ---------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
#%%
class GaussianFit(OptimModule):
    
    def __init__(self,num_base=5,num_t=15,batch=1,target=None,sigma=5.0):
        
        super(GaussianFit, self).__init__()
        self.pi  = torch.tensor(np.float32(math.pi))
        self.nk  = num_base
        
        if(target is None):
            pass
        else:
            self.num_frames,self.num_curves = target.shape
            self.xs  = torch.linspace(0,self.num_frames-1,self.num_frames)
            params = self.initParams(target,sigma=sigma)
            self.params = Parameter(params,requires_grad=True)
        
    def initCoeffs(self,target,sigma=5.0):
        
        num_frames = self.num_frames
        nk = self.nk
        
        sigm = sigma*torch.ones(nk,dtype=torch.float32)
        
        xk = torch.linspace(0,num_frames-1,nk,dtype=torch.float)
        ik = np.linspace(0,num_frames-1,nk,dtype=np.long)
        M  = self.M(xk,sigm,xk).detach().numpy()
        yk = target[ik,:]
        
        wk = np.linalg.solve(M,yk)
        return torch.FloatTensor(xk),torch.FloatTensor(wk)
    
    def initParams(self,target,sigma=5.0):
        
        xs,ws = self.initCoeffs(target,sigma=sigma)
        num_frames,num_curves = target.shape
        nk    = self.nk
        
        params = torch.zeros((2+num_curves,nk),dtype=torch.float32)
        params[0,:]  = xs
        params[1,:]  = sigma
        params[2:,:] = ws.t()
        
        return params.flatten()

    def M (self,xk,sigm,xs):
         
        sigm.data = torch.clamp(sigm.data, min=0.0001)
    
        x_mu = xs.unsqueeze(1) - xk.unsqueeze(0)
        if (True):
            x_mu_sigma = 0.5 * x_mu * sigm.reciprocal()
            p = torch.exp(-x_mu_sigma**2)
        else:
            x_mu_sigma = x_mu * sigm.reciprocal()
            p2 = torch.exp(-0.5*x_mu_sigma**2)
            p1 = (sigm*torch.sqrt(2*self.pi)).reciprocal()
            p  = p1*p2
        
        return p
    
    def F (self,xk,sigm,yk,xs):
        
        M = self.M(xk,sigm,xs)  
        return M.mm(yk)
    
    def forward(self,params=None,xs=None):
        if(params is None):
            params = self.params
        if(xs is None):
            xs = self.xs
            
        num_curves = self.num_curves 
        params = params.view(2+num_curves,-1)
            
        
        xk   = params[0,:] 
        sigm = params[1,:] 
        yk   = params[2:,:].t()
        
        y = self.F(xk,sigm,yk,xs)
        return y
#%% ---------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
#%%
class LinearFit(OptimModule):
    
    def __init__(self, target, num_base=5,num_t=100,xk=None,yk=None,eps=0.0):
        
        super(LinearFit, self).__init__()
        
        self.pi  = torch.tensor(np.float32(math.pi))

        #initial locations are just linearly spaced
        #num_t is ns which is 100 by default
        self.xs  = torch.linspace(0,num_t-1, num_t, dtype=torch.float)
        self.nk  = num_base

        #target: N (num frames), 66(num joints)
        self.num_frames, self.num_curves = target.shape
        self.xs  = torch.linspace(0,self.num_frames-1,self.num_frames)
        params   = self.initParams(target,eps=eps, xk=xk,yk=yk)
        self.params = Parameter(params,requires_grad=True)
            
            
    def initParams(self,target, eps, xk=None,yk=None):
        
            
        # Recursive segmentation
        if(eps>0.0):
            ik = recSubdivide(target,eps=eps,robustP=True)
            xk = torch.FloatTensor(ik)
            nk = self.nk = len(ik)
        # Equidistant segmentation
        elif(xk is None):
            nk = self.nk
            xk = torch.linspace(0,self.num_frames-1,nk,dtype=torch.float)
            ik = np.linspace(0,self.num_frames-1,nk,dtype=np.long)
        # Externally supplied  
        else:
            nk = self.nk = len(xk)
            ik = np.maximum(0,np.minimum(self.num_frames-1,np.floor(xk)))
            xk = torch.FloatTensor(xk)
            
        target = torch.FloatTensor(target)
        if(yk is None):
            yk = target[ik,:].t()
        else:
            yk = torch.FloatTensor(yk)
                
        num_curves = self.num_curves
        params = torch.zeros((1+num_curves,nk),dtype=torch.float32)
 
        params[0,:]  = xk
        params[1:,:] = yk
        
        return params.flatten()
  
    def forward(self,params,xs=None):
        
        if(params is None):
            params = self.params
        if(xs is None):
            xs = self.xs
            
        num_curves = self.num_curves
        num_frames = self.num_frames
        
        params = params.view(1+num_curves,-1)
        xk   = params[0,:] 
        yk   = params[1:,:].t()
        ys   = torch.zeros([num_frames,num_curves],dtype=torch.float32)
        
        ones = torch.zeros(num_frames,dtype=torch.float32,requires_grad=False)
        nv   = 0
        x1   = 0.0
        y1   = yk[0,:]
        
        for i2,x2 in enumerate(xk):
            y2 = yk[i2,:]
            
            if(x2>x1):
            
                ones[:] = (x1<=xs)*(xs<x2)
                dt2  = xs - x1
                dt1  = x2 - xs
                dt0  = x2-x1
            
                l2   = dt2.view(num_frames,1).repeat(1,num_curves)
                l1   = dt1.view(num_frames,1).repeat(1,num_curves)   
                mask = ones.view(num_frames,1).repeat(1,num_curves)
                nv  += torch.sum(mask)
                ys  += mask*((l1*y1+l2*y2)/ dt0)
            
                x1   = x2
                y1   = y2
         
       
        if(x2<num_frames):
            t2 = np.int(np.floor(x2.detach().numpy()))
            ys[t2:num_frames,:]=yk[-1,:].view(1,num_curves).repeat(num_frames-t2,1)
               
        return ys