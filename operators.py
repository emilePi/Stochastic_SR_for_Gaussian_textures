#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:43:45 2023

@author: pierret
"""


import numpy as np
import torch



##Keys' function
def phi(x) :
    absx = np.abs(x)
    absx2 = absx*absx
    absx3 = absx2*absx
    f = (1.5*absx3 - 2.5*absx2 + 1)*( absx <= 1) + (-0.5*absx3 + 2.5*absx2 - 4*absx + 2)*(1 < absx)*(absx <= 2)
    return f

## Convolution kernel

def Matlab_kernel_fft(r,nb_channels,M,N ) :
    s = 1/r
    y = -1/2*(1-1/s)
    k = 0
    K=[]
    K.append(s*phi(s*(y-k)))
    k=1
    while ( phi(s*(y-k)) != 0 ) or ( phi(s*(y+k)) != 0 ) :
        K.append(s*phi(s*(y-k)))
        K.insert(0,s*phi(s*(y+k)))
        k=k+1

    kernel = np.zeros((len(K),len(K)))

    for k in range(len(K)) :
        for l in range(len(K)) :
            kernel[k,l] = K[k]*K[l]

    cM = torch.zeros(M)
    for k in range(len(K)) :
        cM[k-len(K)//2-r+1] = K[k]

    cN = torch.zeros(N)
    for k in range(len(K)) :
        cN[k-len(K)//2-r+1] = K[k]

    cM = cM/torch.sum(cM)
    cN = cN/torch.sum(cN)
    ##FFT for one channel

    cfft2 = torch.fft.fft2(torch.outer(cM,cN))

    ##FFT for nb_channels

    c3fft = torch.stack([cfft2]*nb_channels)

    return c3fft

def identity_fft(nb_channels,M,N ) :
    c3fft = 1. + 0*torch.zeros(nb_channels,M,N,dtype = torch.complex64)
    return c3fft

def box_filter_fft(n,nb_channels,M,N):
    g = n*n
    
    c = torch.zeros(M,N)
    
    for i in range(-n//2+1,n//2+1) :
        for j in range(-n//2+1,n//2+1) :
            c[i,j] += 1/g
    ##FFT for one channel

    cfft2 = torch.fft.fft2(c)

    ##FFT for nb_channels

    c3fft = torch.stack([cfft2]*nb_channels)

    return c3fft


def Gaussian_blur_fft(sigblur,nb_channels,M,N) :
    # Define Gaussian blur operator:
    s = int(3*sigblur)
    w = 2*s+1
    kernel = np.zeros(w)
    for t in np.arange(w):
     kernel[t] = np.exp(-(t-s)**2/(2*sigblur**2))
    kernel /= sum(kernel)
    gausskernel = np.zeros((w,w))
    for t1 in np.arange(w):
     for t2 in np.arange(w):
         gausskernel[t1,t2] = kernel[t1]*kernel[t2]
    
    cM = torch.zeros(M)
    cN = torch.zeros(N)
    for k in range(w) :
        cM[k-w//2] = kernel[k]
        cN[k-w//2] = kernel[k]
    
    # kernelM = np.zeros(M)
    # for t in np.arange(M):
    #  kernelM[t] = np.exp(-(t-s)**2/(2*sigblur**2))
     
    #  kernelN = np.zeros(N)
    # for t in np.arange(N):
    #  kernelN[t] = np.exp(-(t-s)**2/(2*sigblur**2))
    # kernelM /= sum(kernelM)
    # kernelN /= sum(kernelN)
    # for k in range(M) :
    #     cM[k-M//2] = kernelM[k]
    # for k in range(N) :
    #     cN[k-N//2] = kernelN[k]


    cfft2 = torch.fft.fft2(torch.outer(cM,cN))
    
    ##FFT for nb_channels
    
    c3fft = torch.stack([cfft2]*nb_channels)
    
    return c3fft
