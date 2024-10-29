"""
Matlab_kernel_fft(r,nb_channels,M,N ) returns the FFT kernel associated with the Matlab imresize zoom-out of factor r.
Author : Emile Pierret (29/10/2024)
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