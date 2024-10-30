"""
p = perdecomp(u) computes Lionel Moisan's periodic component of u.
p,s = perdecomp(u) computes Lionel Moisan's periodic plus smooth
decomposition of u = p+s
Works for gray-valued and RGB color images. 
Author : Bruno Galerne (2009/03/17), adapted to pytorch by Emile Pierret

For the resolution of Poisson's equation Laplacian(u) = f see the book
'Numerical Recipes: the Art of Scientific Computing'
"""


import torch
import numpy as np
def perdecomp(u) :
    #Image resolution
    nc,M,N = u.shape
    #Compute LIU the interior Laplacian of u:
    zc = torch.zeros(nc,M, 1)
    zr = torch.zeros(nc,1, N)
    ILU = torch.zeros(nc,M,N)
    ILU[:,:(M-1),:] += u[:,:(M-1),:] - u[:,1:,:]
    ILU[:,1:,:] += u[:,1:,:] - u[:,:(M-1),:]
    ILU[:,:,:(N-1)] += u[:,:,:(N-1)] - u[:,:,1:]
    ILU[:,:,1:] += u[:,:,1:] - u[:,:,:(N-1)]

    #Fourier transform
    F = torch.fft.fft2(ILU,dim=(1,2))

    cx = torch.tensor([2*np.cos((2*np.pi/M)*k) for k in range(M)])
    cy = torch.tensor([2*np.cos((2*np.pi/N)*l) for l in range(N)])

    [CY,CX] = torch.meshgrid(cx,cy, indexing="ij")


    C = 4-CX-CY

    C[0,0] = 1


    C = C.repeat([nc,1,1])

    F = F/C

    F[:,0,0] = torch.sum(u,(1,2))

    P = torch.real(torch.fft.ifft2(F,dim=(1,2)))


    return P

def perdecomp_fft(u) :
    #Image resolution
    nc,M,N = u.shape
    #Compute LIU the interior Laplacian of u:
    zc = torch.zeros(nc,M, 1)
    zr = torch.zeros(nc,1, N)
    ILU = torch.zeros(nc,M,N)
    ILU[:,:(M-1),:] += u[:,:(M-1),:] - u[:,1:,:]
    ILU[:,1:,:] += u[:,1:,:] - u[:,:(M-1),:]
    ILU[:,:,:(N-1)] += u[:,:,:(N-1)] - u[:,:,1:]
    ILU[:,:,1:] += u[:,:,1:] - u[:,:,:(N-1)]

    #Fourier transform
    F = torch.fft.fft2(ILU,dim=(1,2))

    cx = torch.tensor([2*np.cos((2*np.pi/M)*k) for k in range(M)])
    cy = torch.tensor([2*np.cos((2*np.pi/N)*l) for l in range(N)])

    [CY,CX] = torch.meshgrid(cx,cy, indexing='ij')


    C = 4-CX-CY

    C[0,0] = 1


    C = C.repeat([nc,1,1])

    F = F/C

    F[:,0,0] = torch.sum(u,(1,2))

    return F





