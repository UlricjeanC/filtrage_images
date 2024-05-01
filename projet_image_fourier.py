# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:49:16 2024

@author: UlricJeanC
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cbook, cm
from matplotlib.colors import LightSource


def normMatImage(Mat):
    Mat = np.where(Mat > 0, Mat, 0)
    m = np.max(Mat)
    Mat = (Mat / m) * 255
    Mat = np.int_(np.ceil(Mat))
    return Mat

def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


# fonctionne mais inutilisable

MEMO_DFT = {}

def DFT(f):
    M, N = f.shape
    
    F_real = np.zeros((M,N))
    F_imag = np.zeros((M,N))
    
    #Mat_p = np.exp( -1j*2*np.pi* np.tile( np.arange(0,N,1) ,(M,1)) )
    Mat_p = np.exp( -1j*2*np.pi* np.arange(0,M,1) / M )
    
    #Mat_q = np.exp( -1j*2*np.pi* np.arange(0,M,1) )
    Mat_q = np.exp( -1j*2*np.pi* np.tile( np.arange(0,N,1) ,(M,1)) / N )
    
    for p in range(M):
        
        temp = np.power ( Mat_p , p )
        
        for q in range(N):
            
            #F = np.sum ( np.sum( temp * f ,axis = 1 ) * np.power( Mat_q , q ) )
            F = np.sum ( np.sum( np.power( Mat_q , q ) * f ,axis = 1 ) * temp )
            
            F_real[p,q] = F.real
            F_imag[p,q] = F.imag
            
    return F_real, F_imag


#%%      
image = Image.open("C:/Users/lupus/Desktop/images_python/clock.jpg")
image = image.convert('L')
Mat_image = np.asarray(image)
F_image = np.fft.fft2(Mat_image)

dim = Mat_image.shape
Mat = np.zeros(dim)
"""
for i in range(dim[0]):
    for j in range(dim[1]):
        if (i-dim[0]/2)**2 + (j-dim[1]/2)**2 < 100 :
            Mat[i,j] = 1
"""
Mat[190:200,110:150] = 1
#Mat = np.random.normal(0,10,size=dim)
F = np.fft.fft2(Mat)

"""
F_sol = F * F_image
for i in range(dim[0]):
    for j in range(dim[1]):
        if (i-dim[0]/2)**2 + (j-dim[1]/2)**2 < 75000 :
            F_sol[i,j] = 0
        else:
            F_sol[i,j] = F_image[i,j]
"""

sigma1 = 10
sigma2 = 5
lim1 = 2*np.pi*np.sqrt(sigma1)
lim2 = 2*np.pi*np.sqrt(sigma2)
lim = max(lim1,lim2)
X = np.linspace(-lim,lim,dim[0])
Y = np.linspace(-lim,lim,dim[1])

X = np.tile( X , (dim[1],1) ).T
Y = np.tile( Y , (dim[0],1) )

theta = np.pi/4 
A = 1 / ( 2 * np.pi * ( np.sqrt(sigma1) + np.sqrt(sigma2) ) )

a = ((np.cos(theta) ** 2)/(2*sigma1)) + ((np.sin(theta) ** 2)/(2*sigma2))
b =  -((np.sin(2*theta) ** 2)/(4*sigma1)) + (np.sin(2*theta) ** 2)/(4*sigma2)
c = ((np.sin(theta) ** 2)/(2*sigma1)) + ((np.cos(theta) ** 2)/(2*sigma2))

Noyau_Gauss = A * np.exp( - ( a*( X*X ) + 2*b*(X*Y) + c*(Y*Y) ) )


F_sol = Noyau_Gauss * F_image


f_image = np.fft.ifft2(F_image)
f_sol = np.fft.ifft2(F_sol)

#%%

F_utile = np.log( np.sqrt( (F_sol*F_sol.conjugate()).real ) )
f_utile = np.sqrt( (f_sol*f_sol.conjugate()).real ) 
#f_utile = np.sqrt( (f_image*f_image.conjugate()).real ) 



# images initiales
plt.matshow(Mat_image, cmap = 'binary', interpolation='none')
plt.colorbar()
plt.show()


plt.matshow(Noyau_Gauss, cmap = 'plasma', interpolation='none')
plt.colorbar()
plt.show()


# spectre
plt.matshow(F_utile, cmap = 'plasma', interpolation='none')
plt.colorbar()
plt.show()

# Image produite
plt.matshow(f_utile, cmap = 'binary', interpolation='none')
plt.colorbar()
plt.show()
