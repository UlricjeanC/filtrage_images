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


def normMatImage(Mat,maxVal=255):
    Mat = np.where(Mat > 0, Mat, 0)
    m = np.max(Mat)
    Mat = (Mat / m) * maxVal
    Mat = np.int_(np.ceil(Mat))
    return Mat

def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


MEMO_DFT = {}

def DFT(f): # fonctionne mais inutilisable
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

def Noyau_Gaussien_2D(dim,sigma1,sigma2,lim=10):
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

    return Noyau_Gauss


#%%      
image = Image.open("C:/Users/lupus/Desktop/images_python/clock.jpg")
image = image.convert('L')
Mat_image = np.asarray(image)

dim = Mat_image.shape

"""
Mat_image = Mat_image * 0
for i in range(dim[0]):
    for j in range(dim[1]):
        if (i - dim[0]/2)**2 + (j - dim[1]/2)**2 < 1000:
            Mat_image[i,j] = 1
"""


F_image = np.fft.fft2(Mat_image)


"""
Mat = np.zeros(dim)
for i in range(dim[0]):
    for j in range(dim[1]):
        if (i-dim[0]/2)**2 + (j-dim[1]/2)**2 < 100 :
            Mat[i,j] = 1
"""

"""
F_sol = F * F_image
for i in range(dim[0]):
    for j in range(dim[1]):
        if (i-dim[0]/2)**2 + (j-dim[1]/2)**2 < 75000 :
            F_sol[i,j] = 0
        else:
            F_sol[i,j] = F_image[i,j]
"""

sigma1 = 0.01
sigma2 = 1

Noyau_Gauss = Noyau_Gaussien_2D(dim,sigma1,sigma2,lim=10)
F_Gauss = ( np.fft.ifft2(Noyau_Gauss) )

F_sol = ( F_Gauss * F_image )

f_image = np.fft.ifft2(F_image)
f_sol = np.fft.fftshift(np.fft.ifft2(F_sol))


Noyau_Fourier = np.log( np.sqrt( (F_Gauss*F_Gauss.conjugate()).real ) )
F_utile = np.log( np.sqrt( (F_sol*F_sol.conjugate()).real ) )
f_utile = np.sqrt( (f_sol*f_sol.conjugate()).real )
#f_utile = np.sqrt( (f_image*f_image.conjugate()).real ) 

#%%

# image initiale
plt.matshow(Mat_image, cmap = 'binary', interpolation='none')
plt.colorbar()
plt.title("image initiale")
plt.show()

# Noyau
plt.matshow(Noyau_Gauss, cmap = 'plasma', interpolation='none')
plt.colorbar()
plt.title("Noyau")
plt.show()

"""
# Noyau Fourier
plt.matshow(Noyau_Fourier, cmap = 'plasma', interpolation='none')
plt.colorbar()
plt.title("Noyau Fourier")
plt.show()

# spectre
plt.matshow(F_utile, cmap = 'plasma', interpolation='none')
plt.colorbar()
plt.title("spectre")
plt.show()
"""

# Image produite
plt.matshow(f_utile, cmap = 'binary', interpolation='none')
plt.colorbar()
plt.title("Image produite")
plt.show()


#%%


f_s = normMatImage(f_utile)
f_s = Image.fromarray(f_s.astype(np.uint8))
image_s = get_concat_h(image, f_s )
image_s.show()





