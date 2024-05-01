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
image = Image.open("C:/Users/lupus/Desktop/images_python/pillow.png")

image = image.convert('L')

Mat = np.asarray(image)

Mat = np.zeros((1000,1000))
for i in range(1000):
    for j in range(1000):
        if (i-500)**2 + (j-500)**2 < 10000 :
            Mat[i,j] = 1

#F_real, F_imag = DFT(Mat)

F = np.fft.fft2(Mat)
F_inv = np.fft.ifft2(F)

#%%

F_utile = np.log( np.sqrt( (F*F.conjugate()).real ) )

F_utile2 = np.sqrt( (F_inv*F_inv.conjugate()).real ) 

plt.matshow(Mat, cmap = 'binary', interpolation='none')
plt.colorbar()
plt.show()

plt.matshow(F_utile, cmap = 'plasma', interpolation='none')
plt.colorbar()
plt.show()

plt.matshow(F_utile2, cmap = 'binary', interpolation='none')
plt.colorbar()
plt.show()










