# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:31:16 2024

@author: UlricJeanC
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def gradient_dx_cornerDetection(f,eps):
    dim = f.shape
    mat = np.zeros(dim)
    
    centre_x = dim[0]//2
    centre_y = dim[1]//2
    
    mat[centre_x,centre_y] = -1
    mat[centre_x,(centre_y+1)] = -1
    mat[(centre_x+1),centre_y] = 1
    mat[(centre_x+1),(centre_y+1)] = 1
    mat = mat/(2*eps)
    
    F = np.fft.fft2(f)
    Mat = np.fft.fft2(mat)
    
    F = F*Mat
    
    f = np.fft.fftshift(np.fft.ifft2(F))
    f = f.real
    
    return f

def gradient_dy_cornerDetection(f,eps):
    dim = f.shape
    mat = np.zeros(dim)
    
    centre_x = dim[0]//2
    centre_y = dim[1]//2
    
    mat[centre_x,centre_y] = 1
    mat[centre_x,(centre_y+1)] = -1
    mat[(centre_x+1),centre_y] = 1
    mat[(centre_x+1),(centre_y+1)] = -1
    
    mat = mat/(2*eps)
    
    F = np.fft.fft2(f)
    Mat = np.fft.fft2(mat)
    
    F = F*Mat
    
    f = np.fft.fftshift(np.fft.ifft2(F))
    f = f.real
    
    return f


image = Image.open("C:/Users/lupus/Desktop/images_python/clock.jpg")
image = image.convert('L')

f = np.asarray(image) ; dim = f.shape ; eps = 1

# image initiale
plt.matshow(f, cmap = 'binary', interpolation='none')
plt.colorbar()
plt.title("image initiale")
plt.show()


# Gradient
grad_x = gradient_dx_cornerDetection(f,eps)
grad_x = np.reshape(grad_x, ((dim[0]*dim[1]),1) )
grad_y = gradient_dy_cornerDetection(f,eps)
grad_y = np.reshape(grad_y, ((dim[0]*dim[1]),1) )


plt.figure()
plt.scatter(grad_x,grad_y,s=0.2)
plt.title("gradient")
plt.show()





