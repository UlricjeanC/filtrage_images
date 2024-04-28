# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:40:52 2024

@author: lupus
"""

#%% Import/prep

from PIL import Image
import numpy as np

""" 
import os
img = Image.open(filename).convert('RGB')
basename = os.path.splitext(filename)[0] 
"""

image = Image.open("C:/Users/lupus/Desktop/images_python/pillow.png")

#image_save = image
#r,g,b,a = image.split()
#Image.merge("RGBA", (b, r, g, a)).show()

image = image.convert('L')

#Mat_save = np.asarray(image_save)
#Mat_save[:,:,0]
Mat = np.asarray(image)
dim = Mat.shape

image.show()
#image.save('gray-pillow.jpeg', 'jpeg')

#%% Ajout artefacts

image = Image.open("C:/Users/lupus/Desktop/images_python/pillow.png")
image = image.convert('L')
Mat = np.asarray(image)
dim = Mat.shape

def Mat_eps(mu,sigma,dim):
    M = np.random.normal(mu,sigma, size = dim)
    return M


Mat = np.asarray(image) ; mu = 0 ; sigma = 15

s = Mat + Mat_eps(mu,sigma,dim)

def normMatImage(Mat):
    Mat = np.where(Mat > 0, Mat, 0)
    m =  np.max(Mat)
    Mat = (Mat/m) * 255
    Mat = np.int_(np.ceil(Mat))
    return Mat

def normMatImage_troncature(Mat):
    Mat = np.where(Mat > 0, Mat, 0)
    Mat = np.where(Mat < 255, Mat, 255)
    m =  np.max(Mat)
    Mat = (Mat/m) * 255
    Mat = np.int_(np.ceil(Mat))
    return Mat

s = normMatImage(s)

#im = Image.fromarray(np.uint8(cm.gist_earth(Mat)*255))
im = Image.fromarray(s)
Mat_artefact = np.asarray(im)
im.show()



#%% test lissage / filtrage

image = Image.open("C:/Users/lupus/Desktop/images_python/clock.jpg")
image = image.convert('L')

Mat = np.asarray(image) ; dim = Mat.shape ; mu = 0 ; sigma = 10
s = Mat + Mat_eps(mu,sigma,dim)
s = normMatImage(s)
im = Image.fromarray(s)
Mat_artefact = np.asarray(im)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def f_gauss_1D(k,sigma):
    z = 1/np.sqrt(2*np.pi*sigma) * np.exp(-(k**2)/(2*sigma))
    return z

def f_gauss(x,y,sigma):
    z = 1/(2*np.pi*sigma) * np.exp(-(x**2+y**2)/(2*sigma))
    return z

def noyau_gaussien(sigma,dimfp):
    s = np.sqrt(sigma)
    #dim = int(np.ceil(2*np.pi*s))
    xdim, ydim = dimfp
    
    h = np.zeros((xdim,ydim))
    vecx = np.linspace(-2*np.pi*s,2*np.pi*s,xdim)
    vecy = np.linspace(-2*np.pi*s,2*np.pi*s,ydim)
    
    for i in range(xdim):
        for j in range(ydim):
            x = vecx[i]
            y = vecy[j]
            h[i,j] = f_gauss(x,y,sigma) 
    return h

def filtre_noyau(f,h,fonc,k = 3,param = 1):
    if k % 2 == 0:
        k = k + 1
        
    dimf = f.shape
    imx, imy = dimf
    
    g = np.zeros(dimf)
    
    for i in range(imx):
        for j in range(imy):
            fp = f[max(i-k,0):min(i+k,imx) , max(j-k,0):min(j+k,imy)]
            dimfp = fp.shape
            h1 = h(param,dimfp)
            
            S = np.sum ( fp * h1 )
            W = np.sum ( h1 )
            
            g[i,j] = S/W
    return g

def filtre_noyau_bilateral(f,h1,h2,sigma1 = 3,sigma2 = 5,k = 7):
    dimf = f.shape
    imx, imy = dimf
    g = np.zeros(dimf) 
    param = sigma1
    
    for i in range(imx):
        for j in range(imy):
            
            fp = f[max(i-k,0):min(i+k,imx) , max(j-k,0):min(j+k,imy)]
            dimfp = fp.shape
            h_1 = h1(param,dimfp)
            h_2 = h2( fp - f[i,j] , sigma2)
            
            S = np.sum ( fp * h_1 * h_2)
            W = np.sum ( h_1 * h_2)
            
            #S = f[i-M+m,j-N+n]*h1(i-m,j-n,sigma1)*h2(np.abs(f[m,n]-f[i,j]),sigma2)
            #W = h1(i-m,j-n,sigma1)*h2(np.abs(f[m,n]-f[i,j]),sigma2)
            
            g[i,j] = S/W
    return g

def filtre_noyau_median(f,k):
    dimf = f.shape
    imx, imy = dimf
    g = np.zeros(dimf)
    
    ## kernels usually square with odd number of rows/columns
    for i in range(imx):
        for j in range(imy):
            M = f[max(i-k,0):min(i+k,imx) , max(j-k,0):min(j+k,imy)]
            g[i,j] = np.median(M)
    return g

image_s = get_concat_h(image, im)

###############################################################################
param = sigma = 10 ; fonc = f_gauss ; h = noyau_gaussien ; f = Mat_artefact/255 ; 
k = 5 ; tx = 0.3 ; h1 = noyau_gaussien ; h2 = f_gauss_1D ; sigma1 = 10 ; 
sigma2 = 1 ; size = int(np.ceil(2*np.pi*np.sqrt(sigma)))
###############################################################################

# filtre gaussien
g = filtre_noyau(f,h,fonc,k = size,param = sigma)
g = normMatImage(g)
im_gauss = Image.fromarray(g)
image_s = get_concat_h(image_s, im_gauss)
print('ok filtre gaussien')


# filtre median
g1 = filtre_noyau_median(f,k)
g1 = normMatImage(g1)  
im_median = Image.fromarray(g1)
image_s = get_concat_h(image_s, im_median)
print('ok filtre median')


#filtre gaussien corrigé
g2 = filtre_noyau_bilateral(f,h1,h2,sigma1 , 
                            sigma2 ,k = int(np.ceil(2*np.pi*np.sqrt(sigma1))))
g2 = normMatImage(g2)  
im_gauss_corr = Image.fromarray(g2)
image_s = get_concat_h(image_s, im_gauss_corr)
print('ok filtre gaussien corrigé')


image_s.show()

#%% Filtre median appliqué 

image = Image.open("C:/Users/lupus/Desktop/images_python/pillow.png")
r,g,b,a = image.split()
#Image.merge("RGBA", (b, r, g, a)).show()
Mat_r = np.asarray(r)
Mat_g = np.asarray(g)
Mat_b = np.asarray(b)
Mat_a = np.asarray(a)

dim = Mat_a.shape
dimx, dimy = dim

def Image_blanks(dim,p):
    dimx, dimy = dim
    M = np.int_(np.random.rand(dimx,dimy) < p)
    return M

p = 0.8
M = Image_blanks(dim,p)

Mat_r = Mat_r * M + (1-M) * 255
Mat_g = Mat_g * M + (1-M) * 255
Mat_b = Mat_b * M + (1-M) * 255

im_r = Image.fromarray(Mat_r).convert('L')
im_g = Image.fromarray(Mat_g).convert('L')
im_b = Image.fromarray(Mat_b).convert('L')

image_color_trous = Image.merge("RGBA", (im_r, im_g, im_b, a))

k = 2 ; sigma = 0.5 ; f = f_gauss ; h = noyau_gaussien(sigma,f)

# on applique 2 fois le filtre median, une fois un filtre gaussien

p_r = filtre_noyau_median(Mat_r,k)
p_r = filtre_noyau_median(p_r,k)
p_r = filtre_noyau(p_r,h)
p_r = normMatImage(p_r)  
im_r_med = Image.fromarray(p_r).convert('L')

p_g = filtre_noyau_median(Mat_g,k)
p_g = filtre_noyau_median(p_g,k)
p_g = filtre_noyau(p_g,h)
p_g = normMatImage(p_g)  
im_g_med = Image.fromarray(p_g).convert('L')

p_b = filtre_noyau_median(Mat_b,k)
p_b = filtre_noyau_median(p_b,k)
p_b = filtre_noyau(p_b,h)
p_b = normMatImage(p_b)  
im_b_med = Image.fromarray(p_b).convert('L')


im_median_recomp = Image.merge("RGBA",
                               (im_r_med, im_g_med, im_b_med, a))


image_comp = get_concat_h(image_color_trous, im_median_recomp)
image_comp = get_concat_h(image, image_comp)
image_comp.show()




