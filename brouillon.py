# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:07:34 2024

@author: lupus
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cbook, cm
from matplotlib.colors import LightSource

from PIL import Image
import numpy as np


def Mat_eps(mu, sigma, dim):
    return np.random.normal(mu, sigma, size=(dim,dim))

def Image_blanks(dim,p):
    dimx, dimy = dim
    M = np.int_(np.random.rand(dimx,dimy) < p)
    return M


def normMatImage(Mat):
    Mat = np.where(Mat > 0, Mat, 0)
    m = np.max(Mat)
    Mat = (Mat / m) * 255
    Mat = np.int_(np.ceil(Mat))
    return Mat


def normMatImage_troncature(Mat):
    Mat = np.where(Mat > 0, Mat, 0)
    Mat = np.where(Mat < 255, Mat, 255)
    m = np.max(Mat)
    Mat = (Mat / m) * 255
    Mat = np.int_(np.ceil(Mat))
    return Mat


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new("RGB", (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def f_gauss_1D(k, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma) * np.exp(-(k * k) / (2 * sigma))


GAUSS_MEMO = {}


def f_gauss(x, y, sigma):
    x, y = min(x, y), max(x, y)
    res = GAUSS_MEMO.get((x, y, sigma))
    if res is None:
        partial = GAUSS_MEMO.get((x, y))
        if partial is None:
            partial = np.exp(-(x * x + y * y) / 2)
            GAUSS_MEMO[(x, y)] = partial
        res = (partial ** (1 / sigma)) * (1 / (2 * np.pi * sigma))
        GAUSS_MEMO[(x, y, sigma)] = res

    return res


MEMO_NOYAU = {}


def noyau_gaussien(sigma, dimfp):
    h = MEMO_NOYAU.get((sigma, dimfp))
    if h is None:
        s = np.sqrt(sigma)
        xdim, ydim = dimfp

        h = np.zeros((xdim, ydim))
        vecx = np.linspace(-2 * np.pi * s, 2 * np.pi * s, xdim)
        vecy = np.linspace(-2 * np.pi * s, 2 * np.pi * s, ydim)

        for i in range(xdim):
            for j in range(ydim):
                x = vecx[i]
                y = vecy[j]
                h[i, j] = f_gauss(x, y, sigma)

        MEMO_NOYAU[(sigma, dimfp)] = h

    return h


def filtre_noyau(f, h, fonc, k=3, param=1):
    if k % 2 == 0:
        k = k + 1

    dimf = f.shape
    imx, imy = dimf

    g = np.zeros(dimf)

    for i in range(imx):
        for j in range(imy):
            fp = f[max(i - k, 0) : min(i + k, imx), max(j - k, 0) : 
                   min(j + k, imy)]
            dimfp = fp.shape
            h1 = h(param, dimfp)

            S = np.sum(fp * h1)
            W = np.sum(h1)

            g[i, j] = S / W

    return g


def filtre_noyau_bilateral(f, h1, h2, sigma1=3, sigma2=5, k=7):
    dimf = f.shape
    imx, imy = dimf
    g = np.zeros(dimf)
    param = sigma1

    for i in range(imx):
        for j in range(imy):

            fp = f[max(i - k, 0) : min(i + k, imx), max(j - k, 0) : 
                   min(j + k, imy)]
            dimfp = fp.shape
            h_1 = h1(param, dimfp)
            h_2 = h2(fp - f[i, j], sigma2)

            S = np.sum(fp * h_1 * h_2)
            W = np.sum(h_1 * h_2)

            g[i, j] = S / W

    return g


def filtre_noyau_median(f, k):
    dimf = f.shape
    imx, imy = dimf
    g = np.zeros(dimf)

    # kernels usually square with odd number of rows/columns
    for i in range(imx):
        for j in range(imy):
            M = f[max(i - k, 0) : min(i + k, imx), max(j - k, 0) : 
                  min(j + k, imy)]
            g[i, j] = np.median(M)

    return g

# Load and format data
dim = 100
lim = 3
z = np.zeros((dim,dim))
x = np.linspace(-lim,lim,dim)
y = np.linspace(-lim,lim,dim)

sigma = 1
for i in range(dim):
    for j in range(dim):
        z[i,j] =  ( np.exp(-(x[i] * x[i] + y[j] * y[j]) / 2) *
                   (1 / sigma) * (1 / (2 * np.pi * sigma)))

x, y = np.meshgrid(x, y)

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.set_facecolor('black')
plt.axis('off')

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)


plt.show()

#%% 

dim = 70
lim = 3
z = np.zeros((dim,dim))
x = np.linspace(0,20,dim)
y = np.linspace(0,20,dim)

for i in range(dim):
    for j in range(dim):
        if(i<=j):
            z[i,j] = 1 

mu = 0 ; sigma = 0.25

x, y = np.meshgrid(x, y)

z = z + Mat_eps(mu, sigma, (dim,dim))

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.set_facecolor('black')
plt.axis('off')

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

#%% 

f = z ; h = noyau_gaussien ; fonc = f_gauss
sigma = 30
size = int(np.ceil(2 * np.pi * np.sqrt(sigma)))

g = filtre_noyau(f, h, fonc, k=size, param=sigma)
g = filtre_noyau(g, h, fonc, k=size, param=sigma)
g = filtre_noyau(g, h, fonc, k=size, param=sigma)

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.set_facecolor('black')
plt.axis('off')

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(g, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

#%%

f = z ; h1 = noyau_gaussien ; h2 = f_gauss_1D ; fonc = f_gauss
sigma1 = 30
sigma2 = 0.3
size = int(np.ceil(2 * np.pi * np.sqrt(sigma)))

g = filtre_noyau_bilateral(
    f, h1, h2, sigma1, sigma2, k=int(np.ceil(2 * np.pi * np.sqrt(sigma1)))
)
g = filtre_noyau_bilateral(
    g, h1, h2, sigma1, sigma2, k=int(np.ceil(2 * np.pi * np.sqrt(sigma1)))
)

g = filtre_noyau_bilateral(
    g, h1, h2, sigma1, sigma2, k=int(np.ceil(2 * np.pi * np.sqrt(sigma1)))
)


# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.set_facecolor('black')
plt.axis('off')

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(g, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
