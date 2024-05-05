# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:40:52 2024

@author: UlricJeanC
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def Mat_eps(mu, sigma, dim):
    return np.random.normal(mu, sigma, size=(dim))

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

def gradient_dx(f,eps):
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
    return f

def gradient_dy(f,eps):
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
    
    return f

def Matrice_sobel(k):
    
    if k%2 == 0:
        k += 1
    
    centre = k//2 + 1
    vec1 = np.zeros(k)
    vec2 = np.zeros(k)
    
    c = 1
    s = 1
    
    for i in range(k):
        if i < (centre-1):
            vec1[i] = c
            vec2[i] = s
            c += 1
            s += 1
        elif i > (centre-1):
            vec1[i] = c
            vec2[i] = -s
            c -= 1
            s -= 1
        else:
            vec1[i] = c
            c -= 1
            s -= 1
          
    vec2 = - vec2
    Mat = (vec1.reshape(k, 1) @ vec2.reshape(k, 1).T )
    
    return Mat
    

def gradient_sobel_dx(f,eps,k=3):
    dim = f.shape
    dimx, dimy = dim
    
    mat = Matrice_sobel(k)
    mat = mat/(2*eps)
    
    matx, maty = mat.shape
    
    mat = np.block([
        [ mat , np.zeros(( (matx) , (dimy-maty) )) ],
        [ np.zeros(( (dimx-matx) , (maty) )) , np.zeros(( (dimx-matx) , (dimy-maty) )) ],
        ])
    
    
    F = np.fft.fft2(f)
    Mat = np.fft.fft2(mat)
    
    F = F*Mat
    
    f = (np.fft.ifft2(F))
    return f

def gradient_sobel_dy(f,eps,k=3):
    dim = f.shape
    dimx, dimy = dim
    
    mat = Matrice_sobel(k).T
    mat = mat/(2*eps)
    
    matx, maty = mat.shape
    
    mat = np.block([
        [ mat , np.zeros(( (matx) , (dimy-maty) )) ],
        [ np.zeros(( (dimx-matx) , (maty) )) , np.zeros(( (dimx-matx) , (dimy-maty) )) ],
        ])
    
    F = np.fft.fft2(f)
    Mat = np.fft.fft2(mat)
    
    F = F*Mat
    
    f = (np.fft.ifft2(F))
    return f

def Laplacien(f,eps):
    dim = f.shape
    mat = np.zeros(dim)
    
    mat[0,0] = mat[2,0] = mat[0,2] = mat[2,2] = 1
    mat[1,0] = mat[2,1] = mat[1,2] = mat[0,1] = 4 
    mat[1,1] = -20

    mat = mat / (6*eps**2)
    
    F = np.fft.fft2(f)
    Mat = np.fft.fft2(mat)
    
    F = F*Mat
    
    f = (np.fft.ifft2(F))
    
    return f


if __name__ == "__main__":
    image = Image.open("C:/Users/lupus/Desktop/images_python/clock.jpg")
    #image = Image.open("C:/Users/lupus/Desktop/images_python/cell.png")
    r,g,b = image.split()
    image = image.convert("L")

    Mat = np.asarray(r)
    dim = Mat.shape
    mu = 0
    sigma = 15
    p = 0.9
    M = Image_blanks(dim,p)
    
    s = (Mat + Mat_eps(mu, sigma, dim)) * M + (1-M) * 255
    
    gmat = 255 - np.asarray(g)
    s = gmat
    
    s = normMatImage(s)
    
    im = Image.fromarray(s.astype(np.uint8))
    Mat_artefact = np.asarray(im)
    image_s = get_concat_h(image, im)

    ###############################################################################
    param = sigma = 5
    fonc = f_gauss
    h = noyau_gaussien
    f = Mat_artefact / 255
    k = 2
    tx = 0.3
    h1 = noyau_gaussien
    h2 = f_gauss_1D
    sigma1 = 5
    sigma2 = 0.001
    size = int(np.ceil(2 * np.pi * np.sqrt(sigma)))
    ###############################################################################

    """
    # filtre gaussien
    g = filtre_noyau(f, h, fonc, k=size, param=sigma)
    g = normMatImage(g)
    im_gauss = Image.fromarray(g.astype(np.uint8))
    image_s = get_concat_h(image_s, im_gauss)
    print("ok filtre gaussien")

    # filtre median
    g1 = filtre_noyau_median(f, k)
    g1 = normMatImage(g1)
    im_median = Image.fromarray(g1.astype(np.uint8))
    image_s = get_concat_h(image_s, im_median)
    print("ok filtre median")

    # filtre gaussien corrigé
    g2 = filtre_noyau_bilateral(
        f, h1, h2, sigma1, sigma2, k=int(np.ceil(2 * np.pi * np.sqrt(sigma1)))
    )
    g2 = normMatImage(g2)
    im_gauss_corr = Image.fromarray(g2.astype(np.uint8))
    image_s = get_concat_h(image_s, im_gauss_corr)
    print("ok filtre gaussien corrigé")
    
    #combinaison de filtres
    
    g3 = filtre_noyau_median(f, k)
    g3 = filtre_noyau_bilateral(
        g3, h1, h2, sigma1, sigma2 = 0.005, k=int(np.ceil(2 * np.pi * 
                                                          np.sqrt(sigma1)))
    )
    g3 = normMatImage(g3)
    im_comb = Image.fromarray(g3.astype(np.uint8))
    image_s = get_concat_h(image_s, im_comb)
    print("ok filtre combinés")
    
    #fine tuned post filtre
    
    g4 = np.where(g2 > 75, g3, g2)
    #g4 = filtre_noyau_median(g4, k=1)
    g4 = filtre_noyau_bilateral(
        g4, h1, h2, sigma1=6, sigma2=0.1, k=int(np.ceil(2 * np.pi * 
                                                        np.sqrt(sigma1)))
    )
    g4 = filtre_noyau_bilateral(
        g4, h1, h2, sigma1=10, sigma2=1000, k=int(np.ceil(2 * np.pi * 
                                                          np.sqrt(sigma1)))
    )
    g4 = normMatImage(g4)
    im_fineTuned = Image.fromarray(g4.astype(np.uint8))
    image_s = get_concat_h(image_s, im_fineTuned)
    print("ok filtre fined tuned")
    
    image_s.show()
    """
    
    f = s ; eps = 2 ; k = 3
    
    """
    f_dx = gradient_dx(f,eps)
    im_edge_x = Image.fromarray(f_dx.astype(np.uint8))
    image_s = get_concat_h(image_s, im_edge_x)
    
    f_dy = gradient_dy(f,eps)
    im_edge_y = Image.fromarray(f_dy.astype(np.uint8))
    image_s = get_concat_h(image_s, im_edge_y)
    
    mag_grad = (f_dx + f_dy)
    im_edge = Image.fromarray(mag_grad.astype(np.uint8))
    image_s = get_concat_h(image_s, im_edge)
    
    """
    
    f_sobel_dx = gradient_sobel_dx(f,eps,k)
    f_sobel_dx = np.sqrt( (f_sobel_dx * f_sobel_dx.conjugate()).real )
    im_edge_sobel_x = Image.fromarray(f_sobel_dx.astype(np.uint8))
    image_s = get_concat_h(image_s, im_edge_sobel_x)
    
    f_sobel_dy = gradient_sobel_dy(f,eps,k)
    f_sobel_dy = np.sqrt( (f_sobel_dy * f_sobel_dy.conjugate()).real )
    im_edge_sobel_y = Image.fromarray(f_sobel_dy.astype(np.uint8))
    image_s = get_concat_h(image_s, im_edge_sobel_y)
    
    
    mag_grad = f_sobel_dy + f_sobel_dx
    im_edge = Image.fromarray(mag_grad.astype(np.uint8))
    image_s = get_concat_h(image_s, im_edge)
    
    mag_grad = np.where(mag_grad > 50 , 255 , 0)
    im_edge = Image.fromarray(mag_grad.astype(np.uint8))
    image_s = get_concat_h(image_s, im_edge)
    
    ###########
    
    eps = 2 ; f = s ; dim = f.shape ; sigma1 = sigma2 = 0.005
    Noyau_gauss = Noyau_Gaussien_2D(dim,sigma1,sigma2,lim=10)
    
    """
    plt.matshow(Noyau_gauss, cmap = 'plasma', interpolation='none')
    plt.colorbar()
    plt.title("Noyau")
    plt.show()
    """
    
    '''
    F_lisse = np.fft.ifft2(Noyau_gauss) * np.fft.ifft2(f)
    f_lisse = np.fft.fftshift(np.fft.ifft2(F_lisse)).T
    f_lisse = np.sqrt( (f_lisse * f_lisse.conjugate()).real )
    '''
    
    """
    plt.matshow(f_lisse, cmap = 'binary', interpolation='none')
    plt.colorbar()
    plt.title("image")
    plt.show()
    """

    f_laplace = Laplacien(f,eps)
    fsss = f_laplace.real
    f_laplace = fsss
    f_laplace = np.sqrt( (f_laplace * f_laplace.conjugate() ).real )
    f_laplace = normMatImage(f_laplace)
    #f_laplace = np.where(f_laplace > 10, 255, 0)
    im_laplace = Image.fromarray(f_laplace.astype(np.uint8))
    image_s = get_concat_h(image_s, im_laplace)
    
    
    image_s.show()
    
    
    
    
    
    
    
    
    
