# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:52:23 2024

@author: UlricJean
"""

#%% IMPORTS/FONCTIONS

# Il faut que les images soit semblables en dimentions.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math as math

# Copie de la décomposition en vp

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

def NormalMat(Mat):
    mi = np.min(Mat)
    Mat = Mat - mi
    ma = np.max(Mat)
    Mat = Mat/ma
    Mat = np.int_(Mat*255)
    return Mat

def creationMatSigma(U,V,sigma):
    
    dimU = U.shape[0]
    dimV = V.shape[0]
    
    if dimU < dimV:
        S = np.diag(sigma,k = (dimU-dimV))
        S = S[dimV-dimU:S.shape[0],:]
    elif dimU > dimV:
        S = np.diag(sigma,k = (dimU-dimV))
        S = S[:,(dimU-dimV):S.shape[0]]
    return S

def SVD(f,k):
    
    U, sigma, V = np.linalg.svd(f)
    S = creationMatSigma(U,V,sigma)
    
    u = U[:,0:k]
    v = V[0:k,:]
    s = S[0:k,0:k]
    return u@s@v

def ConvertVec(M):
    x, y = M.shape
    m1 = np.reshape(M,(x*y,1))
    return m1, np.array((x,y))

def ConvertMat(v,dim):
    w = np.reshape(v,dim)
    return w

"""
def MoyenneVec(v1,v2):
    return (v1+v2)/2
"""

def MoyenneVec(M):
    return (np.sum(M, axis=1)/M.shape[1]).reshape((M.shape[0], 1))

def MatriceVarCovarVP(M1,M2):
    cov = M1.T @ M2
    return cov

def redefineMat(*args): # for matrix
    minX = math.inf
    minY = math.inf
    for Mat in args:
        x, y = Mat.shape
        if x < minX:
            minX = x
        if y < minY:
            minY = y
    l = list()
    for Mat in args:
        Mat = Mat[0:minX,0:minY]
        l.append(Mat)
    return l

"""
def eigenFace(PHI,eigVec):
    eigx, eigy = eigVec.shape
    phix, phiy = PHI.shape
    #u = np.zeros((phix,eigx))
    #for i in range(eigx):
        #eig = np.tile(np.array( eigVec[:,i].reshape(eigx,1) ), (1, eigy))
        #u[:,i] = np.sum(PHI @ eig , axis = 1)
        #u[:,i] = PHI @ eig
    u = PHI @ eigVec.T
    return u

def eigenFace(PHI,eigVec):
    x, y = PHI.shape
    s = np.zeros(x)
    for i in range(y):
        s = s + (eigVec[:,i].reshape(y,1)) @ (PHI[:,i].reshape(1,x))
    return s
"""

def sortEig(eigVal, eigVec):
    sx_eigVal = np.argsort(eigVal)[::-1]
    eigVec = eigVec[:,sx_eigVal]
    return eigVec

def eigvalf(M,eigVec):
    dim = M.shape
    U = M@(eigVec)
    for i in range(dim[1]):
        U[:,i] = U[:,i] / np.linalg.norm(U[:,i])
    return U

def creatOmega(eigf,phi):
    x, y = eigf.shape
    omega = np.zeros(y)
    for i in range(y):
        omega[i] = (eigf[:,i].reshape(x,1)).T @ phi
    return omega

def compLineaire(omega,f):
    s = f[:,0]*0
    for i in range(len(omega)):
        s = s + omega[i] * f[:,i]
    return s

#%% 

n = 70
n+=1
dicoImagesMatrices = {}
for i in range(n):
    var_name = f"f{i}"
    string = f"C:/Users/ulric/OneDrive/Bureau/iloveimg-resized/h ({i+1}).jpg"
    image = Image.open(string)
    image = image.convert('L')
    dicoImagesMatrices[var_name] = np.asarray(image)

#%%

ImVec = {}
for i in range(n):
    key = f"f{i}"
    var = dicoImagesMatrices[key]
    var_name = f"v{i}"
    ImVec[var_name] = ConvertVec(var)[0]
dim = dicoImagesMatrices["f0"].shape

P = ImVec["v0"]
for i in range(1,n):
    key = f"v{i}"
    P = np.concatenate((P,ImVec[key]),axis=1)
psi = MoyenneVec(P)



dicoPhi = {}
for i in range(n):
    var_name = f"phi{i}"
    var = f"v{i}"
    dicoPhi[var_name] = ImVec[var] - psi

psiA = NormalMat(ConvertMat(psi,dim))
imAverage = Image.fromarray(psiA.astype(np.uint8))
imAverage.show()

M = dicoPhi["phi0"]
for i in range(1,n):
    key = f"phi{i}"
    M = np.concatenate((M,dicoPhi[key]),axis=1)



sigma = MatriceVarCovarVP(M,M)
eigVal, eigVec = np.linalg.eig(sigma)
eigVec = sortEig(eigVal, eigVec)
eigf = eigvalf(M,eigVec)


Matf0 = ConvertMat(eigf[:,0],dim)
Matf0 = NormalMat(Matf0)
im0 = Image.fromarray(Matf0.astype(np.uint8))
im_concat = im0
for i in range(1,n):
    Matf = ConvertMat(eigf[:,i],dim)
    Matf = NormalMat(Matf)
    im = Image.fromarray(Matf.astype(np.uint8))
    im_concat = get_concat_h(im_concat,im)
im_concat.show()


#%%

#image = Image.open("C:/Users/ulric/OneDrive/Bureau/(Aucun objet)/testh0.jpg") #Test
image = Image.open("C:/Users/ulric/OneDrive/Bureau/iloveimg-resized/h (45).jpg")
imT = image.convert('L')
fT = np.asarray(imT)

mT, dim = ConvertVec(fT)
phiT = mT - psi

omega = creatOmega(eigf,phiT)

vT = compLineaire(omega,eigf)
vT = NormalMat(vT) + np.matrix.flatten(psi)
MatfT = ConvertMat(vT,dim)

MatfT = NormalMat(MatfT)

imTb = Image.fromarray(MatfT.astype(np.uint8))
ims = get_concat_h(imT, imTb)
ims.show()


#%%

def comparaison(omegaTest,M,eigf):
    dim = M.shape
    om = np.zeros(dim[1])
    for i in range(dim[1]):
        omega = creatOmega(eigf,M[:,i])
        om[i] = np.linalg.norm(omegaTest-omega)
    s = np.argsort(om)
    return s, om
        
s, om = comparaison(omega,M,eigf)

liste = list()
for i in range(n):
    liste.append(f"f{i}")

sol = liste[s[0]]
fsol = dicoImagesMatrices[sol]
fsol = Image.fromarray(fsol.astype(np.uint8))
get_concat_h(imT, fsol).show()

#%%
"""
def comparaisonOmega(omega,omegaApp):
    om = np.zeros(len(omegaApp))
    for i in range(len(omegaApp)):
        t = np.asarray(omegaApp[i])
        om[i] = np.linalg.norm(omegaApp - t)
    s = np.argsort(om)
    return s, om


s, om = comparaisonOmega(omega,omega_A)

liste = [f0,f6,f11,f16]

fbis = liste[s[0]]
fbis = Image.fromarray(fbis.astype(np.uint8))

get_concat_h(imT, fbis).show()
"""








