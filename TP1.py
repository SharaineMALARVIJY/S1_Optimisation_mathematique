# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 08:29:11 2025

@author: MALARVIJY Sharaine 21206543
"""

import numpy as np
import matplotlib.pyplot as plt

def J(X):
    x1, x2 = X
    return (x1-1)**2+2*(x1**2-x2)**2

def gradJ(X):
    x1, x2 = X
    dj_dx1 = 2*x1-2+8*x1**3-8*x1*x2
    dj_dx2 = 4*(-x1**2+x2)
    return [dj_dx1, dj_dx2]

def descente_gradient_fixe(X0, alpha, precision, n_max):
    alpha_adaptatif = alpha
    X = [X0]
    n = 0
    converge = False
    while (not(converge) and n < n_max) :
        x= X[n][0]-alpha_adaptatif*gradJ(X[n])[0]
        y= X[n][1]-alpha_adaptatif*gradJ(X[n])[1]     
        X.append([x,y])
        dX = np.sqrt((X[n+1][0]-X[n][0])**2+(X[n+1][1]-X[n][1])**2)
        
        if (J([x, y]) < J(X[n])) and alpha_adaptatif >= 0.05:
            print(alpha_adaptatif)
            alpha_adaptatif = alpha_adaptatif*0.9
        
        n=n+1
        converge = dX<=precision
    return X, converge , n

X0 = [-1, 2]
alpha = 0.1
precision = 10**-10
n_max=10000

X, converge, n = descente_gradient_fixe(X0, alpha, precision, n_max)

print("Point initial :", X0)
print("Point final :", X[-1])
print("Convergence :", converge)
print("Nombre d'iteration :", n)

x, y = [], []
for i in range(len(X)):
    x.append(X[i][0])
    y.append(X[i][1])


# Définition du domaine de tracé
xmin, xmax, nx = -2.5, 2.5, 101
ymin, ymax, ny = -2.5, 2.5, 101

# Discrétisation du domaine de tracé
x1d = np.linspace(xmin,xmax,nx)
y1d = np.linspace(ymin,ymax,ny)
x2d, y2d = np.meshgrid(x1d,y1d)

nIso = 201
plt.contour(x2d,y2d,J([x2d,y2d]),nIso)
plt.plot(x, y, 'x-')
plt.title('Isovaleurs')
plt.xlabel('Valeurs de x')
plt.ylabel('Valeurs de y')
plt.grid()
plt.axis('square')
plt.show()
