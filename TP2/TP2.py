import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def P1(v):
    return -((v-2)**2)/2+1

def P2(u):
    return (u-3)**2+2

xmin, xmax, nx = -2.5, 5, 101
ymin, ymax, ny = -2.5, 5, 101

x1d = np.linspace(xmin,xmax,nx)
y1d = np.linspace(ymin,ymax,ny)

nIso = 101
plt.plot(P1(x1d), x1d, label='P1')
plt.plot(x1d, P2(x1d), label='P2')
plt.xlabel('Valeurs de u')
plt.ylabel('Valeurs de v')
plt.legend()
plt.grid()
plt.axis('square')
plt.show()

# x = [u1, u2, v1, v2, l1, l2]
x0 = np.random.rand(6)


def fun(X):
    return [
        -2 * (X[1] - X[0]) + X[4],
        2 * (X[1] - X[0]) - X[5] * 2 * (X[1] - 3),
        -2 * (X[3] - X[2]) + X[4] * (X[2] - 2),
        2 * (X[3] - X[2]) + X[5],
        X[0] + ((X[2] - 2) ** 2) / 2 - 1,
        X[3] - (X[1] - 3) ** 2 - 2,
    ]


s = optimize.root(fun, x0)
print(s.success, s.x)

# %% Exercice 2

Xn = [-1, 2]


def J(x, y):
    return (x - 1) ** 2 + 2 * (x**2 - y) ** 2


def dJ(x, y):
    dJ_x = 2 * (x - 1) + 8 * x * (x**2 - y)
    dJ_y = -4 * (x**2 - y)
    return [dJ_x, dJ_y]


def C(x, y, xn, yn):
    dJ_xn, dJ_yn = dJ(xn, yn)
    return dJ_yn * (x - xn) - dJ_xn * (y - yn)


def dC(xn, yn):
    dJ_xn, dJ_yn = dJ(xn, yn)
    dC_x = dJ_yn
    dC_y = -dJ_xn
    return [dC_x, dC_y]


def system(X, param):
    x = X[0]
    y = X[1]
    lam = X[2]
    [xn, yn] = param
    dJ_x, dJ_y = dJ(x, y)
    dC_xn, dC_yn = dC(xn, yn)
    return [dJ_x + lam * dC_xn, dJ_y + lam * dC_yn, C(x, y, xn, yn)]


X_ini = [0, 0, 0]

# initial
print(f"X0 = {(Xn[0],Xn[1])}")
print(f"value J(X0) = {J(Xn[0],Xn[1])}")
plt.plot(Xn[0], Xn[1], "o")
# first
sol = optimize.root(system, X_ini, args=Xn)
print(f"success : {sol.success} X = ({sol.x[0]:.3},{sol.x[1]:.3}),lam= {sol.x[2]:.3}")
print(f"value J(X1) = {J(sol.x[0],sol.x[1])}")
plt.plot(sol.x[0], sol.x[1], "o")
# second
sol = optimize.root(system, X_ini, args=[sol.x[0], sol.x[1]])
print(f"success : {sol.success} X = ({sol.x[0]:.3},{sol.x[1]:.3}),lam= {sol.x[2]:.3}")
print(f"value J(X2) = {J(sol.x[0],sol.x[1])}")
plt.plot(sol.x[0], sol.x[1], "o")
# third
sol = optimize.root(system, X_ini, args=[sol.x[0], sol.x[1]])
print(f"success : {sol.success} X = ({sol.x[0]:.3},{sol.x[1]:.3}),lam= {sol.x[2]:.3}")
print(f"value J(X3) = {J(sol.x[0],sol.x[1])}")
plt.plot(sol.x[0], sol.x[1], "o")


# Définition du domaine de tracé
xmin, xmax, nx = -1.5, 2.5, 550
ymin, ymax, ny = -1.5, 2.5, 550

# Discrétisation du domaine de tracé
x1d = np.linspace(xmin, xmax, nx)
y1d = np.linspace(ymin, ymax, ny)
x2d, y2d = np.meshgrid(x1d, y1d)

# Tracé des isovaleurs de fl
nIso = 491
plt.contour(x2d, y2d, J(x2d, y2d), nIso)
plt.title("Isovaleurs")
plt.xlabel("Valeurs de x")
plt.ylabel("Valeurs de y")
plt.grid()
plt.axis("square")
plt.show()


# %%
