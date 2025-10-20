import numpy as np 
import matplotlib.pyplot as plt
from scipy import optimize


# def P1(v):
#     return -((v-2)**2)/2+1

# def P2(u):
#     return (u-3)**2+2

# xmin, xmax, nx = -2.5, 5, 101
# ymin, ymax, ny = -2.5, 5, 101

# x1d = np.linspace(xmin,xmax,nx)
# y1d = np.linspace(ymin,ymax,ny)

# nIso = 101
# plt.plot(P1(x1d), x1d, label='P1')
# plt.plot(x1d, P2(x1d), label='P2')
# plt.xlabel('Valeurs de u')
# plt.ylabel('Valeurs de v')
# plt.legend()
# plt.grid()
# plt.axis('square')
# plt.show()

# x = [u1, u2, v1, v2, l1, l2]
x0 = np.random.rand(6)

def fun(X):
    return [-2*(X[1]-X[0])+X[4],
            2*(X[1]-X[0])-X[5]*2*(X[1]-3),
            -2*(X[3]-X[2])+X[4]*(X[2]-2),
            2*(X[3]-X[2])+X[5],
            X[0]+((X[2]-2)**2)/2-1,
            X[3]-(X[1]-3)**2-2]

s = optimize.root(fun, x0)
print(s.success, s.x)

# %% Exercice 2


def J(X):
    x1, x2 = X
    return (x1-1)**2+2*(x1**2-x2)**2

def gradJ(X):
    x1, x2 = X
    dj_dx1 = 2*x1-2+8*x1**3-8*x1*x2
    dj_dx2 = 4*(-x1**2+x2)
    return [dj_dx1, dj_dx2]

def C(X, Xp):
    x1, x2 = X
    xp1, xp2 = Xp
    return gradJ(X)(xp1-x1)-gradJ(X)(xp2-x2)

def fun(X):
    return [
            gradJ(X)[1], #x+1
            gradJ(X)[0]
            ]

def descente_gradient_fixe(X0, alpha, precision, n_max):
    X = [X0]
    n = 0
    converge = False
    while (not(converge) and n < n_max) :
        x= X[n][0]-alpha*gradJ(X[n])[0]
        y= X[n][1]-alpha*gradJ(X[n])[1]     
        X.append([x,y])
        dX = np.sqrt((X[n+1][0]-X[n][0])**2+(X[n+1][1]-X[n][1])**2)


        
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



xmin, xmax, nx = -2.5, 2.5, 101
ymin, ymax, ny = -2.5, 2.5, 101


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



# %%
