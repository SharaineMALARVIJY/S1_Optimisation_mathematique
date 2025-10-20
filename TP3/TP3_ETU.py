#%%
import numpy as np
import matplotlib.pyplot as plt

def genere_population(n_population, bornes) :
    
    np.random.seed(42)
    abcisses = np.random.uniform(bornes[0], bornes[1], size = n_population)
    ordonnees = np.random.uniform(bornes[2], bornes[3], size = n_population)

    return np.array([abcisses, ordonnees])

def calcule_performance(XY) :
    J1 = (XY[0]-8)**2 + (XY[1]-8)**2 + 4
    J2 = (XY[0]-12)**2 + (XY[1]-12)**2 + 4
    return np.array([J1, J2])

bornes_pop =[0, 20, 0, 20]
bornes_obj =[0, 200, 0, 200]

# Constitution de la population initiale
n_population = 100
population = genere_population(n_population, bornes_pop)
# Calcul des performances 
objectifs = calcule_performance(population)


fig = plt.figure()
ax = fig.add_subplot()
plt.plot(population[0], population[1],".k", label = "initial")
plt.legend()
plt.axis(bornes_pop)
ax.set_aspect('equal', adjustable='box')
plt.title(f"Espace de décision")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid()
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(objectifs[0], objectifs[1],".k", label = "initial")
plt.legend()
plt.axis(bornes_obj)
plt.title(f"Espace des objectifs")
plt.xlabel("x [m^2]")
plt.ylabel("y [m^2]")
plt.grid()
plt.show()
# %%
