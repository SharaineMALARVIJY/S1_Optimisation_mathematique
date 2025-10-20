#Sharaine MALARVIJY
#%%

from fonctions_etu import genere_population,    \
                                calcule_performance,  \
                                selection_population, \
                                croisement

import numpy as np
import matplotlib.pyplot as plt


#%% PREMIERE GENERATION

bornes_pop =[0, 20, 0, 20]
bornes_obj =[0, 200, 0, 200]

# Constitution de la population initiale
n_population = 100
population = genere_population(n_population, bornes_pop)
# Calcul des performances 
objectifs = calcule_performance(population)


#%% 1 ITERATION DE L'ALGORITHME GENETIQUE

nb_iteration = 4

for i in range (1, nb_iteration+1):
        
        # 1. Sélection des 50% meilleurs individus (classement selon rang de Pareto)
        selection_pop, selection_obj = selection_population(population, objectifs)

        # 2. Création des enfants pour compléter la population
        children = []
        taille_population_selectionnee = selection_pop.shape[1]
        # Création d'autant d'enfants que de parents
        for _ in range(taille_population_selectionnee // 2) :
        
                # Choisir aléatoirement 2 parents
                id_p1 = np.random.randint(taille_population_selectionnee) 
                parent1 = selection_pop[:,id_p1]
                id_p2 = np.random.randint(taille_population_selectionnee) 
                parent2 = selection_pop[:,id_p2]
        
                # Le croisement de deux parents donne deux enfants
                child1, child2 = croisement(parent1, parent2)
                # Les enfants sont ajoutés à la lste des descendants
                children.append(child1)
                children.append(child2)
                
        children = np.array(children).transpose()

        # 3. Calcul des performances des enfants
        children_obj = calcule_performance(children)


        # 4. Représentation dans l'espace de décision
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(population[0], population[1],".k", label = "initial")
        plt.plot(selection_pop[0], selection_pop[1],".b",label = "sélection")
        plt.plot(children[0], children[1],".r", label = "enfants")
        plt.legend()
        plt.axis(bornes_pop)
        ax.set_aspect('equal', adjustable='box')
        plt.title(f"Espace de décision, itération {i}")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid()
        plt.show()

        # 5. Représentation dans l'espace des objectifs
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(objectifs[0], objectifs[1],".k", label = "initial")
        plt.plot(selection_obj[0], selection_obj[1],".b", label = "sélection")
        plt.plot(children_obj[0], children_obj[1],".r", label = "enfants")
        plt.legend()
        plt.axis(bornes_obj)
        ax.set_aspect('equal', adjustable='box')
        plt.title(f"Espace des objectifs, itération {i}")
        plt.xlabel("x [m^2]")
        plt.ylabel("y [m^2]")
        plt.grid()
        plt.show()

        # 6. Fusion de la population sélectionnée et de sa descendance   
        population = np.concatenate([selection_pop, children], axis=1)
        objectifs = np.concatenate([selection_obj, children_obj], axis=1)



# %%
