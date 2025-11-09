# Sharaine MALARVIJY
# %%
import numpy as np
import struct


# %%
def genere_population(n_population, bornes):

    np.random.seed(42)
    abcisses = np.random.uniform(bornes[0], bornes[1], size=n_population)
    ordonnees = np.random.uniform(bornes[2], bornes[3], size=n_population)

    return np.array([abcisses, ordonnees])


# %%
def calcule_performance(XY):
    J1 = (XY[0] - 8) ** 2 + (XY[1] - 8) ** 2 + 4
    J2 = (XY[0] - 12) ** 2 + (XY[1] - 12) ** 2 + 4
    return np.array([J1, J2])


# %%
def analyse_dominance(population, objectifs):
    """
    Détermination des solutions non dominées
    input : objectifs : array (n_points, 2) contenant les objectifs de chaque individu
    output :
        objectifs_tries : array(n_points,2) objectifs réorganisés (renumérotés)
        dominant : array(n_points, 2) contenant le statut de chaque point

    """
    # Tri des points suivant l'objectif 1
    idx_tries = np.argsort(objectifs[0])
    objectif_0_tries = objectifs[0][idx_tries]
    objectif_1_tries = objectifs[1][idx_tries]

    # Initialisation du statut des points
    est_domine = np.array([False] * len(objectif_0_tries))
    # Détermine le statut de chaque point
    for i in range(len(objectif_0_tries)):
        est_domine[i] = np.any(objectif_1_tries[:i] < objectif_1_tries[i])
    objectifs_tries = np.array([objectif_0_tries, objectif_1_tries])
    dominant = np.logical_not(est_domine)

    # Tri de la population
    population_0_triee = population[0][idx_tries]
    population_1_triee = population[1][idx_tries]
    population_triee = np.array([population_0_triee, population_1_triee])

    return population_triee, objectifs_tries, dominant


# %%
def selection_population(population, objectifs):

    taille_selection = len(population[0]) // 2
    selection_pop = np.array([[], []])
    selection_obj = np.array([[], []])
    pop_trv = np.copy(population)
    obj_trv = np.copy(objectifs)

    termine = False
    while not termine:

        # Détermination des points non dominés
        population_triee, objectifs_tries, dominant = analyse_dominance(
            pop_trv, obj_trv
        )
        id_dominant = np.where(dominant)
        id_drop = np.where(np.logical_not(dominant))

        if len(selection_pop[0]) + len(id_dominant[0]) <= taille_selection:

            # Sélection de tous les points du front
            x_sel = np.concatenate([selection_pop[0], population_triee[0][id_dominant]])
            y_sel = np.concatenate([selection_pop[1], population_triee[1][id_dominant]])
            selection_pop = np.array([x_sel, y_sel])
            j1_sel = np.concatenate([selection_obj[0], objectifs_tries[0][id_dominant]])
            j2_sel = np.concatenate([selection_obj[1], objectifs_tries[1][id_dominant]])
            selection_obj = np.array([j1_sel, j2_sel])
            # Conservation des points hors front
            x_trv = population_triee[0][id_drop]
            y_trv = population_triee[1][id_drop]
            pop_trv = np.array([x_trv, y_trv])
            j1_trv = objectifs_tries[0][id_drop]
            j2_trv = objectifs_tries[1][id_drop]
            obj_trv = np.array([j1_trv, j2_trv])

        else:

            # Sélection aléatoire de points du front pour compléter la sélection
            idx_sel = np.random.choice(
                id_dominant[0], taille_selection - len(x_sel), replace=False
            )
            x_sel = np.concatenate([selection_pop[0], population_triee[0][idx_sel]])
            y_sel = np.concatenate([selection_pop[1], population_triee[1][idx_sel]])
            selection_pop = np.array([x_sel, y_sel])
            j1_sel = np.concatenate([selection_obj[0], objectifs_tries[0][idx_sel]])
            j2_sel = np.concatenate([selection_obj[1], objectifs_tries[1][idx_sel]])
            selection_obj = np.array([j1_sel, j2_sel])

            termine = True

    return selection_pop, selection_obj


# %%
def float_to_binary(num):
    # Convertir le flottant en une représentation binaire IEEE 754
    binary = "".join(f"{c:08b}" for c in struct.pack("!f", num))
    return binary


# %%
def binary_to_float(binary_string):

    if len(binary_string) != 32:
        print("Il faut une chaine de longueur 32")
        return None

    list_int = []
    for i in range(4):
        # print(binary_string[i*8 : (i+1)*8])
        list_int.append(int(binary_string[i * 8 : (i + 1) * 8], 2))

    return struct.unpack(">f", bytes(list_int))[0]


# %% Fonction de croisement


def croisement(parent1, parent2):

    # Construction des chromosomes des parents
    chromo_p1 = float_to_binary(parent1[0]) + float_to_binary(parent1[1])
    chromo_p2 = float_to_binary(parent2[0]) + float_to_binary(parent2[1])

    seuil = np.random.randint(1, 64)
    chromo_ch1 = chromo_p1[:seuil] + chromo_p1[seuil:]
    chromo_ch2 = chromo_p2[:seuil] + chromo_p2[seuil:]

    # Conversion en float du chromosome des enfants
    x_ch1 = binary_to_float(chromo_ch1[0:32])
    y_ch1 = binary_to_float(chromo_ch1[32:64])
    child1 = [x_ch1, y_ch1]

    x_ch2 = binary_to_float(chromo_ch2[0:32])
    y_ch2 = binary_to_float(chromo_ch2[32:64])
    child2 = [x_ch2, y_ch2]

    return child1, child2
