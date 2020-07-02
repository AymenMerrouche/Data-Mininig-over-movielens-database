import numpy as np
import pandas as pd
import math
import scipy.cluster.hierarchy
from datetime import datetime as dt
import matplotlib.pyplot as plt


def normalisation(df):
    minimum = df.min()
    maximum = df.max()
    return (df-minimum)/(maximum-minimum)


def dist_vect(x, y):
    return math.sqrt(((x-y)**2).sum())


def centroide(df):
    return pd.DataFrame(df.mean()).T


def inertie_cluster(df):
    return df.apply(lambda x: dist_vect(x, centroide(df).iloc[0])**2, axis=1).sum()


def initialisation(K, df):
    return df.sample(n=K)


def plus_proche(exp, df):
    inter = df.apply(lambda x: dist_vect(x, exp)**2, axis=1)
    return inter.reset_index(drop=True).idxmin()


def affecte_cluster(df, centroides):
    dico = {}
    inter = df.as_matrix()
    for i in range(len(inter)):
        a = plus_proche(inter[i], centroides)
        if a not in dico:
            dico[a] = [i]
        else:
            dico[a].append(i)
    return dico


def nouveaux_centroides(df, dico):
    l = []
    for i in dico:
        l.append(centroide(df.iloc[dico[i]]))
    return pd.concat(l).reset_index(drop=True)


def inertie_globale(df, dico):
    sum = 0
    for i in dico:
        sum += inertie_cluster(df.iloc[dico[i]])
    return sum


def kmoyennes(K, df, epsilon, tmax):
    centroides = initialisation(K, df)
    dico = affecte_cluster(df, centroides)
    centroides = nouveaux_centroides(df, dico)
    oldj = inertie_globale(df, dico)

    t = 0
    fin = False
    while((t <= tmax)and(not fin)):
        dico = affecte_cluster(df, centroides)
        centroides = nouveaux_centroides(df, dico)
        newj = inertie_globale(df, dico)
        if math.fabs(newj-oldj) < epsilon:
            fin = True
        # print('iteration', t, 'inertie', newj,
        #      'difference', math.fabs(newj-oldj))
        oldj = newj
        t += 1
    return centroides, dico
