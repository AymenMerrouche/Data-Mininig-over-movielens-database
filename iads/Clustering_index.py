import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .import kmoyennes as km


def dist_vect_data(w, df):
    return df.apply(lambda x: km.dist_vect(x, w), axis=1).max()


def dist_intracluster(df):
    return df.apply(lambda x: dist_vect_data(x, df), axis=1).max()


def global_intraclusters(df, dico):
    l = []
    maxi = 0
    for i in dico:
        val = dist_intracluster(df.iloc[dico[i]])
        if(val > maxi):
            maxi = val
    return maxi


def dist_vecte(x, w):
    if x.equals(w):
        return 999
    return km.dist_vect(x, w)


def dist_vect_centro(w, df):
    return df.apply(lambda x: dist_vecte(x, w), axis=1).min()


def sep_clusters(df):
    return df.apply(lambda x: dist_vect_centro(x, df), axis=1).min()


def evaluation(index, df, centres, dico):
    if index == "Dunn":
        return global_intraclusters(df, dico)/sep_clusters(centres)
    elif index == "XB":
        return km.inertie_globale(df, dico)/sep_clusters(centres)
