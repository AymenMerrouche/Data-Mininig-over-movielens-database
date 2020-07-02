import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from . import LabeledSet as ls
from . import Classifiers as cl
import random

def shannon(P):
    k=len(P)
    if (k==1 or k==0) : return 0
    l=[p*math.log(p,k) if p!=0 else 0 for p in P]
    return float(-np.sum(l))

def classe_majoritaire(the_set):
    cpt=0
    for i in range (the_set.size()-1):
        cpt+= the_set.getY(i)
    if cpt>=0:
        return 1
    return -1


def entropie(the_set):
    if the_set.size()==0:
        return 0
    classes={}
    for i in range(the_set.size()):
        if (the_set.getY(i)[0]not in classes.keys()):
            classes[the_set.getY(i)[0]]=1
        else:
            classes[the_set.getY(i)[0]]+=1
    tab=list(classes.values())
    tab=np.array(tab)
    tab=tab/the_set.size()

    return shannon(tab)


def discretise(LSet, col):
    """ LabelledSet * int -> tuple[float, float]
        Hypothèse: LSet.size() >= 2
        col est le numéro de colonne sur X à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0     
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)
    
    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2       
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E. 
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins 
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        else:
            sup_plus += 1
    nb_total = (sup_plus + sup_moins) # nombre d'exemples total dans E
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
        else:
            inf_plus += 1
            sup_plus -= 1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = (inf_moins + inf_plus)*1.0     # rem: on en fait un float pour éviter
        nb_sup = (sup_moins + sup_plus)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf \
                       + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)


def divise(LSet,att,seuil):
    Set1 = ls.LabeledSet(2)
    Set2 = ls.LabeledSet(2)
    for i in range (LSet.size()):
        if LSet.getX(i)[att]>seuil:
            Set2.addExample(LSet.getX(i),LSet.getY(i))
        else:
            Set1.addExample(LSet.getX(i),LSet.getY(i))
    return Set1,Set2


def construit_AD(LSet,epsilon):
    un_arbre= ArbreBinaire()
    ent=entropie(LSet)
    if (ent<=epsilon):
        un_arbre.ajoute_feuille(classe_majoritaire(LSet))
        return un_arbre
    else:
        mini=1.1
        imin=0

        for i in range (len(LSet.x[0])):
            seuil,entro=discretise(LSet,i)
            if entro<=mini:
                mini = entro
                imin=i
        
        seuil,entro=discretise(LSet,imin)
        gain= ent-entro
        if (gain<=epsilon):
            un_arbre.ajoute_feuille(classe_majoritaire(LSet))
            return un_arbre
        Linf, Lsup = divise(LSet,imin,seuil)
        
        un_arbre1 = construit_AD(Linf,epsilon)
        un_arbre2 = construit_AD(Lsup,epsilon)
        
        un_arbre3 = ArbreBinaire()
        un_arbre3.ajoute_fils(un_arbre1,un_arbre2,imin,seuil)
        
        return un_arbre3
import graphviz as gv
# Eventuellement, il peut être nécessaire d'installer graphviz sur votre compte:
# pip install --user --install-option="--prefix=" -U graphviz


class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille """
        return self.seuil == None
    
    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        """ ABinf, ABsup: 2 arbres binaires
            att: numéro d'attribut
            seuil: valeur de seuil
        """
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup
    
    def ajoute_feuille(self,classe):
        """ classe: -1 ou + 1
        """
        self.classe = classe
        
    def classifie(self,exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple: +1 ou -1
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.inferieur.classifie(exemple)
        return self.superieur.classifie(exemple)
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir
            l'afficher
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))
        
        return g

    
class ArbreDecision(cl.Classifier):
    # Constructeur
    def __init__(self,epsilon):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None
    
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        classe = self.racine.classifie(x)
        if (classe == 1):
            return(1)
        else:
            return(-1)
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision 
        self.set=set
        self.racine = construit_AD(set,self.epsilon)

    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        v = [] 
        for i in range(dataset.size()):
            acc = dataset.getY(i) - self.predict(dataset.getX(i))
            if acc == 0:
                v.append(1) 

        return sum(v) / dataset.size()

def tirage(VX,m,r):
    m=int(m)
    if (r):
        liste=[]
        for i in range (m):
            liste.append(random.choice(VX))
        return liste
    return random.sample(VX,m)

def echantillonLS(X,m,r):
    indice=np.arange(X.size())
    tab=tirage(indice,m,r)
    Set=ls.LabeledSet(2)
    for i in tab:
        Set.addExample(X.getX(i),X.getY(i))
    return Set

class ClassifierBaggingTree(cl.Classifier):
    def __init__(self,B,pourcent,epsilon,r):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.foret = []
        self.pourcent=pourcent
        self.B=B
        self.r=r
    
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        sum=0
        for i in self.foret:
            sum+=i.classifie(x)
        if (sum >= 0):
            return(1)
        else:
            return(-1)
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,Set):
        # construction de l'arbre de décision 
        self.set=Set    
        for i in range(self.B):
            ech=echantillonLS(self.set,self.pourcent*self.set.size(),self.r)
            self.foret.append(construit_AD(ech,self.epsilon))

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        v = [] 
        for i in range(dataset.size()):
            acc = dataset.getY(i) - self.predict(dataset.getX(i))
            if acc == 0:
                v.append(1) 

        return sum(v) / dataset.size()

def difference(set1,set2):
    set3 =ls.LabeledSet(2)
    for i in range(set1.size()):
        exist=False
        for j in range(set2.size()):
            if (set2.getX(j)[0]==set1.getX(i)[0])and(set2.getX(j)[1]==set1.getX(i)[1]):
                exist=True
        if (exist==False):
            set3.addExample(set1.getX(i),set1.getY(i))
    return set3

class ClassifierBaggingTreeOOB(ClassifierBaggingTree):
    def __init__(self,B,pourcent,epsilon,r):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.foret = []
        self.pourcent=pourcent
        self.B=B
        self.r=r
    
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        sum=0
        for i in self.foret:
            sum+=i.classifie(x)
        if (sum >= 0):
            return(1)
        else:
            return(-1)
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,Set):
        # construction de l'arbre de décision 
        self.x = []
        self.t = []
        self.set=Set    
        for i in range(self.B):
            ech=echantillonLS(self.set,self.pourcent*self.set.size(),self.r)
            self.foret.append(construit_AD(ech,self.epsilon))
            self.x.append(ech)
            self.t.append(difference(self.set,ech))

    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        v = [] 
        for i in range(dataset.size()):
            
            acc = dataset.getY(i) - self.predict(dataset.getX(i))
            if acc == 0:
                v.append(1) 

        return sum(v) / dataset.size()
    def OOB(self):
        acca=[]
        for i in range(len(self.foret)):
            Sum=0
            for k in range(self.t[i].size()):
                acc = self.t[i].getY(k) - self.foret[i].classifie(self.t[i].getX(k))
                if acc == 0:
                    Sum+=1
            acca.append(Sum/self.t[i].size())
        return sum(acca)/len(self.foret)