# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
import random
import math



# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        raise NotImplementedError("Please Implement this method")
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        v = [] 
        for i in range(dataset.size()):
            acc = dataset.getY(i) - self.predict(dataset.getX(i))
            if acc == 0:
                v.append(1) 

        return sum(v) / dataset.size()

# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.w = np.random.randn(input_dimension)
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        return np.dot(x, self.w)

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        print("Pas d'apprentissage pour ce classifieur")
    
# ---------------------------
class ClassifierKNN(Classifier):
    def __init__(self, input_dimension,k):
            self.input_dimension=input_dimension
            self.k=k
            
    def train(self, labeledSet):
            self.matrice=[]
            for i in range(labeledSet.size()):
                self.matrice.append([labeledSet.getX(i),labeledSet.getY(i)])
                
    def distance(self,a,b):
        sum=0
        for i in range(len(a)):
            sum += (a[i]-b[i])**2
        return sum
    
    def predict(self,x):

        dist=[]
        b=[]
        for i in self.matrice:
            a=i[0]
            d=self.distance(a,x)
            dist.append([i,d])
            b.append(d)    
        sortedInd=np.argsort(b)
        sum=0
        for i in range(self.k):
            sum+=dist[sortedInd[i]][0][1]
           
        return np.sign(sum)
    
    def accuracy(self, dataset):
            cpt=0
            for i in range (dataset.size()):
                if ((self.predict(dataset.getX(i)))==dataset.getY(i)):
                    cpt+=1
            
            return (cpt/dataset.size())
# ---------------------------
class ClassifierPerceptronRandom(Classifier):
    def __init__(self, input_dimension):
        """ Argument:
                - input_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.w = np.random.rand(input_dimension)

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        if z > 0:
            return +1
        else:
            return -1
        
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        print("No training needed")


################################### perceptron kernel ##############################################
class KernelPoly:
    def transform(self,x):
        x1=x[0]
        x2=x[1]
        return  np.asarray([1,x1,x2,x1*x1,x2*x2,x1*x2])
        
class ClassifierPerceptronKernel(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.dimension_kernel=dimension_kernel
        self.learning_rate=learning_rate
        self.kernel=kernel
        self.w= [0]*dimension_kernel
        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(self.kernel.transform(x), self.w)
        if z < 0:
            return -1
        else:
            return +1

    
    def train(self,labeledSet):       
        for i in range(1):
            j=random.randint(0,labeledSet.size()-1)
            x=labeledSet.getX(j)
            y=labeledSet.getY(j)
            if ((self.predict(x)*y)<=0):
                self.w+= self.learning_rate*self.kernel.transform(x)*y
                
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        cpt=0
        for i in range (dataset.size()):
            if ((self.predict(dataset.getX(i)))==dataset.getY(i)):
                cpt+=1
        return (cpt/dataset.size())


###################################### perceptron #####################################################
class ClassfierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        self.w=[0]*input_dimension


    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        if z < 0:
            return -1
        else:
            return +1

    def score(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        return z
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        for i in range(1):
            j=random.randint(0,labeledSet.size()-1)
            x=labeledSet.getX(j)
            y=labeledSet.getY(j)
            if ((self.predict(x)*y)<0):
                self.w+= self.learning_rate*x*y
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        cpt=0
        for i in range (dataset.size()):
            if ((self.predict(dataset.getX(i)))==dataset.getY(i)):
                cpt+=1
        return (cpt/dataset.size())

    
################################################## Moindre caree regression #######################################################
class ClassifierMcDetScore(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        self.w=[0]*input_dimension


    def score(self,x):
        return np.dot(x, self.w)
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        grad=[0]*self.input_dimension
        for i in range(labeledSet.size()):
            x=labeledSet.getX(i)
            y=labeledSet.getY(i)
            grad = grad + ((2*x)*(y-self.score(x)))
        self.w = self.w + (self.learning_rate*(grad / labeledSet.size()))
        
    def cost(self, dataset):
        cpt=0
        somme =0
        for i in range (dataset.size()):
            somme +=math.fabs(self.score(dataset.getX(i))-dataset.getY(i))
            cpt+=1
        return (somme /cpt)



############################## Moindre caree stochastique ########################################
class ClassifierMcSto(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate,itern):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        self.w=[0]*input_dimension
        self.itern=itern


    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        if z < 0:
            return -1
        else:
            return +1
    def score(self,x):
        return np.dot(x, self.w)

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        for i in range(self.itern):
            j=random.randint(0,labeledSet.size()-1)
            x=labeledSet.getX(j)
            y=labeledSet.getY(j)  
            self.w+= self.learning_rate*2*x*(y-self.predict(x))
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        cpt=0
        for i in range (dataset.size()):
            if ((self.predict(dataset.getX(i)))==dataset.getY(i)):
                cpt+=1
        return (cpt/dataset.size())
    def cost(self, dataset):
        cpt=0
        sum=0
        for i in range (dataset.size()):
            sum+=math.fabs(self.score(dataset.getX(i))-dataset.getY(i))
            cpt+=1
        print(sum/cpt)
        return (sum/cpt)
###################################Perceptron deterministe ##############################################
class ClassifierPerceptronDet(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        self.w=[0]*input_dimension



    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        if z < 0:
            return -1
        else:
            return +1

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        grad=[0]*self.input_dimension
        for i in range(labeledSet.size()):

            x=labeledSet.getX(i)
            y=labeledSet.getY(i)  
            
            if (self.predict(x)*y<=0):
                grad+=x*y
        self.w+= self.learning_rate*grad
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        cpt=0
        for i in range (dataset.size()):
            if ((self.predict(dataset.getX(i)))==dataset.getY(i)):
                cpt+=1
        return (cpt/dataset.size())
############################################## Moindre caree detreministe ##################################
class ClassifierMcDet(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        self.w=[0]*input_dimension



    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        if z < 0:
            return -1
        else:
            return +1

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        grad=[0]*self.input_dimension
        for i in range(labeledSet.size()):
            x=labeledSet.getX(i)
            y=labeledSet.getY(i)  
            grad+=2*x*(y-self.predict(x))
        self.w+= self.learning_rate*grad/labeledSet.size()
    def accuracy(self, dataset):
        cpt=0
        for i in range (dataset.size()):
            if ((self.predict(dataset.getX(i)))==dataset.getY(i)):
                cpt+=1
        return (cpt/dataset.size())
