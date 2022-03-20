import numpy as np
import matplotlib.pyplot as plt
from random import random
from random import randint
import copy
import os

def deplacement(x,y,orient,g,d): #une roue à 1 et une à 0 tourne sur place de 45deg
#une roue à 2 et une à 0 tourne sur place de 90deg
#une roue à 2 et une à 1 decrit un arc de cercle de périphérie 1 et à la fin aura tournée de 45deg
    vecorient=np.array([cos(orient),sin(orient)])
    vec2=np.array([np.cos(orient-(np.pi/2)),np.sin(orient-(np.pi/2))])
    if g==1 and d==2:
        dep=vec2*(((2*sqrt(2))/np.pi)-(4/np.pi))+vecorient*((2*sqrt(2))/np.pi)
        y=y+dep[1]
        x=x+dep[0]
        orient=orient+(np.pi/4)
        return(x,y,orient)
    if g==2 and d==1:
        dep=vec2*((4/np.pi)-((2*sqrt(2))/np.pi))+vecorient*((2*sqrt(2))/np.pi)
        y=y+dep[1]
        x=x+dep[0]
        orient=orient-(np.pi/4)
        return(x,y,orient)
    if g==0 and d==0:
        return(x,y,orient)
    if (g==0)^(d==0):
        orient= orient - g*(np.pi/4) + d * (np.pi/4)
        return(x,y,orient)
    if g==d:
        dep=vecorient*g
        x=x+dep[0]
        y=y+dep[1]
        return(x,y,orient)


def norme(x,y):
    return(np.sqrt(pow(x,2)+pow(y,2)))


def rotation(orient,X,Y):
    n=len(X)
    vecorient=np.array([cos(orient),sin(orient)])
    vec2=np.array([np.cos(orient-(np.pi/2)),np.sin(orient-(np.pi/2))])
    for i in range(0,n):
        dep=vec2*Y[i]+vecorient*X[i]
        X[i]=dep[0]
        Y[i]=dep[1]
    return(X,Y)

def matriceperception(predx,predy,orient,proiex,proiey): #pas obligé de refaire une matrice car le cout est le même si on cherche d'abord la case de la proie dans l'ancienne matrice pour la mettre à 0 (complexité en ligne*colonne dans les deux cas)
    vecorient=np.array([np.cos(orient),np.sin(orient)])
    vec2=np.array([np.cos(orient+(np.pi/2)),np.sin(orient+(np.pi/2))])
    temp=np.array([proiex-predx,proiey-predy])
    x=np.dot(temp,vecorient) #projection de la position relative de la proie sur le nouvel axe x (vecorient)
    y=np.dot(temp,vec2) #projection de la position relative de la proie sur le nouvel axe y (vec2)
    mat=[]
    for i in range(0,5): #création de la matrice
        ligne=[]
        for j in range(0,5):
            ligne=ligne+[0]
        mat=mat+[ligne]
    if max(abs(x),abs(y))<2.5: #positionnement de la proie que si elle est dans la matrice
        i=np.floor(y+2.5) #la position allais de -2.5 à 2.5, maintenant elle va de 0 à 4(entier) pour correspondre aux ligne etcolonne de la matrice
        j=np.floor(x+2.5)
        mat[4-int(i)][int(j)]=1 #floor ne transforme pas en int du coup je le fait là
    return mat

def posproie(mat):
    for i in range(0,5):
        for j in range(0,5):
            if mat[i][j]==1:
                return (i,j) #le return nous fais sortir des for, résultat ligne d'abord puis colonne
    return False #on ne rentre ici que si on a rien trouvé

def score(mat,bonus,malus):
    if mat[2][2]==1:
        mat[2][2]=0
        return bonus
    temp=posproie(mat)
    if temp: #utilisé comme ça if verif l'existence de temp donc quand temp = (2,4) par exemple on rentre dans le for même si c'est pas un bool, sinon temp=False donc on rentre pas dedans
        temp=(abs(temp[0]-2),abs(temp[1]-2)) #ça permet de vérifier plus rapidement si on est sur un bord car 0 devient abs(-2) et 4 devient 2 donc il suffit que le max des coord soit égal à 2 pour être sur un bord
        if max(temp)==2:
            return malus
    return 0 #dans les autres cas on sort 0

