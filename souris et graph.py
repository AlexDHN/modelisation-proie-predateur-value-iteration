import numpy as np
import matplotlib.pyplot as plt
from random import randint
from random import random
import copy
import os

def deplacement(x,y,orient,g,d): #une roue à 1 et une à 0 tourne sur place de 45deg
#une roue à 2 et une à 0 tourne sur place de 90deg
#une roue à 2 et une à 1 decrit un arc de cercle de périphérie 1 et à la fin aura tournée de 45deg
    vecorient=np.array([np.cos(orient),np.sin(orient)])
    vec2=np.array([np.cos(orient-(np.pi/2)),np.sin(orient-(np.pi/2))])
    r=random()
    if g==1 and d==2:
        dep=vec2*(((2*np.sqrt(2))/np.pi)-(4/np.pi))+vecorient*((2*np.sqrt(2))/np.pi)
        y=y+dep[1]
        x=x+dep[0]
        orient=orient+(np.pi/4)
        vecorient=np.array([np.cos(orient),np.sin(orient)])
        vec2=np.array([np.cos(orient-(np.pi/2)),np.sin(orient-(np.pi/2))])
        if r<0.1:
            x=x+vecorient[0]
            y=y+vecorient[1]
        elif r<0.2:
            x=x-vecorient[0]
            y=y-vecorient[1]

        return(x,y,orient)
    if g==2 and d==1:
        dep=vec2*((4/np.pi)-((2*np.sqrt(2))/np.pi))+vecorient*((2*np.sqrt(2))/np.pi)
        y=y+dep[1]
        x=x+dep[0]
        orient=orient-(np.pi/4)
        vecorient=np.array([np.cos(orient),np.sin(orient)])
        vec2=np.array([np.cos(orient-(np.pi/2)),np.sin(orient-(np.pi/2))])
        if r<0.1:
            x=x+vecorient[0]
            y=y+vecorient[1]
        elif r<0.2:
            x=x-vecorient[0]
            y=y-vecorient[1]
        return(x,y,orient)
    if g==0 and d==0:
        vecorient=np.array([np.cos(orient),np.sin(orient)])
        vec2=np.array([np.cos(orient-(np.pi/2)),np.sin(orient-(np.pi/2))])
        if r<0.1:
            x=x+vecorient[0]
            y=y+vecorient[1]
        elif r<0.2:
            x=x-vecorient[0]
            y=y-vecorient[1]
        return(x,y,orient)
    if (g==0)^(d==0):
        orient= orient - g*(np.pi/4) + d * (np.pi/4)
        vecorient=np.array([np.cos(orient),np.sin(orient)])
        vec2=np.array([np.cos(orient-(np.pi/2)),np.sin(orient-(np.pi/2))])
        if r<0.1:
            x=x+vecorient[0]
            y=y+vecorient[1]
        elif r<0.2:
            x=x-vecorient[0]
            y=y-vecorient[1]
        return(x,y,orient)
    if g==d:
        dep=vecorient*g
        x=x+dep[0]
        y=y+dep[1]
        vecorient=np.array([np.cos(orient),np.sin(orient)])
        vec2=np.array([np.cos(orient-(np.pi/2)),np.sin(orient-(np.pi/2))])
        if r<0.1:
            x=x+vecorient[0]
            y=y+vecorient[1]
        elif r<0.2:
            x=x-vecorient[0]
            y=y-vecorient[1]
        return(x,y,orient)

def deplacement2(x,y,orient,g,d): #une roue à 1 et une à 0 tourne sur place de 45deg
#une roue à 2 et une à 0 tourne sur place de 90deg
#une roue à 2 et une à 1 decrit un arc de cercle de périphérie 1 et à la fin aura tournée de 45deg
    vecorient=np.array([np.cos(orient),np.sin(orient)])
    vec2=np.array([np.cos(orient-(np.pi/2)),np.sin(orient-(np.pi/2))])
    if g==1 and d==2:
        dep=vec2*(((2*np.sqrt(2))/np.pi)-(4/np.pi))+vecorient*((2*np.sqrt(2))/np.pi)
        y=y+dep[1]
        x=x+dep[0]
        orient=orient+(np.pi/4)
        return(x,y,orient)
    if g==2 and d==1:
        dep=vec2*((4/np.pi)-((2*np.sqrt(2))/np.pi))+vecorient*((2*np.sqrt(2))/np.pi)
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
    if temp: #utilisé comme ça if verif l'existence de temp donc quand temp = (2,4) par exemple on rentre dans le if même si c'est pas un bool, sinon temp=False donc on rentre pas dedans
        temp=(abs(temp[0]-2),abs(temp[1]-2)) #ça permet de vérifier plus rapidement si on est sur un bord car 0 devient abs(-2) et 4 devient 2 donc il suffit que le max des coord soit égal à 2 pour être sur un bord
        if max(temp)==2:
            return malus
    return 0 #dans les autres cas on sort 0

#On va considérer que le robot peut avancer à x fois la vitesse de la proie
x = 0.8
angle_degre = randint(0,359)
angle_pi = np.pi*angle_degre/180 #direction pour un chemin linéaire

rayon = 2 #rayon pour le chemin circulaire
angle = 0 #Angle initial pour une trajectoire circulaire

def distance(x,y):
    return(np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2))

def deplacement_lineaire(old_pos,distance):
    add_x = distance*np.cos(angle_pi)
    add_y = distance*np.sin(angle_pi)
    return([old_pos[0]+add_x,old_pos[1]+add_y])

def deplacement_circulaire(centre,old_angle,distance):
    add_angle = np.arccos((distance**2-2*rayon**2)/(-2*rayon**2)) #théorème d'Al Kashi
    angle = old_angle + np.abs(add_angle)
    return([centre[0] + rayon*np.cos(angle), centre[1] + rayon*np.sin(angle), angle])

def deplacement_aleatoire(old_pos,distance):
    angl_pi = np.pi*randint(0,359)/180
    add_x = distance*np.cos(angl_pi)
    add_y = distance*np.sin(angl_pi)
    return([old_pos[0]+add_x,old_pos[1]+add_y])

def plot_direction_lineraire(pos_souris):
    pas = 20
    X = []
    Y = []
    for i in range(0,pas):
        poss = deplacement_lineaire(pos_souris,i)
        X.append(poss[0])
        Y.append(poss[1])
    plt.plot(X, Y,color="y",linestyle = '--')
    return()

def plot_ideal_direction_lineaire(pos_robot,pos_souris,distance):
    intersection = pos_souris
    compteur = 0
    while(np.sqrt((pos_souris[0]-intersection[0])**2+(pos_souris[1]-intersection[1])**2) < (np.sqrt((pos_robot[0]-intersection[0])**2+(pos_robot[1]-intersection[1])**2)/x)):
        compteur += 1
        intersection = deplacement_lineaire(intersection,distance)
        if compteur > 200:
            print("Le robot ne peut pas rattraper la proie, il ne va pas assez vite")
            return()
    plt.plot([intersection[0],pos_robot[0]], [intersection[1],pos_robot[1]],color="r",linestyle = '--',label = "Chemin optimal atteignable en "+str(compteur) +" itération(s)")
    plt.plot(intersection[0],intersection[1],marker='x',color='r')
    return()

def plot_direction_circulaire(centre):
    U = []
    V = []
    angl = angle
    for i in range(360):
        angl += np.pi*1/180
        U.append(centre[0] + rayon*np.cos(angl))
        V.append(centre[1] + rayon*np.sin(angl))
    plt.plot(U, V,color="y",linestyle = ':')
    return()

def plot_ideal_direction_circulaire(pos_robot,distance,centre,angle):
    L = []
    for i in range(100):
        aux = deplacement_circulaire(centre,angle,distance)
        L.append([aux[0],aux[1]])
        angle += aux[2]
    W = []
    for i in L:
        W.append(np.sqrt((pos_robot[0]-i[0])**2+(pos_robot[1]-i[1])**2)/(distance*x))
    aux = 0
    min = W[0]
    for i in range(1,len(W)):
        if (W[i]<min and i+1<W[i]):
            min = W[i]
            aux = i
    plt.plot([L[aux][0],pos_robot[0]], [L[aux][1],pos_robot[1]],color="r",linestyle = '--',label = "Chemin optimal atteignable en "+str(aux+1) +" itération(s)")
    plt.plot(L[aux][0],L[aux][1],marker='x',color='r')
    return()

def plot_direction_aleatoire(pos_souris,distance):
    U = []
    V = []
    angl = 0
    for i in range(360):
        angl += np.pi*1/180
        U.append(pos_souris[0] + distance*np.cos(angl))
        V.append(pos_souris[1] + distance*np.sin(angl))
    plt.plot(U, V,color="y",linestyle = ':')
    return()

def plot_ideal_direction_aleatoire(pos_robot,pos_souris):
    plt.plot([pos_souris[0],pos_robot[0]], [pos_souris[1],pos_robot[1]],color="r",linestyle = '--',label = "Chemin optimal")

def lineaire(x):
    return(a*x+b)

def rotation(orient,X,Y,predx,predy):
    n=len(X)
    vecorient=np.array([np.cos(orient),np.sin(orient)])
    vec2=np.array([np.cos(orient-(np.pi/2)),np.sin(orient-(np.pi/2))])
    for i in range(0,n):
        dep=(vec2*Y[i]+vecorient*X[i])+[predx,predy]
        X[i]=dep[0]
        Y[i]=dep[1]
    return(X,Y)

def tracer_carte(pos_robot, pos_souris,distance,g,d,angle,theta,nb_iteration,mode,Q,miam): # pos = [x,y] position de la souris par rapport au robot ; 0 mode linéaire ; 1 mode circulaire ; 2 mode aléatoire ; distance = avancée de la souris, pos_souris = position initiale de la souris, nb_iteration = nombre d'itération ; angle uniquement pour deplacement circulaire
    mat=matriceperception(pos_robot[0],pos_robot[1],theta,pos_souris[0],pos_souris[1])
    if (score(mat,5,-1)==5):
        print("miam miam miam")
        miam=miam+1
    if mat==matriceperception(0,0,0,10,10):
        angle_degre = randint(0,359)
        angle_pi = np.pi*angle_degre/180 #direction pour un chemin linéaire
        pos_souris=[random()*3+pos_robot[0]-1.5,random()*3+pos_robot[1]-1.5]

    pos_x_robot,pos_y_robot,new_theta = deplacement(pos_robot[0],pos_robot[1],theta,g,d)
    plt.figure(nb_iteration)
    plt.plot(pos_robot[0],pos_robot[1], "s",color="k",label = "Position initial du Robot")
    plt.plot(pos_x_robot,pos_y_robot, "s",color="b",label = "Nouvelle position du Robot")
    plt.plot([pos_x_robot,pos_robot[0]], [pos_y_robot,pos_robot[1]],color="b",linestyle = '--',label = "Avancement du Robot")
    if mode != 1:
        plt.plot(pos_souris[0], pos_souris[1], "o",color="g",label = "Souris")
    if mode ==0:
        new_pos = deplacement_lineaire(pos_souris,distance)
        plt.plot(new_pos[0],new_pos[1], "o",color="chartreuse",label = "Nouvelle position de la souris")
        plot_direction_lineraire(pos_souris)
        plot_ideal_direction_lineaire([pos_x_robot,pos_y_robot],new_pos,distance)
    elif mode == 1:
        new_pos = deplacement_circulaire(pos_souris,angle,distance)
        plt.plot(new_pos[0],new_pos[1], "o",color="chartreuse",label = "Nouvelle position de la souris")
        plot_ideal_direction_circulaire([pos_x_robot,pos_y_robot],distance,pos_souris,angle)
        plt.plot(new_pos[0], new_pos[1], "o",color="g",label = "Souris")
        angle = new_pos[2]
        plot_direction_circulaire(pos_souris)
    elif mode ==2:
        new_pos = deplacement_aleatoire(pos_souris,distance)
        plot_direction_aleatoire(new_pos,distance)
        plt.plot(new_pos[0],new_pos[1], "o",color="chartreuse",label = "Nouvelle position de la souris")
        plot_ideal_direction_aleatoire([pos_x_robot,pos_y_robot],new_pos)
    plt.title("Let's the hunt begin")
    plt.legend()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.yticks(np.arange(-10, 10, 1))
    plt.xticks(np.arange(-10, 10, 1))
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    for x in range(-3,3):
        X = [x+0.5 for i in range(6)]
        Y = [i+0.5 for i in range(-3,3)]
        (X,Y) = rotation(new_theta,X,Y,pos_x_robot,pos_y_robot)
        plt.plot(X, Y,color="black",linewidth = 0.5)
        Y = [x+0.5 for i in range(6)]
        X = [i+0.5 for i in range(-3,3)]
        (X,Y) = rotation(new_theta,X,Y,pos_x_robot,pos_y_robot)
        plt.plot(X, Y,color="black",linewidth = 0.5)
    vecorient=np.array([np.cos(new_theta),np.sin(new_theta)])
    fleche0 = np.array([pos_x_robot,pos_y_robot]) + vecorient*2.5
    fleche1 = vecorient*1
    plt.arrow(fleche0[0],fleche0[1],fleche1[0],fleche1[1],head_width = 0.3, facecolor = "k", edgecolor = "k")
    plt.show()
    mat=matriceperception(pos_x_robot,pos_y_robot,theta,new_pos[0],new_pos[1])
    #print(mat)
    if mat==matriceperception(0,0,0,10,10):
        angle_degre = randint(0,359)
        angle_pi = np.pi*angle_degre/180 #direction pour un chemin linéaire
        new_pos=[random()*3+pos_x_robot-1.5,random()*3+pos_y_robot-1.5]
    if nb_iteration == 0:
        print(miam)
        return(miam)
    else:
        (i,j)=calposproie(pos_robot[0],pos_robot[1],theta,pos_souris[0],pos_souris[1])
        (k,l)=calposproie(pos_x_robot,pos_y_robot,new_theta,new_pos[0],new_pos[1])
        (g,d)=calculmu(i,j,k,l,Q)
        if mode != 1:
            return(tracer_carte([pos_x_robot,pos_y_robot],new_pos,distance,g,d,angle,new_theta,nb_iteration-1,mode,Q,miam))
        else:
            return(tracer_carte([pos_x_robot,pos_y_robot],pos_souris,distance,g,d,angle,new_theta,nb_iteration-1,mode,Q,miam))


def score2(i,j,bonus,malus):
    if i==2 and j==2:
        return bonus
    temp=(abs(i-2),abs(j-2)) #ça permet de vérifier plus rapidement si on est sur un bord car 0 devient abs(-2) et 4 devient 2 donc il suffit que le max des coord soit égal à 2 pour être sur un bord
    if max(temp)==2:
        return malus
    return 0 #dans les autres cas on sort 0

def initV():
    V=[]
    for i in range(0,5):
        ligne1=[]
        for j in range(0,5):
            col1=[]
            for k in range(0,5):
                ligne2=[]
                for l in range(0,5):
                    ligne2=ligne2 + [score2(k,l,5,-1)]
                col1=col1 + [ligne2]
            ligne1=ligne1 + [col1]
        V=V+[ligne1]
    return V

def calposproie(predx,predy,orient,proiex,proiey): #pas obligé de refaire une matrice car le cout est le même si on cherche d'abord la case de la proie dans l'ancienne matrice pour la mettre à 0 (complexité en ligne*colonne dans les deux cas)
    vecorient=np.array([np.cos(orient),np.sin(orient)])
    vec2=np.array([np.cos(orient+(np.pi/2)),np.sin(orient+(np.pi/2))])
    temp=np.array([proiex-predx,proiey-predy])
    x=np.dot(temp,vecorient) #projection de la position relative de la proie sur le nouvel axe x (vecorient)
    y=np.dot(temp,vec2) #projection de la position relative de la proie sur le nouvel axe y (vec2)
    if max(abs(x),abs(y))<2.5: #positionnement de la proie que si elle est dans la matrice
        i=np.floor(y+2.5) #la position allais de -2.5 à 2.5, maintenant elle va de 0 à 4(entier) pour correspondre aux ligne etcolonne de la matrice
        j=np.floor(x+2.5)
        return(4-int(i),int(j))
    return False


def deplacementInverse(k,l):
    possible=[]
    for i in range(0,3): #création de la matrice
        ligne=[]
        for j in range(0,3):
            ligne=ligne+[[]]
        possible=possible+[ligne]
    for i in range(0,5):
        for j in range(0,5):
            for g in range(0,3):
                for d in range(0,3):
                    (x,y,orient)=deplacement2(0,0,0,g,d)
                    posp=calposproie(x,y,orient,j-2,2-i)
                    if posp:
                        if abs(posp[1]-l)==1 and posp[0]==k:
                            possible[g][d]= possible[g][d]+ [[posp[0],posp[1],0.1]]
                        if posp[0]==k and posp[1]==l:
                            possible[g][d]= possible[g][d]+ [[posp[0],posp[1],0.8]]
    return possible







def calculV(gamma,eps,V):
    V2=copy.deepcopy(V)
    diff=0
    for i in range(0,5):
        for j in range(0,5):
            for k in range(0,5):
                for l in range(0,5):
                    possible=deplacementInverse(i,j)
                    value=[]
                    for g in range(0,3):
                        for d in range(0,3):
                            val=0
                            for x in possible[g][d]:
                                val = val + x[2]*V[x[0]][x[1]][i][j]
                            value=value+[score2(k,l,5,-1)+gamma*val]
                    V2[i][j][k][l]=min(value)
                    diff=diff + abs(V2[i][j][k][l]-V[i][j][k][l])
    print(diff)
    if diff<eps*(1-gamma)/(2*gamma):
        return V2
    else:
        V=V2
        return calculV(gamma,eps,V)

def calculdepQ(k,l,g,d):
    (x,y,orient)=deplacement2(0,0,0,g,d)
    posp=calposproie(x,y,orient,l-2,2-k)
    if posp:
        if posp[1]==0:
            return [[posp[0],posp[1],0.8],[posp[0],posp[1]+1,0.1]]
        if posp[1]==4:
            return [[posp[0],posp[1],0.8],[posp[0],posp[1]-1,0.1]]
        return [[posp[0],posp[1],0.8],[posp[0],posp[1]-1,0.1],[posp[0],posp[1]+1,0.1]]




def calculQ(gamma,V):
    Q=[]
    for i in range(0,5):
        ligne1=[]
        for j in range(0,5):
            col1=[]
            for k in range(0,5):
                ligne2=[]
                for l in range(0,5):
                    col2=[]
                    for g in range(0,3):
                        gauche=[]
                        for d in range(0,3):
                            value=0
                            depq=calculdepQ(k,l,g,d)
                            if depq:
                                for x in depq:
                                    value=value+x[2]*V[k][l][x[0]][x[1]]
                                value=score2(k,l,5,-1)+gamma*value
                            else:
                                value=-10000
                            gauche=gauche+[value]
                        col2=col2+[gauche]
                    ligne2=ligne2+[col2]
                col1=col1 + [ligne2]
            ligne1=ligne1 + [col1]
        Q=Q+[ligne1]
    return Q

def calculmu(i,j,k,l,Q):
    max=-10000000
    maxgd=(0,0)
    for g in range(0,3):
        for d in range(0,3):
            if Q[i][j][k][l][g][d]>max:
                max=Q[i][j][k][l][g][d]
                maxgd=(g,d)
    return maxgd
##
V=initV()
Vstar=calculV(0.1,0.1,V)
Q=calculQ(0.1,Vstar)


##
x = 1.5
angle_degre = randint(0,359)
angle_pi = np.pi*angle_degre/180 #direction pour un chemin linéaire
pos_souris=[random()*3-1.5,random()*3-1.5]
(posi,posj)=calposproie(0,0,0,pos_souris[0],pos_souris[1])
(gauche,droite)=calculmu(posi,posj,posi,posj,Q)
tracer_carte([0,0],pos_souris,1,gauche,droite,0,0*np.pi/4,5,2,Q,0)



























