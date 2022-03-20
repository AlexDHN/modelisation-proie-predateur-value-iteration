#actions possibles: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)

A = [ (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2) ] #liste des actions

etat = []
intermediaire_etat = [[[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]]

base1 = [[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]


for i in range(5):
    for j in range(5):
        base1[i][j] = 1
        intermediaire_etat.append(base1)
        base1 = [[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]


for i in range(26):
    for j in range(26):
        L = [intermediaire_etat[i],intermediaire_etat[j]]
        etat.append(L)
        L = []

#etat est la matrice contenant les 26x26 = 676 états. Un état = deux matrices (l'une représentant les carrés perceptifs à t-1 et l'autre à t). Tous les cas possibles sont recensés.


def value_iteration(gamma, epsilon):

    V = [0]*676 #26x26 = 676 états possibles
    V_prime = [0]*676

    while V < espilon :
        for i in range(len(V)):
            V[i] = score(etat[i])
            T = [0]*9
            for j in range(len(V)):

                T[j]+= gamma*V[j]*p(etat_j,etat_i,A[j]) # probas pour aller à i depuis j avec action a[j]
            V[i] = max(T)


        somme = 0
        for k in range(len(V)):
            somme += abs(V[i]-V_prime[i])

        if somme < tol:
            break

    PLAN OPTIMAL POUR CHAQUE ETAT

