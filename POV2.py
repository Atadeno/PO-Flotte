import json
import pandas as pd
import random
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


### LISTES DES OBJETS ###

## PORT ##

# list_port_med              un dictionnaire représentant la liste des ports de méditerranée avec comme clés ['NOM_SUBDI', 'LONGITUDE_SUBDI', 'LATITUDE_SUBDI', 'DEPARTEMENT_SUBDI']
# coord_port_med             un tableau numpy donnant les coordonnées des ports de méditerranée
# nom_port_med               une liste des noms des ports de méditerranée

## INTERVENTIONS ##

# data_dict_intervention     un dictionnaire des données sur les interventions avec comme clés ['beginning', 'end', 'type', 'nbJoursInt', 'nomEsm', 'esm', 'typeEsm', 'coord', 'time', 'listBoat', 'retrieve', 'double', 'reste']
# dict_intervention_type     un dictionnaire découpant les interventions selon le type avec comme clés ['1', '2', '3'] chaque intervention étant un dictionnaire
# coord_intervention_i       une liste des coordonnées des interventions de type i
# coord_interventions        une liste de toutes les interventions sans distinction de type
# kmeans                     réalistion de l'algorithme du kmean sur la liste des interventions coord_interventions
# numero_cluster             une liste qui donne le numéro de cluster de chaque intervention dans l'ordre des éléments de coord_interventions
# centres                    une liste des centres de gravité de chaque cluster
# valeur_graphique           une liste de listes des coordonnées des interventions par numéro de cluster afin de les représenter sur la carte
# cluster_plot               une liste des clusters comme valeur_graphique mais pour tracer 

## BATEAUX ##

# data_dict_boat             un dictionnaire des données sur les bateaux avec comme clés  ['Nom', 'Port', 'Nature', 'Vitesse', 'Rayon', 'Habitable', 'Conso', 'Cout', 'Nombre_Semaines', 'Nombre_Heures', 'Jours_Disponibles_Reels', 'Capacite_Emport_GO', 'Escale']
# list_boat_med              une liste des bateaux de méditerranée où chaque bateau est un dictionnaire
# jours_Disponibles_Type     une liste de listes des jours disponibles réels pour chaque type de bateaux [0]->Vedettes [1]->Baliseurs [2]->Navires de travaux
# jours_Disponibles_Moyen    une liste des jours disponibles réels en moyenne pour chaque type de bateaux [0]->Vedettes [1]->Baliseurs [2]->Navires de travaux

print('\n'+'Traitement des fichiers json et csv'+'\n')

### LISTE DES PORTS ###

list_port_med = pd.read_csv('..\data\portMed.csv').to_dict()
#print(list_port_med.keys())
coord_port_med=np.array([list(list_port_med['LONGITUDE_SUBDI'].values()),list(list_port_med['LATITUDE_SUBDI'].values())]).transpose()
nom_port_med=list(list_port_med['NOM_SUBDI'].values())
print('Liste des ports de la côte méditerranéenne: ',nom_port_med,'\n')

### LISTE DES INTERVENTIONS MEDITERRANNEE ###

with open('..\data\interventions\interventionsTestMed1_hyp1.json') as json_data:
    data_dict_intervention = json.load(json_data)
#print(data_dict_intervention[0].keys())

### LISTE DES BATEAUX MEDITERRANNEE ###

with open('..\data\BoatSheet0_V5.json') as json_data:
    data_dict_boat = json.load(json_data)
print(data_dict_boat)

list_boat_med = []
for i in range(len(data_dict_boat)):
    if data_dict_boat[i]["Port"] in nom_port_med:
        list_boat_med.append(data_dict_boat[i])
#print(list_boat_med) #affiche la liste des bateaux

print('La liste des bateaux de la côte méditerranéenne:'+'\n')
for i in range(len(list_boat_med)):
    print(list_boat_med[i]["Port"],list_boat_med[i]["Nature"])
#print(list_boat_med)
#print(data_dict[:3])
print('')

jours_Disponibles_Type=[[] for i in range(3)] #On fait un tableau pour différencier les types afin de déterminer le nombre de bateau pouvant traiter les interventions de type 1, 2 ou 3
for d in list_boat_med:
    if d['Nature']=='Vedette':
        jours_Disponibles_Type[0].append(d['Jours_Disponibles_Reels'])
    elif d['Nature']=='Navire de travaux':
        jours_Disponibles_Type[2].append(d['Jours_Disponibles_Reels'])
    else:
        jours_Disponibles_Type[1].append(d['Jours_Disponibles_Reels'])

jours_Disponibles_Moyen=[]
for i in range (len(jours_Disponibles_Type)):
    s=0
    l=len (jours_Disponibles_Type[i])
    for j in range (l):
        s+=jours_Disponibles_Type[i][j]
    jours_Disponibles_Moyen.append(s/l)


### DICTIONNAIRE DES INTERVENTIONS ###

dict_intervention_type= {}
dict_intervention_type['1']=[]
dict_intervention_type['2']=[]
dict_intervention_type['3']=[]

for t in range(1,4):
    for i in range(len(data_dict_intervention)):
        if data_dict_intervention[i]["type"]==t:
            dict_intervention_type[str(t)].append(data_dict_intervention[i])

### LES INTERVENTIONS DE NIVEAU 3 ###

list_intervention_3=[] #liste des coordonnées des interventions de type 3
#print(dict_intervention_type['3'][0].keys()) #affiche les clés d'une intervention
for i in range(len(dict_intervention_type['3'])):
    #list_intervention_3.append([dict_intervention_type['3'][i].get('retrieve'),dict_intervention_type['3'][i].get('double'),dict_intervention_type['3'][i].get('reste')])
    list_intervention_3.append(dict_intervention_type['3'][i].get('coord')) #dict_intervention_type['3'][i].get('nomEsm')])
      
coord_intervention_3=list_intervention_3 #liste des coordonnées des interventions de type 3
#print(coord_intervention_3)


### LES INTERVENTIONS DE NIVEAU 2 ###

list_intervention_2=[]
for i in range(len(dict_intervention_type['2'])):
    list_intervention_2.append(dict_intervention_type['2'][i].get('coord'))
coord_intervention_2=list_intervention_2 #liste des coordonnées des interventions de type 2

### LES INTERVENTIONS DE NIVEAU 1 ###

list_intervention_1=[]
for i in range(len(dict_intervention_type['1'])):
    list_intervention_1.append(dict_intervention_type['1'][i].get('coord'))
coord_intervention_1=list_intervention_1 #liste des coordonnées des interventions de type 1


### PROPORTION SELON LE TYPE D'INTERVENTION DANS LA MER MEDITERRANEE ###
total=len(list_intervention_1+list_intervention_2+list_intervention_3)
print('Interventions de niveau 1: ',len(list_intervention_1),round(100*len(list_intervention_1)/total,1),'%')
print('Interventions de niveau 2: ',len(list_intervention_2),round(100*len(list_intervention_2)/total,1),'%')
print('Interventions de niveau 3: ',len(list_intervention_3),round(100*len(list_intervention_3)/total,1),'%')
print('Total:                     ',total)
print('\n')

### CLUSTER AVEC ALGORITHME K MEAN ###

k=36 #On remarque qu'il y a 3 grandes zones contenant certains ports (Corse, Sète et Marseille et Toulon et Cannes), on décompose en 12 mois et 3 zones soit 36 clusters
print('Nombre de Clusters:',k)

coord_interventions=coord_intervention_1+coord_intervention_2+coord_intervention_3

kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(coord_intervention_1+coord_intervention_2+coord_intervention_3))
numero_cluster=kmeans.predict(coord_interventions) #On associe à chaque intervention un numéro de cluster
centres=kmeans.cluster_centers_
centres=centres.transpose()

valeur_graphique=[[] for i in range(k)]
for i in range(len(numero_cluster)):
    valeur_graphique[numero_cluster[i]].append(coord_interventions[i])


#colors=['b','g','orange','c','m','y']
colors=list(mcolors.TABLEAU_COLORS.values()) #OU CSS4 OU BASE
cluster_plot=[]
plt.scatter(centres[0], centres[1],c = 'k', marker = 'o', linewidth = 1)
for i in range(k):
    cluster_plot.append(np.array(valeur_graphique[i]).transpose())
    plt.scatter(cluster_plot[i][0], cluster_plot[i][1],c = colors[i%len(colors)], marker = 'd', linewidth = 0.5)
for i in range(len(coord_port_med)):
    plt.scatter(coord_port_med[i][0], coord_port_med[i][1],c = 'r', marker = 's', linewidth = 3)


cluster=[[] for i in range(k)] #On redéfnie les clusters correctement en une liste d'intervention avec chaque intervention sous forme de dictionnaire
for i in range(len(numero_cluster)):
    for j in range(len(data_dict_intervention)):
        if data_dict_intervention[j].get('coord') == coord_interventions[i]:
            cluster[numero_cluster[i]].append(data_dict_intervention[i])
            break

for i in range(len(cluster)):
    print('Cluster n°',i+1,len(cluster[i]))


### AFFICHAGE MEDITERRANEE AVEC LE TRAIT DE CÔTE ###

"""
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#setup Lambert Conformal basemap.

m = Basemap(llcrnrlon=2.8,llcrnrlat=41,urcrnrlon=10,urcrnrlat=44,resolution='h', lat_0 = 42.5, lon_0 = 6.5)
#draw coastlines
m.drawcoastlines()
#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(lake_color='aqua')

plt.show()
#plt.savefig('Cluster.png')
#plt.close()
"""

### CALCUL DU NOMBRE K DE CLUSTER ###
#           
#print(kmeans.predict(coord_port_med)) #Association d'un port à son cluster le plus proche
##print(nom_port_med)

plt.show()

def fromClusterCoordToClusterInterventions():
    """ clusterCoord (liste de liste d'interventions): la liste des clusters ne contenant que les coordonnÃ©es
    renvoie les clusters avec toutes les donnÃ©es des interventions, ie une liste de liste d'interventions """
    clusterInterventions = [[] for i in range(k)]
    j = 0
    for num_clus in numero_cluster:
        for i in range(len(data_dict_intervention)):
            if data_dict_intervention[i]["coord"] == coord_interventions[j]:
                clusterInterventions[num_clus].append(data_dict_intervention[i])
                break
        j += 1
    return clusterInterventions

clusterInter = fromClusterCoordToClusterInterventions()


nb_semaine = 52
nb_jours_par_semaine = 4
nb_interventions = len(data_dict_intervention)
nb_interventions_par_semaine = nb_interventions / nb_semaine

def repartitionAlea(listInter):
    alea = [[] for i in range(nb_semaine)]
    listInterRemoved = listInter[:]
    moy_interventions_par_semaine = len(listInter) // nb_semaine
    for i in range(nb_semaine):
        compt = 0
        listInter = listInterRemoved[:]
        print(len(listInter))
        for inter in listInter:
            if compt == moy_interventions_par_semaine:
                break
            elif inter['beginning'] <= i*7 and inter['end'] >= (i+1)*7 - 1:
                alea[i].append(inter)
                listInterRemoved.remove(inter)
                compt += 1
    return alea

def repartitionAlea1(listInter):
    alea = [[] for i in range(nb_semaine)]
    listInterRemoved = listInter[:]
    moy_interventions_par_semaine = len(listInter) // nb_semaine
    for j in range(len(listInter)):
        for i in range(nb_semaine):
            listInter = listInterRemoved[:]
            for inter in listInter:
                if inter['beginning'] <= i*nb_jours_par_semaine and inter['end'] >= (i+1)*nb_jours_par_semaine - 1:
                    alea[i].append(inter)
                    listInterRemoved.remove(inter)
                    break
    return alea

print(len(clusterInter[0]))
alea = repartitionAlea1(clusterInter[0])
for l in alea:
    print(len(l), "  ", l)