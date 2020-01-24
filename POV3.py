import json
import pandas as pd
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

print('\n')

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
#print(data_dict_boat)

list_boat_med = []
for i in range(len(data_dict_boat)):
    if data_dict_boat[i]["Port"] in nom_port_med:
        list_boat_med.append(data_dict_boat[i])
#print(list_boat_med) #affiche la liste des bateaux

print('La liste des bateaux de la côte méditerranéenne:'+'\n')
for i in range(len(list_boat_med)):
    print(list_boat_med[i]["Nom"],list_boat_med[i]["Port"])
#print(list_boat_med)
#print(data_dict[:3])
print('\n')

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
print('Nombre de Clusters:', k, '\n')

coord_interventions=coord_intervention_1+coord_intervention_2+coord_intervention_3 #Liste de toutes les interventions

kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(coord_interventions))
numero_cluster=kmeans.predict(coord_interventions) #On associe à chaque intervention un numéro de cluster
centres=kmeans.cluster_centers_
centres=centres.transpose()

valeur_graphique=[[] for i in range(k)]
for i in range(len(numero_cluster)):
    valeur_graphique[numero_cluster[i]].append(coord_interventions[i])

plt.figure()
#colors=['b','g','orange','c','m','y']
colors=list(mcolors.TABLEAU_COLORS.values()) #OU CSS4 OU BASE
cluster_plot=[]
#plt.scatter(centres[0], centres[1],c = 'k', marker = 'o', linewidth = 1)
for i in range(k):
    cluster_plot.append(np.array(valeur_graphique[i]).transpose())
    plt.scatter(cluster_plot[i][0], cluster_plot[i][1],c = colors[i%len(colors)], marker = 'd', linewidth = 0.5)
for i in range(len(coord_port_med)):
    plt.scatter(coord_port_med[i][0], coord_port_med[i][1],c = 'r', marker = 's', linewidth = 3)


cluster=[[] for i in range(k)] #On redéfinie les clusters correctement en une liste d'intervention avec chaque intervention sous forme de dictionnaire
for i in range(len(numero_cluster)):
    for j in range(len(data_dict_intervention)):
        if data_dict_intervention[j].get('coord') == coord_interventions[i]:
            cluster[numero_cluster[i]].append(data_dict_intervention[i])
            break

for i in range(len(cluster)):
    print('Cluster n°',i+1,len(cluster[i]))
    s+=len(cluster[i])

### AFFICHAGE MEDITERRANEE AVEC LE TRAIT DE CÔTE ###   
    
print('\n'+'Représentation des ESM et des Ports sur une Carte'+'\n')
plt.show()
'''
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
         
#print(kmeans.predict(coord_port_med)) #Association d'un port à son cluster le plus proche
#print(nom_port_med)
'''

### Répartition des interventions sur les semaines ###

nb_semaines = 52
nb_jours_par_semaine = 4
nb_interventions = len(data_dict_intervention)
nb_interventions_par_semaine = nb_interventions / nb_semaines


def repartitionAlea(listInter):
    alea = [[] for i in range(nb_semaines)] #Renvoie une liste d'interventions par semaine
    listInterRemoved = listInter[:]
    for j in range(len(listInter)):
        for i in range(nb_semaines):
            listInter = listInterRemoved[:]
            for inter in listInter:
                if inter['beginning'] <= i*nb_jours_par_semaine and inter['end'] >= (i+1)*nb_jours_par_semaine - 1:
                    alea[i].append(inter)
                    listInterRemoved.remove(inter)
                    break
    return alea

alea = repartitionAlea(cluster[1])
"""
for l in alea:
    print(len(l), "  ", l)
""" 

### Calcul nombre d'interventions maximales possibles ###

nombre_interventions_potentielles_par_bateau = [0, 0, 0, 0, 0, 0]
nombre_interventions_seules_par_bateau = [0, 0, 0, 0, 0, 0]


for intervention in dict_intervention_type['3']:
    for j in range(len(intervention['listBoat'])):
        nombre_interventions_potentielles_par_bateau[j]+=intervention['listBoat'][j]
    if sum(intervention['listBoat'])==1:
        k = 0
        while intervention['listBoat'][k]!=1:
            k+=1
        nombre_interventions_seules_par_bateau[k]+=1
#print(nombre_interventions_potentielles_par_bateau)
#print(nombre_interventions_seules_par_bateau) 

### Focus sur le navire n°5 Iles Lavezzi ###

interventions_lavezzi=[]
for intervention in dict_intervention_type['3']:
        if intervention['listBoat'] == [0, 0, 0, 0, 1, 0]:
            interventions_lavezzi.append(intervention)
#print(len(interventions_lavezzi))
            
'''
coord_intervention_lavezzi = []
for intervention in interventions_lavezzi:
    coord_intervention_lavezzi.append(np.array(intervention.get('coord')))
print(np.array(coord_intervention_lavezzi))
for k in range(len(coord_intervention_lavezzi)):
    plt.scatter(coord_intervention_lavezzi[k][0],coord_intervention_lavezzi[k][1]) #plot sur un carte
print(interventions_lavezzi[2])
print(interventions_lavezzi[8])
'''
### Visualisation d'un planning sous forme de bar chart ###

def visualiser_planning(liste_interventions):
    planning_interventions_beginning = [[],[],[]]
    planning_interventions_end = [[],[],[]]
    planning_interventions = [[],[],[]]
    planning_interventions_duree = [[],[],[]]
    compteur=0
    for intervention in liste_interventions: 
        compteur+=1
        t = intervention['type']-1
        planning_interventions_beginning[t].append(intervention.get('beginning'))
        planning_interventions_end[t].append(intervention.get('end'))
        planning_interventions_duree[t].append(planning_interventions_end[t][-1]-planning_interventions_beginning[t][-1])
        planning_interventions[t].append(compteur)
    print("Planning d'une liste d'intervention")
    
    plt.bar(x=planning_interventions[0], height=planning_interventions_duree[0], bottom=planning_interventions_beginning[0], align='edge', color = 'g')
    plt.bar(x=planning_interventions[1], height=planning_interventions_duree[1], bottom=planning_interventions_beginning[1], align='edge', color = 'b')
    plt.bar(x=planning_interventions[2], height=planning_interventions_duree[2], bottom=planning_interventions_beginning[2], align='edge', color = 'r')

'''
print(interventions_lavezzi[0])

for intervention in data_dict_intervention:
    if intervention['nbJoursInt']>1 and intervention['reste']==False:
        print('Oui')
'''

def echange(i,j,liste):
    liste[i], liste[j] = liste[j], liste[i]

def tri_intervention(liste_interventions):
    for i in range(len(liste_interventions)):
        for j in range(i,len(liste_interventions)):
            if liste_interventions[j]['beginning']<liste_interventions[i]['beginning']:
                echange(i,j,liste_interventions)
            elif liste_interventions[j]['beginning']==liste_interventions[i]['beginning'] and liste_interventions[j]['end']<liste_interventions[i]['end']:
                echange(i,j,liste_interventions)