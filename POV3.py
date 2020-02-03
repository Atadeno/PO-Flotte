import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap


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
    list_intervention_3.append(dict_intervention_type['3'][i].get('coord'))
      
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

def visualiser_esm(liste_coord_intervention, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(liste_coord_intervention))
    numero_cluster=kmeans.predict(liste_coord_intervention) #On associe à chaque intervention un numéro de cluster
    centres = kmeans.cluster_centers_
    centres = centres.transpose()
    
    cluster_coord = [[] for i in range(k)]
    for i in range(len(numero_cluster)):
        cluster_coord[numero_cluster[i]].append(liste_coord_intervention[i])
    
    plt.figure()
    #colors=['b','g','orange','c','m','y']
    colors=list(mcolors.TABLEAU_COLORS.values()) #OU CSS4 OU BASE
    cluster_plot=[]
    #plt.scatter(centres[0], centres[1],c = 'k', marker = 'o', linewidth = 1)
    for i in range(len(coord_port_med)):
        plt.scatter(coord_port_med[i][0], coord_port_med[i][1],c = 'r', marker = 's', linewidth = 3)
    for i in range(k):
        cluster_plot.append(np.array(cluster_coord[i]).transpose())
        plt.scatter(cluster_plot[i][0], cluster_plot[i][1],c = colors[i%len(colors)], marker = 'd', linewidth = 0.5)
    print('\n'+'Représentation des ESM et des Ports sur une Carte'+'\n')
    '''
    m = Basemap(llcrnrlon=2.8,llcrnrlat=41,urcrnrlon=10,urcrnrlat=44,resolution='l', lat_0 = 42.5, lon_0 = 6.5)
    #draw coastlines
    m.drawcoastlines()
    #m.drawmapboundary(fill_color='aqua')
    #m.fillcontinents(lake_color='aqua')
    '''
    plt.show()

kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(coord_interventions))
numero_cluster=kmeans.predict(coord_interventions) #On associe à chaque intervention un numéro de cluster
cluster=[[] for i in range(k)] #On redéfinie les clusters correctement en une liste d'intervention avec chaque intervention sous forme de dictionnaire
for i in range(len(coord_interventions)):
    for j in range(len(data_dict_intervention)): #On enlève les doublons
        if data_dict_intervention[j].get('coord') == coord_interventions[i] and not(data_dict_intervention[j] in cluster[numero_cluster[i]]):
            cluster[numero_cluster[i]].append(data_dict_intervention[j])
            break
'''
for i in range(len(cluster)):
    print('Cluster n°',i+1,len(cluster[i]))
    s+=len(cluster[i])
'''

#visualiser_esm(coord_interventions, k)

### AFFICHAGE MEDITERRANEE AVEC LE TRAIT DE CÔTE ###   

'''

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

"""
for l in alea:
    print(len(l), "  ", l)
""" 

### Calcul nombre d'interventions maximales possibles ###



def coord_intervention_type_t_bateau_b(b,t): #coord des interventions potentielles de type j du bateau i 
    coord = []
    for intervention in dict_intervention_type[str(t)]:
            if intervention['listBoat'][b]>0:
                coord.append(intervention['coord'])
    return coord
        
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
    print('\n'+"Planning d'une liste d'intervention")
    
    plt.bar(x=planning_interventions[0], height=planning_interventions_duree[0], bottom=planning_interventions_beginning[0], align='edge', color = 'g')
    plt.bar(x=planning_interventions[1], height=planning_interventions_duree[1], bottom=planning_interventions_beginning[1], align='edge', color = 'b')
    plt.bar(x=planning_interventions[2], height=planning_interventions_duree[2], bottom=planning_interventions_beginning[2], align='edge', color = 'r')
    plt.show()

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
'''                
visualiser_planning(cluster[4])             
tri_intervention(cluster[4])
visualiser_planning(cluster[4])
'''
def addition(liste1, liste2):
    for i in range(len(liste1)):
        liste1[i]+=liste2[i]

def ecriture_repartition(liste_clusters):
    #f = open("repartition.txt", "w")
    alea = repartitionAlea(liste_clusters[0])
    for i in range(1,len(liste_clusters)):
        addition(alea,repartitionAlea(liste_clusters[i]))
    #f.write(str(alea))
    #f.close()
    return alea

repartition = ecriture_repartition(cluster)

#On enlève les interventions impossibles à réaliser, ie qui ont un listBoat = [0, 0, 0, 0, 0, 0]

couple_impossible = []
impossible=[]

for i in range(len(repartition)):
    for j in range(len(repartition[i])):
        if repartition[i][j]['listBoat']==[0 for k in range(6)]:
            couple_impossible.append((i,j))
            impossible.append(repartition[i][j])
            
for couple in couple_impossible:
    del(repartition[couple[0]][couple[1]])
    
### Heuristique Sans Campagne ###

### Lecture des matrices de distances ###

matrice_bp = pd.read_csv('..\data\matriceDistance\DistanceMedBP.csv').to_dict()
matrice_bb = pd.read_csv('..\data\matriceDistance\DistanceMedBB.csv').to_dict()
matrice_bp20nm = pd.read_csv('..\data\matriceDistance\DistanceMedBP20NM.csv').to_dict()
matrice_bb20nm = pd.read_csv('..\data\matriceDistance\DistanceMedBB20NM.csv').to_dict()

esm = list(matrice_bb.keys()) # Liste des esm ordonnées, on fera référence aux indices pour déterminer la bonne esm
esm20nm = list(matrice_bb20nm.keys()) # Liste des esm ordonnées, on fera référence aux indices pour déterminer la bonne esm

def find_neareast_boat_available(intervention):
    bateau = 6
    distance = 10000.0
    for i in [0,1,2,3,4,5]:
        if intervention['listBoat'][i]:
            if (0 < matrice_bp[list_boat_med[i]['Port']][esm.index(intervention['esm'])]) and (matrice_bp[list_boat_med[i]['Port']][esm.index(intervention['esm'])] < distance): #Le <= permet de donner des interventions faisable au bateau 5 puisqu'il est dans le même port que le 1
                bateau = i
                distance = matrice_bp[list_boat_med[i]['Port']][esm.index(intervention['esm'])]
    direct = (intervention['time']+2*matrice_bp[list_boat_med[bateau]['Port']][esm.index(intervention['esm'])]/list_boat_med[bateau]['Vitesse'])<=14
    return bateau, direct

def reassigner_intervention(intervention):
    bateau = 6
    distance = 10000.0
    for i in [0,1]:
        if intervention['listBoat'][i]:
            if (0 < matrice_bp[list_boat_med[i]['Port']][esm.index(intervention['esm'])]) and (matrice_bp[list_boat_med[i]['Port']][esm.index(intervention['esm'])] < distance): #Le <= permet de donner des interventions faisable au bateau 5 puisqu'il est dans le même port que le 1
                bateau = i
                distance = matrice_bp[list_boat_med[i]['Port']][esm.index(intervention['esm'])]
    return bateau, bateau!=6

intervention_par_semaine = []
impossible_2 = []
campagne = [[],[]]

for liste_interventions_semaine in repartition:
    intervention_par_bateau = [[] for i in range(6)]
    for intervention in liste_interventions_semaine:
        num = esm.index(intervention['esm']) # Référence pour obtenir la distance dans les matrices
        bateau, direct = find_neareast_boat_available(intervention)
        if not direct and bateau in [2,3,4,5]:
            bateau, success = reassigner_intervention(intervention)
            if success:
                campagne[bateau].append(intervention)
            else:
                impossible_2.append(intervention)
        elif not direct and bateau in [0,1]:
            campagne[bateau].append(intervention)
        else:
            intervention_par_bateau[bateau].append(intervention)
    intervention_par_semaine.append(intervention_par_bateau)
    
def taille(liste_interventions): #Taille des sous listes et la somme
    taille = []
    for i in range(len(liste_interventions)):
        taille.append(len(liste_interventions[i]))
    return taille, sum(taille)

def liste_interventions_to_liste_coord(liste_interventions):
    coord = []
    for intervention in liste_interventions:
        coord.append(intervention['coord'])
    return coord

def visualiser_semaine(intervention_semaine):
    for intervention_bateau in intervention_semaine:
        if len(intervention_bateau) != 0:
            visualiser_esm(liste_interventions_to_liste_coord(intervention_bateau),1)
            
def total_intervention(intervention_par_semaine):
    total = [0 for i in range(6)]
    for intervention_semaine in intervention_par_semaine:
        addition(total,taille(intervention_semaine)[0])
    return total

def test_distance(liste_interventions, bateau):
    temps = []
    for intervention in liste_interventions:
        temps.append((intervention['time']+2*matrice_bp[list_boat_med[bateau]['Port']][esm.index(intervention['esm'])]/list_boat_med[bateau]['Vitesse'],matrice_bp[list_boat_med[bateau]['Port']][esm.index(intervention['esm'])],matrice_bp20nm[list_boat_med[bateau]['Port']][esm20nm.index(intervention['esm'])]))
    return temps

def disponibilité(intervention_par_semaine, bateau, plot):
    somme = [0 for i in range(52)]
    for i in range(len(intervention_par_semaine)):
        somme[i]+=len(intervention_par_semaine[i][bateau])
    if plot:
        plt.bar(x=[j for j in range(52)], height=somme, align='edge', color = 'b')
        plt.show()
    else:
        return somme

def fusionner(liste1, liste2, bateau):
    if not liste1[0][0]['double']:
        liste3=liste1+liste2
        d =  matrice_bp[list_boat_med[bateau]['Port']][esm.index(liste3[0][0]['esm'])]
        for i in range(len(liste3)-1):
            d+=matrice_bb[liste3[i][0]['esm']][esm.index(liste3[i+1][0]['esm'])]
        d+=matrice_bp[list_boat_med[bateau]['Port']][esm.index(liste3[-1][0]['esm'])]
        temps = 0
        for couple_intervention_index in liste3:
            temps+=couple_intervention_index[0]['time']
        temps+=d/list_boat_med[bateau]['Vitesse']
        return liste3, temps
    else:
        liste3=liste1+liste2
        d =  2*matrice_bp[list_boat_med[bateau]['Port']][esm.index(liste3[0][0]['esm'])]
        for i in range(len(liste3)-1):
            d+=matrice_bb[liste3[i][0]['esm']][esm.index(liste3[i+1][0]['esm'])]
        d+=matrice_bb[liste3[-1][0]['esm']][esm.index(liste3[0][0]['esm'])]
        temps = liste3[0][0]['time']
        for couple_intervention_index in liste3:
            temps+=couple_intervention_index[0]['time']
        temps+=d/list_boat_med[bateau]['Vitesse']
        return liste3, temps
        
def affichage_tournée(liste_interventions, jours):
    for tournée in jours:
        l=[]
        for intervention in tournée:
            l.append(liste_interventions.index(intervention))
        print(l)  
    
def saving_VRP(L, bateau):
    if (len(L)>0):
        (d,f) = nombre_double_fixe(L)
        liste_interventions = L.copy()
        new_liste = []
        for intervention in liste_interventions:
            new_liste.append((intervention,liste_interventions.index(intervention)))
        if (d,f)!=(0,0):
            #print(d,f)
            n = len(liste_interventions)
            for intervention in liste_interventions:
                if intervention['double'] or intervention['reste']:
                    for k in range(1,intervention['nbJoursInt']):
                        new_liste.append((intervention,n))
                        n+=1
        liste_interventions=new_liste
        #print(new_liste)
        n = len(liste_interventions)
        #print(n)
        jours = [[liste_interventions[i]] for i in range(n)]    
        cost = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                cost[i][j]=matrice_bb[liste_interventions[i][0]['esm']][esm.index(liste_interventions[j][0]['esm'])]
        saving = np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                if i!=j:
                    saving[i][j]=matrice_bp[list_boat_med[bateau]['Port']][esm.index(liste_interventions[i][0]['esm'])]+matrice_bp[list_boat_med[bateau]['Port']][esm.index(liste_interventions[j][0]['esm'])]-cost[i][j]
                    if saving[i][j]<0:
                        saving[i][j]=0
        maxi = np.max(saving)
        couple_saving_desc=[]
        while(maxi!=0):
            sortir = False
            for i in range(n):
                if sortir:
                    break
                for j in range(n):
                    if saving[i][j]==maxi:
                        couple_saving_desc.append((i,j))
                        saving[i][j]=0
                        sortir = True
                        maxi=np.max(saving)
                    elif sortir:
                        break
        #print(couple_saving_desc)
        for couple in couple_saving_desc:
            #print(couple)
            #affichage_tournée(liste_interventions, jours)
            a,b = couple
            if liste_interventions[a][0]!=liste_interventions[b][0]:
                if liste_interventions[b][0]['double']:
                    a,b=b,a
                if liste_interventions[a] not in jours[b] and liste_interventions[b] not in jours[a]:
                    if (liste_interventions[a] == jours[a][-1]) and (liste_interventions[b] == jours[b][0]):
                        liste_fusion, temps = fusionner(jours[a],jours[b],bateau)
                        #print(temps)
                        #print('\n')
                        if temps<=14:
                            for couple_intervention_index in liste_fusion:
                                jours[couple_intervention_index[1]]=liste_fusion
        jours_sans_doublon = [] #Les tournées fusionnées sont en doubles voire plus
        for liste_intervention_index in jours:
            if liste_intervention_index not in jours_sans_doublon:
                jours_sans_doublon.append(liste_intervention_index)
        #affichage_tournée(liste_interventions,jours_sans_doublon) #Affichage des tournées en plus compréhensible
        jours = []
        for liste_couple in jours_sans_doublon:
            l=[]
            for couple in liste_couple:
                l.append(couple[0])
            jours.append(l)
        #affichage_tournée(L,jours)  
        return jours
    else:
        return []

def test_tournée(tournée,bateau):
    if len(tournée)>0:
        if not tournée[0]['double']:
            d =  matrice_bp[list_boat_med[bateau]['Port']][esm.index(tournée[0]['esm'])]
            for i in range(len(tournée)-1):
                d+=matrice_bb[tournée[i]['esm']][esm.index(tournée[i+1]['esm'])]
            d+=matrice_bp[list_boat_med[bateau]['Port']][esm.index(tournée[-1]['esm'])]
            temps = 0
            for intervention in tournée:
                temps+=intervention['time']
            temps+=d/list_boat_med[bateau]['Vitesse']
            return temps
        else:
            d =  2*matrice_bp[list_boat_med[bateau]['Port']][esm.index(tournée[0]['esm'])]
            for i in range(len(tournée)-1):
                d+=matrice_bb[tournée[i]['esm']][esm.index(tournée[i+1]['esm'])]
            d+=matrice_bb[tournée[-1]['esm']][esm.index(tournée[0]['esm'])]
            temps = tournée[0]['time']
            for intervention in tournée:
                temps+=intervention['time']
            temps+=d/list_boat_med[bateau]['Vitesse']
            return temps
    else:
        return 0
    
def nombre_double_fixe(intervention_semaine): #Estimation des problèmes double/fixe
    d = 0
    f = 0
    for intervention in intervention_semaine:
        if intervention['double']:
            d+=1
            #print(intervention)
        if intervention['reste']:
            f+=1
            #print('fixe',intervention['nbJoursInt'])
    return (d,f)

def replannifier(intervention, semaine, bateau, intervention_par_semaine):
    if intervention['end']>4*(semaine+1):
        index = intervention_par_semaine[semaine][bateau].index(intervention)
        del(intervention_par_semaine[semaine][bateau][index])
        intervention_par_semaine[semaine+1][bateau].append(intervention)
    else:
        print("Revoir la replannification")
    
deplacement = True
while(deplacement):
    deplacement = False
    for intervention_semaine in intervention_par_semaine:
       if deplacement:
              break
       semaine = intervention_par_semaine.index(intervention_semaine)
       for intervention_bateau in intervention_semaine:
           jours = saving_VRP(intervention_bateau,intervention_semaine.index(intervention_bateau))
           if len(jours)>4:
               deplacement = True
               replannifier(jours[-1][0],semaine,intervention_semaine.index(intervention_bateau),intervention_par_semaine)
               break


intervention_par_jours = [[] for i in range(6)]
for s in range(52):
    for b in range(6):
        jours = saving_VRP(intervention_par_semaine[s][b],b)
        while(len(jours)<4):
            jours.append([])
        for k in range(4):
            intervention_par_jours[b].append(jours[k])
        
def disponibilité_jours(intervention_par_jours, bateau):
    somme = [0 for i in range(208)]
    for i in range(len(intervention_par_jours[bateau])):
        somme[i]+=test_tournée(intervention_par_jours[bateau][i],bateau)
    plt.bar(x=[j for j in range(208)], height=somme, align='edge', color = 'r')
    plt.show()

'''
Iles Sanguinaires II AJACCIO
Provence MARSEILLE
Saint Clair 7 SETE
L'Esquillade TOULON
Iles Lavezzi BONIFACIO
Arnette MARSEILLE
'''
def changer_intervention(campagne):
   index = campagne[0].index({'beginning': 96.0,
  'end': 201.0,
  'type': 2,
  'nbJoursInt': 1,
  'nomEsm': "COTE D'AZUR - BOUEE METEO FRANCE - ODAS",
  'esm': '1300350',
  'typeEsm': 'BouÃ©e',
  'coord': [7.828333333333333, 43.38166666666667],
  'time': 2.5,
  'listBoat': [1, 1, 0, 0, 0, 0],
  'retrieve': False,
  'double': False,
  'reste': False})
   campagne[1].append({'beginning': 96.0,
  'end': 201.0,
  'type': 2,
  'nbJoursInt': 1,
  'nomEsm': "COTE D'AZUR - BOUEE METEO FRANCE - ODAS",
  'esm': '1300350',
  'typeEsm': 'BouÃ©e',
  'coord': [7.828333333333333, 43.38166666666667],
  'time': 2.5,
  'listBoat': [1, 1, 0, 0, 0, 0],
  'retrieve': False,
  'double': False,
  'reste': False})
   del(campagne[0][index])
    
def ecriture_campagne(campagne):
    f = open("campagne.txt", "w")
    for intervention_bateau in campagne:
        f.write(str(intervention_bateau))
    f.close()
