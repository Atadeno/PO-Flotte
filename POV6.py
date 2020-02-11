import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap
import statistics

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
#print('Liste des ports de la côte méditerranéenne: ',nom_port_med,'\n')

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
    
    ### AFFICHAGE MEDITERRANEE AVEC LE TRAIT DE CÔTE ###  
    
    m = Basemap(llcrnrlon=2.8,llcrnrlat=41,urcrnrlon=10,urcrnrlat=44,resolution='l', lat_0 = 42.5, lon_0 = 6.5)
    #draw coastlines
    m.drawcoastlines()
    #m.drawmapboundary(fill_color='aqua')
    #m.fillcontinents(lake_color='aqua')
    plt.show()

kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(coord_interventions))
numero_cluster=kmeans.predict(coord_interventions) #On associe à chaque intervention un numéro de cluster
cluster=[[] for i in range(k)] #On redéfinie les clusters correctement en une liste d'intervention avec chaque intervention sous forme de dictionnaire
for i in range(len(coord_interventions)):
    for j in range(len(data_dict_intervention)): #On enlève les doublons
        if data_dict_intervention[j].get('coord') == coord_interventions[i] and not(data_dict_intervention[j] in cluster[numero_cluster[i]]):
            cluster[numero_cluster[i]].append(data_dict_intervention[j])
            break

#taille(cluster) #Donne la taille totale de cluster avec la réparition en nombre d'interventions par semaines

#visualiser_esm(coord_interventions, k) #Affiche la répartition totale des interventions

'''
Hyp 1: On a 548 interventions uniques pour un total de jours d'intervention cumulés de 640
'''

#print(kmeans.predict(coord_port_med)) #Association d'un port à son cluster le plus proche
#print(nom_port_med)

### Répartition des interventions sur les semaines ###
'''
def repartitionAlea(listInter):
    alea = [[] for i in range(52)] #Renvoie une liste d'interventions par semaine
    listInterRemoved = listInter[:]
    for j in range(len(listInter)):
        for i in range(52):
            listInter = listInterRemoved[:]
            for inter in listInter:
                if inter['beginning'] <= i*4 and inter['end'] >= (i+1)*4:
                    alea[i].append(inter)
                    listInterRemoved.remove(inter)
                    break
                else:
                    print(i*4)
                    print((i+1)*4)
                    print(inter)
    return alea
'''

def repartitionBeginning(liste_interventions):
    semaines = [[] for i in range(52)] #Renvoie une liste d'interventions par semaine
    for intervention in liste_interventions:
        beg = intervention['beginning']
        i = 0
        while i < beg:
            i+=4
        semaines[int(i/4)].append(intervention)
    return semaines

### Calcul nombre d'interventions maximales possibles ###
'''
def coord_intervention_type_t_bateau_b(b,t): #coord des interventions potentielles de type t du bateau b
    coord = []
    for intervention in dict_intervention_type[str(t)]:
            if intervention['listBoat'][b]>0:
                coord.append(intervention['coord'])
    return coord
'''     
def nombre_interventions_potentielles_et_seules_par_bateau(type_intervention):
    nombre_interventions_potentielles_par_bateau = [0, 0, 0, 0, 0, 0]
    nombre_interventions_seules_par_bateau = [0, 0, 0, 0, 0, 0]  
    for intervention in dict_intervention_type[str(type_intervention)]:
        for j in range(len(intervention['listBoat'])):
            nombre_interventions_potentielles_par_bateau[j]+=intervention['listBoat'][j]
        if sum(intervention['listBoat'])==1:
            k = 0
            while intervention['listBoat'][k]!=1:
                k+=1
            nombre_interventions_seules_par_bateau[k]+=1
    return nombre_interventions_potentielles_par_bateau, nombre_interventions_seules_par_bateau

#potentielles, seules = nombre_interventions_potentielles_et_seules_par_bateau(3)
#print(potentielles) #Le nombre dinterventions potentielles de type 3 que peuvent faire les bateaux
#print(seules) #Le nombre dinterventions de type 3 que seuls peuvent faire les bateaux

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
            
#visualiser_planning(cluster[4])             
#tri_intervention(cluster[4])
#visualiser_planning(cluster[4])
    
def taille(liste_interventions): #Taille des sous listes et la somme
    taille = []
    for i in range(len(liste_interventions)):
        taille.append(len(liste_interventions[i]))
    return taille, sum(taille)

def echange(i,j,liste):
    liste[i], liste[j] = liste[j], liste[i]

def tri_intervention(liste_interventions):
    for i in range(len(liste_interventions)):
        for j in range(i,len(liste_interventions)):
            if liste_interventions[j]['beginning']<liste_interventions[i]['beginning']:
                echange(i,j,liste_interventions)
            elif liste_interventions[j]['beginning']==liste_interventions[i]['beginning'] and liste_interventions[j]['end']<liste_interventions[i]['end']:
                echange(i,j,liste_interventions)

def tri_intervention_end(liste_interventions):
    for i in range(len(liste_interventions)):
        for j in range(i,len(liste_interventions)):
            if liste_interventions[j]['end']<liste_interventions[i]['end']:
                echange(i,j,liste_interventions)
            elif liste_interventions[j]['end']==liste_interventions[i]['end'] and liste_interventions[j]['type']>liste_interventions[i]['type']:
                echange(i,j,liste_interventions)

def tri_intervention_couple_end(liste_couple_interventions):
    for i in range(len(liste_couple_interventions)):
        for j in range(i,len(liste_couple_interventions)):
            if liste_couple_interventions[j][0]['end']<liste_couple_interventions[i][0]['end']:
                echange(i,j,liste_couple_interventions)
            elif liste_couple_interventions[j][0]['end']==liste_couple_interventions[i][0]['end'] and liste_couple_interventions[j][0]['type']>liste_couple_interventions[i][0]['type']:
                echange(i,j,liste_couple_interventions)
                
def addition(liste1, liste2):
    for i in range(len(liste1)):
        liste1[i]+=liste2[i]

def ecriture_repartition(liste_clusters):
    #f = open("repartition.txt", "w")
    semaines = repartitionBeginning(liste_clusters[0])
    for i in range(1,len(liste_clusters)):
        addition(semaines,repartitionBeginning(liste_clusters[i]))
    #f.write(str(alea))
    #f.close()
    return semaines

repartition = ecriture_repartition(cluster)

#On enlève les interventions impossibles à réaliser, ie qui ont un listBoat = [0, 0, 0, 0, 0, 0]

repartition_possible = [[] for i in range(52)]
impossible = []

for i in range(len(repartition)):
    for j in range(len(repartition[i])):
        if repartition[i][j]['listBoat']!=[0 for k in range(6)]:
            repartition_possible[i].append(repartition[i][j])
        else:
            impossible.append(repartition[i][j])
            
repartition = repartition_possible
#print(taille(repartition))

for i in range(len(repartition)):
    for j in range(len(repartition[i])):
        if repartition[i][j]['listBoat']==[0 for k in range(6)]:
           print('ERREURE')

def min_end(liste_interventions):
    if len(liste_interventions)>0:
        end = []
        for intervention in liste_interventions:
            end.append(intervention['end'])
        return int(min(end))
    else:
        return 0

def max_beg(liste_interventions):
    beg = []
    for intervention in liste_interventions:
        beg.append(intervention['beginning'])
    return int(max(beg))

def min_beg(liste_interventions):
    beg = []
    for intervention in liste_interventions:
        beg.append(intervention['beginning'])
    return int(min(beg))

def min_end_couple(liste_interventions):
    if len(liste_interventions)>0:
        end = []
        for intervention in liste_interventions:
            end.append(intervention[0]['end'])
        return int(min(end))
    else:
        return 0

def max_beg_couple(liste_interventions):
    beg = []
    for intervention in liste_interventions:
        beg.append(intervention[0]['beginning'])
    return int(max(beg))
### Heuristique Sans Campagne ###

### Lecture des matrices de distances ###

matrice_bp = pd.read_csv('..\data\matriceDistance\DistanceMedBP.csv').to_dict()
matrice_bb = pd.read_csv('..\data\matriceDistance\DistanceMedBB.csv').to_dict()
matrice_bp20nm = pd.read_csv('..\data\matriceDistance\DistanceMedBP20NM.csv').to_dict()
matrice_bb20nm = pd.read_csv('..\data\matriceDistance\DistanceMedBB20NM.csv').to_dict()
matrice_pp = pd.read_csv('..\data\matriceDistance\DistanceMedPP.csv').to_dict()

esm = list(matrice_bb.keys()) # Liste des esm ordonnées, on fera référence aux indices pour déterminer la bonne esm
esm20nm = list(matrice_bb20nm.keys()) # Liste des esm ordonnées, on fera référence aux indices pour déterminer la bonne esm

def port_accessible(bateau): 
    list_port = list(list_boat_med[bateau]['Escale'].keys())
    accessible = list(list_boat_med[bateau]['Escale'].values())
    list_port_accessible = [] #Liste des ports qui peuvent accueillir le bateau en campagne
    for i in range(len(list_port)):
        if accessible[i]:
            list_port_accessible.append(list_port[i])
    return list_port_accessible

def find_neareast_boat_available(intervention):
    bateau = 6
    distance = 10000.0
    for i in [0,1,2,3,4,5]:
        if intervention['listBoat'][i]:
            if (0 < matrice_bp[list_boat_med[i]['Port']][esm.index(intervention['esm'])]) and (matrice_bp[list_boat_med[i]['Port']][esm.index(intervention['esm'])] <= distance): #Le <= permet de donner des interventions faisable au bateau 5 puisqu'il est dans le même port que le 1
                bateau = i
                distance = matrice_bp[list_boat_med[i]['Port']][esm.index(intervention['esm'])]
    direct = (intervention['time']+2*matrice_bp[list_boat_med[bateau]['Port']][esm.index(intervention['esm'])]/list_boat_med[bateau]['Vitesse'])<=14
    return bateau, direct

def reassigner_intervention(intervention): #Si l'esm est inatteignable par le bateau le plus proche en moins de 14h, on l'assigne à une campagne
    bateau = 6
    distance = 10000.0
    for i in [0,1]:
        if intervention['listBoat'][i]:
            for port in port_accessible(i): #Assigner l'esm au port accessible le plus proche (campagne)
                if (0 < matrice_bp[port][esm.index(intervention['esm'])]) and (matrice_bp[port][esm.index(intervention['esm'])] < distance):
                    bateau = i
                    distance = matrice_bp[port][esm.index(intervention['esm'])]
    return bateau, bateau!=6

intervention_par_semaine = []
impossible_2 = []
campagne = [[],[]]

for liste_interventions_semaine in repartition:
    intervention_par_bateau = [[] for i in range(6)]
    for intervention in liste_interventions_semaine:
        bateau, direct = find_neareast_boat_available(intervention)
        if not direct and bateau in [2,3,4,5]: #Si l'intervention n'est pas directe, on essaye de l'assigner à une campagne
            bateau, success = reassigner_intervention(intervention)
            if success:
                campagne[bateau].append(intervention) #On l'assigne à une campagne
            else:
                impossible_2.append(intervention) #L'intervention n'est pas faisable par les bateaux habitables
        elif not direct and bateau in [0,1]:
            campagne[bateau].append(intervention)
        else:
            intervention_par_bateau[bateau].append(intervention) #Le bateau peut faire l'intervention directement
    intervention_par_semaine.append(intervention_par_bateau)

def liste_interventions_to_liste_coord(liste_interventions): #Liste d'interventions en liste de coordonnées
    coord = []
    for intervention in liste_interventions:
        coord.append(intervention['coord'])
    return coord

def visualiser_semaine(intervention_semaine): #Affiche une semaine potentielle avec un plot par bateau pour 
    for intervention_bateau in intervention_semaine:
        if len(intervention_bateau) != 0:
            visualiser_esm(liste_interventions_to_liste_coord(intervention_bateau),1)
            
def total_intervention(intervention_par_semaine):
    total = [0 for i in range(6)]
    for intervention_semaine in intervention_par_semaine:
        addition(total,taille(intervention_semaine)[0])
    return total

#print(sum(total_intervention(intervention_par_semaine)))
'''
s=0
for intervention_bateau in intervention_par_semaine:
    for interventions in intervention_bateau:
        for intervention in interventions:
            s+=intervention['nbJoursInt']
print(s)
'''
def test_distance(liste_interventions, bateau):
    temps = []
    for intervention in liste_interventions:
        temps.append((intervention['time']+2*matrice_bp[list_boat_med[bateau]['Port']][esm.index(intervention['esm'])]/list_boat_med[bateau]['Vitesse'],matrice_bp[list_boat_med[bateau]['Port']][esm.index(intervention['esm'])],matrice_bp20nm[list_boat_med[bateau]['Port']][esm20nm.index(intervention['esm'])]))
    return temps

def disponibilité(intervention_par_semaine, bateau, plot): #Affiche le nombre d'intervention par jours du bateau, ou retourne le total
    somme = [0 for i in range(52)]
    for i in range(len(intervention_par_semaine)):
        somme[i]+=len(intervention_par_semaine[i][bateau])
    if plot:
        plt.bar(x=[j for j in range(52)], height=somme, align='edge', color = 'b')
        plt.show()
    else:
        return somme

#disponibilité(intervention_par_semaine, 2, True)

def fusionner(liste1, liste2, port, bateau): #Fusion des tournées avant distinction si la première intervention est double
    if not liste1[0][0]['double']:
        liste3=liste1+liste2
        d =  matrice_bp[port][esm.index(liste3[0][0]['esm'])]
        for i in range(len(liste3)-1):
            d+=matrice_bb[liste3[i][0]['esm']][esm.index(liste3[i+1][0]['esm'])]
        d+=matrice_bp[port][esm.index(liste3[-1][0]['esm'])]
        temps = 0
        for couple_intervention_index in liste3:
            temps+=couple_intervention_index[0]['time']
        temps+=d/list_boat_med[bateau]['Vitesse']
        return liste3, temps
    else:
        liste3=liste1+liste2
        d =  2*matrice_bp[port][esm.index(liste3[0][0]['esm'])]
        for i in range(len(liste3)-1):
            d+=matrice_bb[liste3[i][0]['esm']][esm.index(liste3[i+1][0]['esm'])]
        d+=matrice_bb[liste3[-1][0]['esm']][esm.index(liste3[0][0]['esm'])]
        temps = liste3[0][0]['time']
        for couple_intervention_index in liste3:
            temps+=couple_intervention_index[0]['time']
        temps+=d/list_boat_med[bateau]['Vitesse']
        return liste3, temps
        
def affichage_tournée(liste_interventions, jours): #Affiche les tournées avec le numéro des sommets
    for tournée in jours:
        l=[]
        for intervention in tournée:
            l.append(liste_interventions.index(intervention))
        print(l)  
    
def intervention_presente(intervention, liste_couple_intervention_index):
    for couple in liste_couple_intervention_index:
        if intervention == couple[0]:
            return True
    return False

def liste_presente(liste1, liste2):
    for couple in liste1:
        if intervention_presente(couple[0],liste2):
            return True
    return False

def saving_VRP(L, port, bateau, duree): #Algorithme de Clark and Wright, autrement appelé méthode des économies
    if (len(L)>0):
        (d,f,r) = nombre_double_fixe_retrieve(L)
        liste_interventions = L.copy()
        new_liste = []
        for intervention in liste_interventions:
            new_liste.append((intervention,liste_interventions.index(intervention)))
        if (d,f)!=(0,0): #S'il y a une intervention double ou fixe, avec un nombre de jours d'intervention supérieur à 1, on créé des copies et on donne des indexs à toutes les interventions
            n = len(liste_interventions)
            for intervention in liste_interventions:
                if intervention['double'] or intervention['reste']:
                    for k in range(1,intervention['nbJoursInt']):
                        new_liste.append((intervention,n)) #Les interventions avec nbJourInt>1 sont copiées mais ont un index différent
                        n+=1
        #tri_intervention_couple_end(new_liste)
        liste_interventions=new_liste #On passe d'une liste d'interventions à une liste de couple (intervention, index). La queue de la liste étant les copies des interventions doubles/fixes.
        n = len(liste_interventions)
        jours = [[liste_interventions[i]] for i in range(n)] #Création de n tournée qui ne visite qu'une seule esm  
        cost = np.zeros((n,n)) #Mise en mémoire de la matrice des coûts
        for i in range(n):
            for j in range(n):
                cost[i][j]=matrice_bb[liste_interventions[i][0]['esm']][esm.index(liste_interventions[j][0]['esm'])]
        saving = np.zeros((n,n)) #Calcul de la matrice des économies
        for i in range(n):
            for j in range(n):
                if i!=j:
                    saving[i][j]=matrice_bp[port][esm.index(liste_interventions[i][0]['esm'])]+matrice_bp[port][esm.index(liste_interventions[j][0]['esm'])]-cost[i][j]
                    if saving[i][j]<0: #Si jamais la distance est négative (impossibilité) ou non euclidien?
                        saving[i][j]=0
        maxi = np.max(saving)
        couple_saving_desc=[]
        while(maxi!=0):
            sortir = False
            for i in range(n):
                if sortir:
                    break
                for j in range(n):
                    if saving[i][j]==maxi and maxi!=0:
                        couple_saving_desc.append((i,j))
                        saving[i][j]=0
                        sortir = True
                        maxi=np.max(saving)
                    elif sortir:
                        break
        #print(couple_saving_desc)
        #affichage_tournée(liste_interventions, jours)
        for couple in couple_saving_desc:
            #print(couple)
            #affichage_tournée(liste_interventions, jours)
            a,b = couple
            if liste_interventions[a][0]!=liste_interventions[b][0]:
                if liste_interventions[b][0]['double']:
                    a,b=b,a
                if not liste_presente(jours[a],jours[b]) and not intervention_presente(jours[b],jours[a]):
                    if (liste_interventions[a] == jours[a][-1]) and (liste_interventions[b] == jours[b][0]):
                        liste_fusion, temps = fusionner(jours[a],jours[b],port,bateau)
                        if max_beg_couple(liste_fusion)<=min_end_couple(liste_fusion):
                            #print(temps)
                            #print('\n')
                            if temps<=duree:
                                for couple_intervention_index in liste_fusion:
                                    jours[couple_intervention_index[1]]=liste_fusion
                                #affichage_tournée(liste_interventions, jours)
                                #print('\n')
        jours_sans_doublon = [] #Les tournées fusionnées sont en doubles voire plus
        for liste_intervention_index in jours:
            if liste_intervention_index not in jours_sans_doublon:
                jours_sans_doublon.append(liste_intervention_index)
        #affichage_tournée(liste_interventions,jours_sans_doublon) #Affichage des tournées en plus compréhensible
        jours = []
        for liste_couple in jours_sans_doublon:
            l = []
            for couple in liste_couple:
                l.append(couple[0])
            jours.append(l)
        #affichage_tournée(L,jours)  
        return jours
    else:
        return []

def test_tournée(tournée,port,bateau):
    if len(tournée)>0:
        if not tournée[0]['double']:
            d =  matrice_bp[port][esm.index(tournée[0]['esm'])]
            for i in range(len(tournée)-1):
                d+=matrice_bb[tournée[i]['esm']][esm.index(tournée[i+1]['esm'])]
            d+=matrice_bp[port][esm.index(tournée[-1]['esm'])]
            temps = 0
            for intervention in tournée:
                temps+=intervention['time']
            temps+=d/list_boat_med[bateau]['Vitesse']
            return temps
        else:
            d =  2*matrice_bp[port][esm.index(tournée[0]['esm'])]
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
    
def nombre_double_fixe_retrieve(intervention_semaine): #Estimation des problèmes double/fixe
    if(len(intervention_semaine))>0:
        d = 0
        f = 0
        r = 0
        for intervention in intervention_semaine:
            if intervention['double']:
                d+=1
                #print(intervention)
            if intervention['reste']:
                f+=1
                #print('fixe',intervention['nbJoursInt'])
            if intervention['retrieve']:
                r+=1
        return (d,f,r)
    else:
        return (0,0,0)

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
           jours = saving_VRP(intervention_bateau,list_boat_med[intervention_semaine.index(intervention_bateau)]['Port'],intervention_semaine.index(intervention_bateau),14)
           if len(jours)>4:
               deplacement = True
               #print(semaine,intervention_semaine.index(intervention_bateau))
               replannifier(jours[-1][0],semaine,intervention_semaine.index(intervention_bateau),intervention_par_semaine)
               break


intervention_par_jours = [[] for i in range(6)]
for s in range(52):
    for b in range(6):
        jours = saving_VRP(intervention_par_semaine[s][b],list_boat_med[b]['Port'],b,14)
        while(len(jours)<4):
            jours.append([])
        for k in range(4):
            intervention_par_jours[b].append(jours[k])
       
def disponibilité_jours(intervention_par_jours, bateau):
    somme = [0 for i in range(208)]
    j = 0
    for i in range(len(intervention_par_jours[bateau])):
        somme[i]+=test_tournée(intervention_par_jours[bateau][i],list_boat_med[bateau]['Port'],bateau)
        if intervention_par_jours[bateau][i] != []:
            j+=1
    plt.bar(x=[j for j in range(208)], height=somme, align='edge', color = 'r')
    plt.show()
    return sum(somme)/j, j, statistics.mean(somme)

def consommation_bateau(intervention_par_jours):
    somme = [0 for i in range(6)]
    maxi = [0 for i in range(6)]
    diff = [0 for i in range(6)]
    for b in range(6):
        maxi[b] = list_boat_med[b]['Jours_Disponibles_Reels']
        s=0
        for j in range(208):
            if intervention_par_jours[b][j] != []:
                s+=1
        somme[b]=s
        diff[b]=maxi[b]-somme[b]
    plt.bar(x=[j for j in range(6)], height=somme, align='edge', color = 'r')
    plt.bar(x=[j for j in range(6)], height=diff, bottom=somme, align='edge', color = 'b')
    plt.show()
    return somme
                
def visualiser_tournée(tournée, port):
    l = liste_interventions_to_liste_coord(tournée)
    p = nom_port_med.index(port)
    l.append([coord_port_med[p][0], coord_port_med[p][1]])
    llcx = l[0][0]
    llcy = l[0][1]
    urcx = l[0][0]
    urcy = l[0][1]
    for i in range(len(l)):
        if l[i][0] < llcx:
            llcx = l[i][0]
        if l[i][0] > urcx:
            urcx =  l[i][0]
        if l[i][1] < llcy:
            llcy = l[i][1]
        if l[i][1] > urcy:
            urcy = l[i][1]
    llcx = llcx-0.5
    llcy = llcy-0.5
    urcx = urcx+0.5
    urcy = urcy+0.5
    for i in range(len(l)-1):
        plt.scatter(l[i][0], l[i][1],c = 'g',marker = 'x', linewidth = 3)
    plt.scatter(coord_port_med[p][0], coord_port_med[p][1],c = 'r', marker = 's', linewidth = 5)
    m = Basemap(llcrnrlon=llcx,llcrnrlat=llcy,urcrnrlon=urcx,urcrnrlat=urcy,resolution='i', lat_0 = urcy-(urcy-llcy), lon_0 = urcx-(urcx-llcx))
    m.drawcoastlines()
    plt.show()

def changer_intervention(campagne):
   i1={'beginning': 109.0,
  'end': 207,
  'type': 3,
  'nbJoursInt': 1,
  'nomEsm': "CALVI - BALISE A FLOTTEUR DU  DANGER D'ALGAJOLA",
  'esm': '2B00010',
  'typeEsm': 'Balise Ã\xa0 flotteur',
  'coord': [8.838983333333333, 42.628883333333334],
  'time': 4,
  'listBoat': [1, 1, 0, 0, 0, 0],
  'retrieve': True,
  'double': False,
  'reste': True}
   i2={'beginning': 96.0,
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
  'reste': False}
   index2 = campagne[0].index(i2)
   campagne[1].append(i2)
   del(campagne[0][index2])
   index1 = campagne[0].index(i1)
   campagne[1].append(i1)
   del(campagne[0][index1])

changer_intervention(campagne)

def port_escale(bateau): 
    list_port = list(list_boat_med[bateau]['Escale'].keys())
    accessible = list(list_boat_med[bateau]['Escale'].values())
    list_port_escale = [] #Liste des ports qui peuvent accueillir le bateau en campagne
    for i in range(len(list_port)):
        if accessible[i] and list_port[i]!=list_boat_med[bateau]['Port'] :
            list_port_escale.append(list_port[i])
    return list_port_escale

def find_neareast_port(intervention, list_port, bateau): 
    distance = 10000.0
    port = ''
    for p in list_port:
        if (0 < matrice_bp[p][esm.index(intervention['esm'])]) and (matrice_bp[p][esm.index(intervention['esm'])] <= distance):
            port = p
            distance = matrice_bp[port][esm.index(intervention['esm'])]
    direct = (intervention['time']+2*matrice_bp[port][esm.index(intervention['esm'])]/list_boat_med[bateau]['Vitesse'])<=14
    return port, direct, intervention['time']+2*matrice_bp[port][esm.index(intervention['esm'])]/list_boat_med[bateau]['Vitesse']

def gestion_campagne(campagne, bateau):
    impossible = [] #Liste des interventions accessibles qu'avec une tournée de 38h
    ports = port_escale(bateau)
    intervention_par_port_escale = [[] for i in range(len(ports))]
    for intervention in campagne:
        port, direct, distance = find_neareast_port(intervention, ports, bateau)
        if not direct:
            impossible.append(intervention)
             #print(port,distance)
        else:
            intervention_par_port_escale[ports.index(port)].append(intervention)
    return impossible, intervention_par_port_escale

def ecriture_campagne(campagne):
    f = open("campagne.txt", "w")
    for intervention_bateau in campagne:
        f.write(str(intervention_bateau))
    f.close()
    
impossible_3_0, intervention_par_port_escale_0 = gestion_campagne(campagne[0],0)
impossible_3_1, intervention_par_port_escale_1 = gestion_campagne(campagne[1],1)

def plage_jours(liste_interventions):
    beg = []
    end = []
    for intervention in liste_interventions:
        beg.append(intervention['beginning'])
        end.append(intervention['end'])
    if min(end)-max(beg)>0:
        return min(end)-max(beg),max(beg)
    else:
        return 0,0

        
def test_2_double(intervention_par_jours): #Vérification qu'il y a au maximum une intervention double par jours
    for intervention_bateau in intervention_par_jours:
        for intervention in intervention_bateau:
            if nombre_double_fixe_retrieve(intervention)[0]>1:
                print('Problème')
                
def test_2_retrieve(intervention_par_jours): #Vérification qu'il y a au maximum une intervention double par jours
    for intervention_bateau in intervention_par_jours:
        for interventions in intervention_bateau:
            if nombre_double_fixe_retrieve(interventions)[2]>1:
                print('Problème')
                print(nombre_double_fixe_retrieve(interventions))
                print(interventions)
                print(intervention_par_jours.index(intervention_bateau))
                print(intervention_bateau.index(interventions))

def jours_de_campagne(liste_interventions, jours_minimum): #Renvoie des sous-listes d'interventions et une liste de début de la campagne
    #On essaye de caser des campagnes en 1 semaine complète de travail
    tri_intervention(liste_interventions)
    campagnes = []
    start = []
    i = 0
    j = 1
    while(i<len(liste_interventions)):
        while(plage_jours(liste_interventions[i:j+1])[0]>jours_minimum and j<len(liste_interventions)):
            j+=1
        campagnes.append(liste_interventions[i:j])
        start.append(plage_jours(liste_interventions[i:j])[1])
        i=j
        j+=1
    return campagnes, start

#### Campagne du bateau 0 à BASTIA ###
#    
#camp, start = jours_de_campagne(intervention_par_port_escale_0[0],4)
#tournée_campagne = saving_VRP(camp[0],'BASTIA',0,14) #J'ai vérifié que cette période était vide
#intervention_par_jours[0][int(start[0])-1:int(start[0])+len(tournée_campagne)+1] = [[('AJACCIO','BASTIA')]]+tournée_campagne+[[('BASTIA','AJACCIO')]]
#
#### Campagne 1 du bateau 1 à SETE ###
#
#camp, start = jours_de_campagne(intervention_par_port_escale_1[0],4)
#tournée_campagne = saving_VRP(camp[0],'SETE',1,14)
#intervention_par_jours[1][int(start[0])-1:int(start[0])+len(tournée_campagne)+1] = [[('MARSEILLE','SETE')]]+tournée_campagne+[[('SETE','MARSEILLE')]]
#
#### Campagne 2 du bateau 1 à SETE ###
#
#camp, start = jours_de_campagne(intervention_par_port_escale_1[0],4)
#tournée_campagne = saving_VRP(camp[1],'SETE',1,14) #J'ai vérifié que cette période était vide
#intervention_par_jours[1][int(start[1])-1:int(start[1])+len(tournée_campagne)+1] = [[('MARSEILLE','SETE')]]+tournée_campagne+[[('SETE','MARSEILLE')]]
#
#### Campagne 1 du bateau 1 à TOULON ###
#
#camp, start = jours_de_campagne(intervention_par_port_escale_1[1],10)
#tournée_campagne = saving_VRP(camp[0],'TOULON',1,14)
#intervention_par_jours[1][int(start[0])+1:int(start[0])+len(tournée_campagne)+2] = [[('MARSEILLE','TOULON')]]+tournée_campagne+[[('TOULON','MARSEILLE')]]
#
#### Campagne 2 du bateau 1 à TOULON ###
#
#camp, start = jours_de_campagne(intervention_par_port_escale_1[1],10)
#tournée_campagne = saving_VRP(camp[1],'TOULON',1,14)
#intervention_par_jours[1][133], intervention_par_jours[1][136] = intervention_par_jours[1][136], intervention_par_jours[1][133]
#intervention_par_jours[1][int(start[1])-1:int(start[1])+len(tournée_campagne)+1] = [[('MARSEILLE','TOULON')]]+tournée_campagne+[[('TOULON','MARSEILLE')]]
#
#### Campagne 3 du bateau 1 à TOULON ###
#
#camp, start = jours_de_campagne(intervention_par_port_escale_1[1],10)
#tournée_campagne = saving_VRP(camp[2],'TOULON',1,14)
##print(intervention_par_jours[1][int(start[2])+4:int(start[2])+len(tournée_campagne)+6])
#intervention_par_jours[1][int(start[2])+4:int(start[2])+len(tournée_campagne)+6] = [[('MARSEILLE','TOULON')]]+tournée_campagne+[[('TOULON','MARSEILLE')]]
#
#### Tournée de 38h 1 bateau 0 en partant de AJACCIO ###
#
#camp, start = jours_de_campagne(impossible_3_0,4)
#tournée_campagne = saving_VRP(camp[0],'AJACCIO',0,38)
#intervention_par_jours[0][int(start[0]):int(start[0])+3] = [[('38h',tournée_campagne)]]+[[('38h',tournée_campagne)]]+[[('38h',tournée_campagne)]]
#
#### Tournée de 38h 2 bateau 0 en partant de AJACCIO ###
#
#camp, start = jours_de_campagne(impossible_3_0,4)
#tournée_campagne = saving_VRP(camp[1],'AJACCIO',0,38)
#intervention_par_jours[0][int(start[1]):int(start[1])+3] = [[('38h',tournée_campagne)]]+[[('38h',tournée_campagne)]]+[[('38h',tournée_campagne)]]

def check_interventions_global():
    data_dict_intervention_sans_doublons = []
    detection = []
    total = 0
    maxi = 0
    for intervention in data_dict_intervention:
        if (intervention,intervention['nbJoursInt']) not in data_dict_intervention_sans_doublons:
            data_dict_intervention_sans_doublons.append((intervention,intervention['nbJoursInt']))
            total += intervention['nbJoursInt']
            if  intervention['nbJoursInt']>maxi:
                maxi= intervention['nbJoursInt']
#        else:
#            print(intervention)
    print(len(data_dict_intervention_sans_doublons))
#    return total, maxi
#    print(data_dict_intervention_sans_doublons)
    n = 0
    for intervention_par_bateau in intervention_par_jours:
        n+=taille(intervention_par_bateau)[1]
        for interventions in intervention_par_bateau:
            for intervention in interventions:
                if (intervention,intervention['nbJoursInt']) not in data_dict_intervention_sans_doublons:
                    print('PROBLEME')
                detection.append((intervention,intervention['nbJoursInt']))
    return len(detection)

def aligner_gauche(liste_interventions):
    for j in range(len(liste_interventions)):
        if liste_interventions[j] != []:
            decalage = j-min_beg(liste_interventions[j])
            if decalage > 0:
                i=1
                while liste_interventions[j-i] == [] and i <= decalage:
#                    print("On descend",liste_interventions[j-i])
#                    time.sleep(0.5)
                    i+=1
                if liste_interventions[j-i+1] == []:
#                    print("On échange",j-i+1,"et",j)
                    echange(j-i+1,j,liste_interventions)
                
def nombre_vide(liste):
    vide = 0
    for l in liste:
        if l==[]:
            vide+=1
    return vide

def tri_liste_intervention_beg(liste_interventions):
    for i in range(len(liste_interventions)):
        for j in range(i,len(liste_interventions)):
            if max_beg(liste_interventions[j]) < max_beg(liste_interventions[i]):
                echange(i,j,liste_interventions)
            elif max_beg(liste_interventions[j]) == max_beg(liste_interventions[i]) and min_end(liste_interventions[j]) < min_end(liste_interventions[i]):
                echange(i,j,liste_interventions)

def nombre_paquet(liste_interventions):
    p = 0
    i = 0
    while (i<len(liste_interventions)):
        while(i<len(liste_interventions) and liste_interventions[i] == []):
            i+=1
        while (i<len(liste_interventions) and liste_interventions[i] !=[]):
            i+=1
        p+=1
    if liste_interventions[-1] == []:
        return p-1
    else:
        return p

def decalage_droite(paquet,debut):
    if len(paquet)>0:
        end = []
        decalages = []
        for liste_interventions in paquet:
            end.append(min_end(liste_interventions))
        for i in range(len(end)):
            decalages.append(end[i]-debut-i)
        #print(decalages)
        return min(decalages)
    else:
        return 0
        
def regrouper(liste_interventions):
    continuer = True
    while continuer:
        continuer = False
        debut_paquet = 0
        fin_paquet = debut_paquet+1
        while fin_paquet < len(liste_interventions):
            while(debut_paquet < len(liste_interventions) and liste_interventions[debut_paquet] == []):
                debut_paquet+=1
            fin_paquet=debut_paquet+1
            while(fin_paquet < len(liste_interventions) and liste_interventions[fin_paquet] != []):
                fin_paquet+=1
            decalage = decalage_droite(liste_interventions[debut_paquet:fin_paquet],debut_paquet)
            if decalage > 0 and fin_paquet<len(liste_interventions):
                i=0
                while fin_paquet+i<len(liste_interventions) and liste_interventions[fin_paquet+i] == [] and i < decalage:
                    i+=1
                #print("Déplacement du paquet de taille",fin_paquet-debut_paquet,"débutant à",debut_paquet,"qui debutera",debut_paquet+i)
                continuer = True
                pack = liste_interventions[debut_paquet:fin_paquet].copy()
                liste_interventions[debut_paquet:fin_paquet]=[[] for m in range(fin_paquet-debut_paquet)]
                liste_interventions[debut_paquet+i:fin_paquet+i] = pack
                debut_paquet = fin_paquet+i
                break
            else:
                debut_paquet = fin_paquet
      
def deb_fin_paquet(liste_interventions):
    n = nombre_paquet(liste_interventions)
    paquets = []
    p = 0
    for k in range(n):
        while(liste_interventions[p]==[]):
            p+=1
        d=p
        while(p<len(liste_interventions) and liste_interventions[p]!=[]):
            p+=1
        f=p
        paquets.append((d,f))
    return paquets

def check_final(bateau):
    for interventions in intervention_par_jours[bateau]:
        if interventions!=[]:
            print(max_beg(interventions),intervention_par_jours[bateau].index(interventions),min_end(interventions),max_beg(interventions)<=intervention_par_jours[bateau].index(interventions) and intervention_par_jours[bateau].index(interventions)<=min_end(interventions))
#                return False
#    return True



def plusieurs_listes_en_une_liste(liste_interventions):
    l = []
    for interventions in liste_interventions:
        l+=interventions
    return l

def retirer_doublons(liste_interventions):
    l = []
    for i in range(len(liste_interventions)):
        if liste_interventions[i] not in l:
            l.append(liste_interventions[i])
    return l

def densification(liste_interventions, bateau):
    aligner_gauche(liste_interventions)
    regrouper(liste_interventions)
    paquets = deb_fin_paquet(liste_interventions)
    for couple in paquets:
        l = liste_interventions[couple[0]:couple[1]]
        t = len(l)
        l = plusieurs_listes_en_une_liste(l)
        l = retirer_doublons(l)
        sol = saving_VRP(l,list_boat_med[bateau]['Port'],bateau,14)
        tri_liste_intervention_beg(sol)
        liste_interventions[couple[0]:couple[1]] = [[] for i in range(t)]
        liste_interventions[couple[0]:(couple[0]+len(sol))] = sol
    aligner_gauche(liste_interventions)
    regrouper(liste_interventions)
    disponibilité_jours(intervention_par_jours,bateau)
        
        
    
    
def ecriture_solution(interventions_par_jours):
    f = open("solution.txt", "w")
    for intervention_bateau in interventions_par_jours:
        f.write(str(intervention_bateau))
    f.close()
      
#    for intervention in impossible:
#        n+=intervention['nbJoursInt']
#        detection.append((intervention,intervention['nbJoursInt']))
#        if (intervention,intervention['nbJoursInt']) not in data_dict_intervention_sans_doublons:
#            print('PROBLEME')
#    for intervention in impossible_2:
#        n+=intervention['nbJoursInt']
#        detection.append((intervention,intervention['nbJoursInt']))
#        if (intervention,intervention['nbJoursInt']) not in data_dict_intervention_sans_doublons:
#            print('PROBLEME')
#    for intervention in impossible:
#        n+=intervention['nbJoursInt']
#        detection.append((intervention,intervention['nbJoursInt']))
#        if (intervention,intervention['nbJoursInt']) not in data_dict_intervention_sans_doublons:
#            print('PROBLEME')
#    for intervention in campagne[0]:
#        n+=intervention['nbJoursInt']
#        detection.append((intervention,intervention['nbJoursInt']))
#        if (intervention,intervention['nbJoursInt']) not in data_dict_intervention_sans_doublons:
#            print('PROBLEME')
#    for intervention in campagne[1]:
#        n+=intervention['nbJoursInt']
#        detection.append((intervention,intervention['nbJoursInt']))
#        if (intervention,intervention['nbJoursInt']) not in data_dict_intervention_sans_doublons:
#            print('PROBLEME')
##    for couple in data_dict_intervention_sans_doublons:
##        if couple not in detection:
##            print(couple)
##    print('\n')
##    for couple in detection:
##        if couple not in data_dict_intervention_sans_doublons:
##            print(couple) 
#    doublon_detection = []
#    for couple in detection:
#        if couple not in doublon_detection:
#            doublon_detection.append(couple)
##        else:
##           print(couple)
#    return n

'''
Iles Sanguinaires II AJACCIO
Provence MARSEILLE
Saint Clair 7 SETE
L'Esquillade TOULON
Iles Lavezzi BONIFACIO
Arnette MARSEILLE
'''