"""
Algorithme modulaire modifié puique sur le papier il ne tient pas compte de l'hétérogéneité de la flotte.
Algo:
    Itération :
        Affection des interventions a un bateau
        Affectetion des interventions a une période. 
        Design des routes effectuées par les bateaux.   
    

"""

import io
import cProfile
import numpy as np
import pandas as pd
import json
import Port as po
import sys
import math
import copy
import itertools
import matplotlib.pyplot as plt
from shapely.geometry import asShape
import random
from collections import deque
import pulp as pu
#Normalement marche sur n'importe quelle machine, il faut que le fichier importé soit dans le répertoire précédent.
sys.path.append('../../')
import selector as sel
import Interventions as it
import Solution as so
import Bateau as ba
import config as cfg

ROADPERMONTH = cfg.ROADPERMONTH
TIMEBOAT = cfg.TIMEBOAT
TIMEESCALE = cfg.TIMEESCALE
TIMEBOATNT = cfg.TIMEBOATNT 
TIMEBOATCAMPAGNE = cfg.TIMEBOATCAMPAGNE
TIMEBOATLONG = cfg.TIMEBOATLONG
DAYWEEK = cfg.DAYWEEK
ORDEREDMED = cfg.ORDEREDMED
ORDEREDATL = cfg.ORDEREDATL
MINDAYMOVE = cfg.MINDAYMOVE
BASETECHNIQUENAVIREATL = cfg.BASETECHNIQUENAVIREATL
BASETECHNIQUENAVIREMED = cfg.BASETECHNIQUENAVIREMED
TIMEBOATAFFPORT = cfg.TIMEBOATAFFPORT
NBJOURS = cfg.NBJOURS
SEASON = cfg.SEASON
MAXLONGROUTE = cfg.MAXLONGROUTE

# Tableau des rewards pour chaque balise et chaque type de bateau
REWARD=cfg.REWARD
# REWARD[bat.type-1][int.type-1]

def sortDay(val):
    if self.interventions[val[0]].double or self.interventions[val[0]].reste : 
        return 14 * val[1]
    else:
        return val[2]

def sortSecond(val): 
    return val[1]
    
def sortThird(val):
    return val[2]



class MHalgorithm:
    CPT = 0 
    def __init__(self,nbJours,area):
    
        self.nbJours = nbJours
        
        # TODO liste des solutions diverses, diversité d'une solution à définir
        self.listBestSolution = list()
        self.listDiverseSolution = list()
        self.area = area
        
        
        return
    
    
    def readInstancesFile(self,filenameIntervention,filenameBoat,filenameDistanceBB,filenameDistanceBP,filenameDistancePP,filenameDistance20NMBB,filenameDistance20NMBP,filenameDistance20NMPP,filenamePort):
        """
        filenameIntervention un fichier avec toute les interventions, filenameBoat un fichier avec tout les Bateaux, filenamePort un fichier avec les
        ports considéré (soit les ports d'Atl soit ceux de Med).filenameDistanceBB,BP,PP les fichiers de distance entre les balises et les ports.
        """
        

        bats = json.load(io.open(filenameBoat,'r',encoding='utf-8-sig'))
        boats = list()
        if self.area =="MED":
            for b in range (len(bats)):
                for ind in range (len(ORDEREDMED)):
                    if bats[b]["Port"] in ORDEREDMED[ind]:
                        boats.append(bats[b])
                                     
        else:
            for b in range (len(bats)):
                for ind in range (len(ORDEREDATL)):
                    if bats[b]["Port"] in ORDEREDATL[ind]:
                        boats.append(bats[b])
            pass
        
        interventions=json.load(open(filenameIntervention))
        dfBB=pd.read_csv(filenameDistanceBB)
        dfBP=pd.read_csv(filenameDistanceBP)
        dfPP=pd.read_csv(filenameDistancePP)

        df20NMBB=pd.read_csv(filenameDistance20NMBB)
        df20NMBP=pd.read_csv(filenameDistance20NMBP)
        df20NMPP=pd.read_csv(filenameDistance20NMPP)
        
        dfPort = pd.read_csv(filenamePort)
        
        self.esm = list(dfBB.columns.values)
        self.esm20NM = list(df20NMBB.columns.values)
        
        self.nomPort = list(dfBP.columns.values)
        self.nomPort20NM = list(df20NMBP.columns.values)
        
        self.boats = list()
        for idBoat in range (len(boats)):
            self.boats.append(ba.Bateau(boats[idBoat],idBoat))
        

        
        self.ports = dict()
        
        for idPort in range (len(dfPort)):
            self.ports[dfPort.iloc[idPort]["NOM_SUBDI"]]=po.Port(dfPort.iloc[idPort],self.nomPort.index(dfPort.iloc[idPort]["NOM_SUBDI"]),self.nomPort20NM.index(dfPort.iloc[idPort]["NOM_SUBDI"]),self.nbJours)
        
        # Matrices des distances
        distanceMatriceNMBB = np.asarray(dfBB.values)
        distanceMatriceNMBP = np.asarray(dfBP.values)
        distanceMatriceNMPP = np.asarray(dfPP.values)
        
        distanceMatrice20NMBB = np.asarray(df20NMBB.values)
        distanceMatrice20NMBP = np.asarray(df20NMBP.values)
        distanceMatrice20NMPP = np.asarray(df20NMPP.values)
        
        for i in range (len(distanceMatrice20NMBB)):
            for j in range (len(distanceMatrice20NMBB[i])):
                if distanceMatrice20NMBB[i][j] < 0:
                    distanceMatrice20NMBB[i][j] = math.inf
                    
        for i in range (len(distanceMatrice20NMBP)):
            for j in range (len(distanceMatrice20NMBP[i])):
                if distanceMatrice20NMBP[i][j] < 0:
                    distanceMatrice20NMBP[i][j] = math.inf
                    
        for i in range (len(distanceMatrice20NMPP)):
            for j in range (len(distanceMatrice20NMPP[i])):
                if distanceMatrice20NMPP[i][j] < 0:
                    distanceMatrice20NMPP[i][j] = math.inf


        
        self.distanceMatriceBB = [distanceMatriceNMBB,distanceMatrice20NMBB]
        self.distanceMatriceBP = [distanceMatriceNMBP,distanceMatrice20NMBP]
        self.distanceMatricePP = [distanceMatriceNMPP,distanceMatrice20NMPP]
        
        self.interventions = list()
        idInt = 0
        for i in interventions:
            
            self.interventions.append(it.Intervention(self.esm.index(i["esm"]),self.esm20NM.index(i["esm"]),idInt,i,len(boats),self.nbJours))
            idInt += 1


    
    def initiateAlgo(self,alpha,beta,nbBestSolutionkept,nbDiverseSolutionkept,config = None):
    
        # psi est le paramètre sur l'affectation des interventions à un port.
        self.alpha = alpha
        
        # beta est le paramtre sur l'affectation des interventions à un temps
        self.beta = beta
        
        # Le nombre de solutions que qu'on garde en mémoire.
        self.nbBestSolutionkept = nbBestSolutionkept
        self.nbDiverseSolutionkept = nbDiverseSolutionkept
        
        self.solutionKept = list()
        
        # config est une liste qui indique quel bateau est affecté à quel port,dictionnaire avec clé le port et valeur une liste des bateaux 
        # affectées à ce port.
        # Transforme la config pour avoir des identifiants ports et bateaux à la place des noms.

         
        self.numBoatConfig = list()
        
        if config == None:
            self.numBoatConfig = [boat.num for boat in self.boats]
            for bat in self.boats:
                bat.setPort(self.ports[ bat.portPrimaire].id,bat.portPrimaire,self.ports[bat.portPrimaire].coord,self.area)
        
        else:
            for portName in config.keys():
                    for batName in config[portName]:
                        for bat in self.boats:
                            if batName == bat.nom:
                                bat.setPort(self.ports[portName].id,portName,self.ports[portName].coord,self.area)
        
                                self.numBoatConfig.append(bat.num)
        self.batParType = [[] for i in range (4)]
        for b in self.numBoatConfig:
            self.batParType[self.boats[b].type-1].append(b)
        print(self.batParType)      
        self.batPerZone = list()
        self.zone = list()
        if self.area == "MED":
            self.zone = ORDEREDMED
            for i in range (len(self.zone)):
                self.batPerZone.append([])
        else:
            self.zone = ORDEREDATL
            for i in range (len(self.zone)):
                self.batPerZone.append([])
    
        for b in self.numBoatConfig:
            if self.boats[b].habitable :
                # On regroupe les bateaux habitables par zone
                print(self.boats[b].nom)
                self.batPerZone[self.boats[b].navigationArea].append(b)
                
        # identifiant des ports dans la liste ordonnée
        
        for pName in self.ports.keys():
            for indZone in range (len(self.zone)):
                if pName in self.zone[indZone]:
                    self.ports[pName].setZone(indZone,self.zone[indZone].index(pName)) 
        
        
        cptInt1 = 0
        cptInt2 = 0
        cptInt3 = 0          
        for i in self.interventions:
            if i.type == 1:
                cptInt1 += 1
            if i.type == 2:
                cptInt2 += 1
            
            if i.type == 3:
                cptInt3 += 1
        print("Voici le nombre d'interventions : ",cptInt1,cptInt2,cptInt3)      
                                   

############################################################## DEPLACEMENT DES BATEAUX ##################################################################

    def plotHisto(self,solution):
        """
        Essaye de trouver un déplacement des bateaux pour répondre à la demande en attribuant un bateau au port ou il est le plus demandé.
        """
    
        listBoatPossMove = self.batParType[2]
        # L'ensemble des bateaux pouvant être échangés .
        
                

        #Test PLOT HISTOGRAMME
        histoInt = list()
        for inter in self.interventions:

            for time in range (inter.beginning,inter.end+1):
                histoInt.append(time)
        plt.hist(histoInt, range = (0, self.nbJours), bins = self.nbJours, color = 'yellow',edgecolor = 'red')
        plt.show()
        
        histo = dict()
        
        for b in listBoatPossMove:
            histo[b] = dict()

            for p in self.boats[b].escale.keys():
                histo[b][p] = list()
                
                for idInt in range (len(self.interventions)):
                    inter = self.interventions[idInt]
                    if self.listInterventionsDone[idInt] == False:
                        if inter.listBoat[b] == 1 :
                            if inter.retrieve == False or self.boats[b].escale[p] == 1:
                                if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ p ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time <= TIMEBOAT :
                                
                                    for month in range ((inter.beginning)//16,((inter.end+1)//16)+1):
                                        histo[b][p].append(month)
                                        
        for b in listBoatPossMove:
            listHist = list()        
            labels = list()
            for p in self.boats[b].escale.keys():
                labels.append(p)
                listHist.append(histo[b][p])
                #plt.hist(histo[b][p], range = (0, self.nbJours), bins = self.nbJours, color = 'yellow',edgecolor = 'red')
                #plt.title(str(self.boats[b].nom) + "   "+p)
                #plt.show()
            plt.hist(listHist,range = (0, self.nbJours//13), bins = self.nbJours//13,edgecolor = 'red',label = labels, hatch = '/', histtype = 'bar' )
            plt.title(str(self.boats[b].nom) )
            plt.legend()
            plt.show()
        
        return
        
        
        
    def routeToCampaign(self,solution,b,month,TIME = TIMEBOAT):
        
        
        boat = self.boats[b]
            
        if (self.distanceMatricePP[ self.boats[b].indMatrice ][ self.ports[ boat.dictMonth[month]].id[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)  <= TIME :
        
            # Le bateau n'a pas besoin de faire d'escale.
            
            departure = self.boats[b].nomPortAtt
            arrival = boat.dictMonth[month]
            
            listInt = list()
            
            for idInt in range (len(self.interventions)):
                inter = self.interventions[idInt]
                if self.listInterventionsDone[idInt] == False and self.interventions[idInt].reste == False and self.interventions[idInt].double == False:
                    if inter.listBoat[b] == 1:
                        if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse) + inter.time <= TIME :
                            listInt.append(inter.idInt)

            route,timeSpent,nbIntRetrieve = self.compute_route_semaine(b,month*16,boat.maxHourWeek,departure, arrival, listInt, check = True, stock = boat.stock, route = [],tempsBat = TIME)
            
            
            for idInt in route:
                listInt.remove(idInt)
                self.interventions[idInt].setInterTime(month*16)
                self.interventions[idInt].setInterBoat(b)
            solution.addRoute(b, month*16, route, departure, arrival)
            
            departure = boat.dictMonth[month]
            arrival = boat.nomPortAtt
            
            route2,timeSpent2,nbIntRetrieve2 = self.compute_route_semaine(b,(month+1)*16-1,boat.maxHourWeek,departure, arrival, listInt, check = True, stock = boat.stock, route = [],tempsBat = TIME)
            for idInt in route2:
                self.interventions[idInt].setInterTime( (month+1)*16-1 )
                self.interventions[idInt].setInterBoat(b)
            solution.addRoute(b, (month+1)*16-1, route2, departure, arrival)
            
            self.boats[b].disponibility[month * 16] = False
            self.boats[b].disponibility[ (month+1)*16-1 ] = False
            
            
            return 1,timeSpent,timeSpent2
            
            
        else:
            # Le bateau doit au moins faire une escale.
            portEscale = list()
                                    
            for p in boat.escale.keys():
                if (self.distanceMatricePP[ self.boats[b].indMatrice ][ self.ports[ boat.dictMonth[month]].id[ self.boats[b].indMatrice ] ][ self.ports[ p ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse) <= TIME :
                    if (self.distanceMatricePP[ self.boats[b].indMatrice ][ self.ports[ p ].id[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse) <= TIME :
                    
                        portEscale.append(p)
                        
                        
            if len(portEscale) == 0:
            
                if boat.type == 2 and TIME == TIMEESCALE:
                    # Dans le cas ou on ne peut pas faire d'escale en moins de 6 heures.
                    return self.routeToCampaign(solution,b,month,TIME = TIMEBOAT)
                
                # Une deuxième escale est necessaire
                portEscale = list()
                for p in boat.escale.keys():
                    for p2 in boat.escale.keys():
                        if (self.distanceMatricePP[ self.boats[b].indMatrice ][ self.ports[ boat.dictMonth[month]].id[ self.boats[b].indMatrice ] ][ self.ports[ p2 ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse) <= TIME :
                            if (self.distanceMatricePP[ self.boats[b].indMatrice ][ self.ports[ p ].id[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)  <= TIME :
                                if (self.distanceMatricePP[ self.boats[b].indMatrice ][ self.ports[ p ].id[ self.boats[b].indMatrice ] ][ self.ports[ p2 ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse) <= TIME :
                                    portEscale.append((p,p2))
                
                
                bestEscale = None
                maxRouteLenght = 0
                listRoute_Time = dict()
                for p in portEscale:
                    listPort = [boat.nomPortAtt,p[0],p[1],boat.dictMonth[month]]
                    listRoute_Time[p] = list()
                    listInt = list()
                    maxHour = boat.maxHourWeek
                    for indP in range (1,len(listPort)-1):
                        
                        for idInt in range (len(self.interventions)):
                            inter = self.interventions[idInt]
                            if self.listInterventionsDone[idInt] == False and self.interventions[idInt].reste == False and self.interventions[idInt].double == False:
                                if inter.listBoat[b] == 1:
                                    if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ listPort[indP] ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse) + inter.time <= TIME :
                                        if inter.idInt not in listInt: 
                                                listInt.append(inter.idInt) 
                    for indP in range (1,len(listPort)):
                        
                        departure = listPort[indP - 1]
                        arrival = listPort[indP]
                        route,timeSpent,nbIntRetrieve = self.compute_route_semaine(b,month*16+(indP-1),boat.maxHourWeek,departure, arrival, listInt, check = False, stock = boat.stock, route = [],tempsBat = TIME)
                        
                        maxHour  -= timeSpent
                        
                        for idInt in route:
                            listInt.remove(idInt)



                        listRoute_Time[p].append((route,timeSpent)) 
                
                maxLenRoute = 0
                for p in portEscale:
                    lenRoute = sum([ len(route[0]) for route in listRoute_Time[p]])
                        
                    if lenRoute >= maxLenRoute:
                         maxLenRoute = lenRoute
                         bestEscale = p 

                solution.addRoute(b, month*16, listRoute_Time[bestEscale][0][0], departure = boat.nomPortAtt , arrival = bestEscale[0])
                solution.addRoute(b, month*16+1, listRoute_Time[bestEscale][1][0], departure = bestEscale[0] , arrival = bestEscale[1])
                solution.addRoute(b, month*16+2, listRoute_Time[bestEscale][2][0], departure = bestEscale[1] , arrival = boat.dictMonth[month])
                self.boats[b].disponibility[month * 16] = False
                self.boats[b].disponibility[month * 16 + 1] = False
                self.boats[b].disponibility[month * 16 + 2] = False
                
                cptDay = 0
                for route in listRoute_Time[bestEscale]:
                    for idInt in route[0]:
                        self.listInterventionsDone[idInt] = True 
                        self.interventions[idInt].setInterTime( month*16 + cptDay )
                        self.interventions[idInt].setInterBoat(b)
                    cptDay += 1
                
                bestEscaleRet = None
                maxRouteLenghtRet = 0
                listRoute_TimeRet = dict()
                for p in portEscale:
                    listPort = [boat.nomPortAtt,p[0],p[1],boat.dictMonth[month]]
                    listRoute_TimeRet[p] = list()
                    listInt = list()
                    maxHour = boat.maxHourWeek
                    for indP in range (1,len(listPort)-1):
                        
                        for idInt in range (len(self.interventions)):
                            inter = self.interventions[idInt]
                            if self.listInterventionsDone[idInt] == False and self.interventions[idInt].reste == False and self.interventions[idInt].double == False:
                                if inter.listBoat[b] == 1:
                                    if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ listPort[indP] ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse) + inter.time <= TIME :
                                        if inter.idInt not in listInt: 
                                                listInt.append(inter.idInt) 
                    for indP in range (1,len(listPort)):
                        
                        departure = listPort[indP]
                        arrival = listPort[indP - 1]
                        route,timeSpent,nbIntRetrieve = self.compute_route_semaine(b,(month+1)*16-indP,maxHour,departure, arrival, listInt, check = False, stock = boat.stock, route = [],tempsBat = TIME)
                        
                        maxHour -= timeSpent
                        
                        for idInt in route:
                            listInt.remove(idInt)
                        listRoute_TimeRet[p].append((route,timeSpent)) 
                
                maxLenRoute = 0
                for p in portEscale:
                    lenRoute = sum([ len(route[0]) for route in listRoute_TimeRet[p]])
                        
                    if lenRoute >= maxLenRoute:
                         maxLenRoute = lenRoute
                         bestEscaleRet = p 
                
                
                solution.addRoute(b, (month+1)*16 - 1, listRoute_TimeRet[bestEscaleRet][0][0], departure = bestEscaleRet[0] , arrival = boat.nomPortAtt)
                solution.addRoute(b, (month+1)*16 - 2, listRoute_TimeRet[bestEscaleRet][1][0], departure = bestEscaleRet[1] , arrival = bestEscaleRet[0])
                solution.addRoute(b, (month+1)*16 - 3, listRoute_TimeRet[bestEscaleRet][2][0], departure = boat.dictMonth[month] , arrival = bestEscaleRet[1])
                
                self.boats[b].disponibility[ (month+1)*16-1 ] = False
                self.boats[b].disponibility[ (month+1)*16-2 ] = False
                self.boats[b].disponibility[ (month+1)*16-3 ] = False
                
                cptDay = 0
                for route in listRoute_TimeRet[bestEscaleRet]:
                    for idInt in route[0]:
                        self.listInterventionsDone[idInt] = True
                        self.interventions[idInt].setInterTime( (month+1)*16 - (cptDay+1) )
                        self.interventions[idInt].setInterBoat(b)
                    cptDay += 1
                    

#                print(bestEscale,bestEscaleRet,listRoute_Time,listRoute_TimeRet)
#                print(portEscale,boat.dictMonth[month])           
#                input() 
                return 3,sum([ob[1] for ob in  listRoute_Time[bestEscale] ]), sum([ob[1] for ob in  listRoute_TimeRet[bestEscaleRet] ])
                              
            else:

                for p in portEscale:
                    listInt = list()
                    
                    listRoutePerEscale = dict()
                    listRoutePerEscaleReturn = dict()
                    for idInt in range (len(self.interventions)):
                        inter = self.interventions[idInt]
                        if self.listInterventionsDone[idInt] == False and self.interventions[idInt].reste == False and self.interventions[idInt].double == False:
                            if inter.listBoat[b] == 1:
                                if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ p ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse) + inter.time <= TIME :
                                    listInt.append(inter.idInt)
                    
                    
                    departure = boat.nomPortAtt 
                    arrival = p
                    route,timeSpent,nbIntRetrieve = self.compute_route_semaine(b,month*16,boat.maxHourWeek,departure, arrival, listInt, check = False, stock = boat.stock, route = [],tempsBat = TIME)
                    
                    for idInt in route:
                        listInt.remove(idInt)
                        
                    departure = p
                    arrival = boat.dictMonth[month]
                    route2,timeSpent2,nbIntRetrieve2 = self.compute_route_semaine(b,month*16+1,boat.maxHourWeek,departure, arrival, listInt, check = False, stock = boat.stock, route = [],tempsBat = TIME)
                    
                    for idInt in route2:
                        listInt.remove(idInt)
                    
                    listRoutePerEscale[p] = [route, route2, timeSpent+timeSpent2]    
                    
                    departure = boat.dictMonth[month]
                    arrival = p
                    routeRe,timeSpentRe,nbIntRetrieveRe = self.compute_route_semaine(b,(month+1)*16-2,boat.maxHourWeek,departure, arrival, listInt, check = False, stock = boat.stock, route = [],tempsBat = TIME)
                    
                    for idInt in routeRe:
                        listInt.remove(idInt)
                        
                    departure = p
                    arrival = boat.nomPortAtt 
                    routeRe2,timeSpentRe2,nbIntRetrieveRe2 = self.compute_route_semaine(b,(month+1)*16-1,boat.maxHourWeek,departure, arrival, listInt, check = False, stock = boat.stock, route = [],tempsBat = TIME)
                    
                    
                    listRoutePerEscaleReturn[p] = [routeRe, routeRe2, timeSpentRe+timeSpentRe2]    
                    
                # Prendre la route qui fait le plus d'interventions.
                
                bestEscale = None
                nbInt = 0
                #print(listRoutePerEscale,listRoutePerEscaleReturn)
                for p in listRoutePerEscale.keys():
                    nbIntRetrieve = len( listRoutePerEscale[p][0] ) + len( listRoutePerEscale[p][1] )  
                    if  nbIntRetrieve >= nbInt:
                        nbInt = nbIntRetrieve
                        bestEscale = p
                        
                    
                solution.addRoute(b, month*16, listRoutePerEscale[bestEscale][0], departure = boat.nomPortAtt , arrival = bestEscale)
                solution.addRoute(b, month*16+1, listRoutePerEscale[bestEscale][1], departure = bestEscale , arrival = boat.dictMonth[month])
                for idInt in listRoutePerEscale[bestEscale][0]:
                    self.interventions[idInt].setInterTime(month*16)
                    self.interventions[idInt].setInterBoat(b)
                    self.listInterventionsDone[idInt] = True
                for idInt in listRoutePerEscale[bestEscale][1]:
                    self.interventions[idInt].setInterTime(month*16+1)
                    self.interventions[idInt].setInterBoat(b)
                    self.listInterventionsDone[idInt] = True
                    
                bestEscale2 = None
                nbInt = 0
                for p in listRoutePerEscaleReturn.keys():
                    nbIntRetrieve = len( listRoutePerEscaleReturn[p][0] ) + len( listRoutePerEscaleReturn[p][1] )  
                    if  nbIntRetrieve >= nbInt:
                        nbInt = nbIntRetrieve
                        bestEscale2 = p
                            
                    
                solution.addRoute(b, (month+1)*16-2, listRoutePerEscaleReturn[bestEscale2][0], departure = boat.dictMonth[month] , arrival = bestEscale2)
                solution.addRoute(b, (month+1)*16-1, listRoutePerEscaleReturn[bestEscale2][1], departure = bestEscale2 , arrival = boat.nomPortAtt)
                for idInt in listRoutePerEscaleReturn[bestEscale2][0]:
                    self.interventions[idInt].setInterTime((month+1)*16-1)
                    self.interventions[idInt].setInterBoat(b)
                    self.listInterventionsDone[idInt] = True
                for idInt in listRoutePerEscaleReturn[bestEscale2][1]:
                    self.interventions[idInt].setInterTime((month+1)*16-2)
                    self.interventions[idInt].setInterBoat(b)
                    self.listInterventionsDone[idInt] = True
                    
                    
                self.boats[b].disponibility[month * 16] = False
                self.boats[b].disponibility[month * 16 + 1] = False
                self.boats[b].disponibility[ (month+1)*16-1 ] = False
                self.boats[b].disponibility[ (month+1)*16-2 ] = False
                # On retourne les temps passés pour faire les routes.
                return 2,listRoutePerEscale[bestEscale][2],listRoutePerEscaleReturn[bestEscale2][2]
                                    
                        
                     
        
        
    def boatPositionTest(self,solution,batType = [3,4]):
        """
        Essaye de trouver un déplacement des bateaux pour répondre à la demande en attribuant un bateau au port ou il est le plus demandé.
        """
        
        listBoatPossMove = list()
        
        # L'ensemble des bateaux pouvant être échangés .
        
        for b in self.numBoatConfig:
            if self.boats[b].type in batType:
                listBoatPossMove.append(b)
        
        listIntCovered = list(self.listInterventionsDone)
        
        nbBoatCover = dict()
        campaignLeft = dict()
        possibleMonth = dict()
        campaign = dict()
        for b in listBoatPossMove:
            campaignLeft[b] = 4
            nbBoatCover[b] = (self.boats[b].maxDayYear - solution.evalBatNbJour[b])
            possibleMonth[b] = dict()
            campaign[b] = dict()
            for month in range (self.nbJours//16):
                possibleMonth[b][month] = True
            
            
            
        for idInt in range (len(self.interventions)):
            inter = self.interventions[idInt]
            if listIntCovered[idInt] == False:
                if inter.reste or inter.double:
                # Filtre pour voir si elle est recouverte, recouvrir d'abord les NTs et ensuite les BCs concernant les fixes et doubles, sinon d'abord les BA ensuite NT et enfin BC. 
                    covered = False
                    for typeBat in batType: 
                        if covered:
                            break
                        for b in self.batParType[typeBat-1]:
                            if inter.listBoat[b] == 1:
                            
                                time = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                                if time <= TIMEBOAT and min(inter.nbJoursInt,4) * time < self.boats[b].maxHourWeek:
                                    if nbBoatCover[b] + inter.nbJoursInt <= self.boats[b].maxDayYear:
                                        covered = True
                                        nbBoatCover[b] += inter.nbJoursInt
                                        break
                            
                    if covered:
                        listIntCovered[idInt] = True
                        
        for idInt in range (len(self.interventions)):
            inter = self.interventions[idInt]
            if listIntCovered[idInt] == False:
                if inter.reste == False and inter.double == False:
                    covered = False
                    batType2 = [1,4,2,3]
                    for typeBat in batType2:
                        if typeBat in batType: 
                            if covered:
                                break
                            for b in self.batParType[typeBat-1]:
                                if inter.listBoat[b] == 1:
                                
                                    time = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                                    if time <= TIMEBOAT and min(inter.nbJoursInt,4) * time < self.boats[b].maxHourWeek:
                                        if nbBoatCover[b] + inter.time/14.0 < self.boats[b].maxDayYear:
                                            covered = True
                                            nbBoatCover[b] += inter.time/14.0
                                            break
                            
                    if covered:
                        listIntCovered[idInt] = True
              
              
#        for b in listBoatPossMove:
#            print(self.boats[b].nom,nbBoatCover[b])
                
#        listNum = list()
#        for idInt in range (len(self.interventions)):
#            if listIntCovered[idInt] == False:
#                listNum.append(self.interventions[idInt].numEsm)
#        solution.prepareToPlot()
#        solution.plotNotPossible(listNum)
        
        
        

        
        
        nbNotCovered = sum([1 for i in listIntCovered if i == False])
        nbCampaignLeft = len(listBoatPossMove) * 4
        
        
        while ( nbCampaignLeft != 0 and nbNotCovered != 0):
            # Prendre le bateau avec le moins d'interventions "attribuées", favoriser les baliseurs d'abord et ensuite les baliseur côtier.
            
            

            
            choosenBoat = None
            nbBoatCoverMax = -math.inf
            typeBoat = math.inf
            listPref = [3,4,2,1]
            for b in listBoatPossMove:
                if self.boats[b].maxDayYear - nbBoatCover[b] > nbBoatCoverMax and campaignLeft[b] > 0 and listPref.index(self.boats[b].type) <= typeBoat:
                    choosenBoat = b
                    
                    nbBoatCoverMax = self.boats[b].maxDayYear - nbBoatCover[b]
                    typeBoat = listPref.index(self.boats[choosenBoat].type)
            if nbBoatCover[choosenBoat] > self.boats[choosenBoat].maxDayYear - 16 :
                # Dans le cas ou le bateau travaille deja trop.

                nbCampaignLeft -= 1
                campaignLeft[choosenBoat] -= 1
                continue
                
            # Regarder parmi l'ensemble des ports où il peut se déplacer, celui ou le nombre d'interventions qu'il peut réaliser est le plus grand.
            possibleIntPerHarbour = dict()
            maxPortMonth = 0
            maxs = []
            bestHarbour = None
            bestMonth = None
            for p in self.boats[choosenBoat].escale.keys():
                possibleIntPerHarbour[p] = dict()
                for month in range (self.nbJours//16):
                    if possibleMonth[choosenBoat][month]:
                        possibleIntPerHarbour[p][month] = list()
                        for idInt in range ( len(listIntCovered) ):
                            if listIntCovered[idInt] == False:
                                inter = self.interventions[idInt]
                                

                                if inter.listBoat[choosenBoat] == 1 and ( (inter.beginning <=month*16 and inter.end > month*16 ) or (inter.beginning > month*16 and inter.beginning < (month+1)*16 )):
                                    if inter.retrieve == False or self.boats[choosenBoat].escale[p] == 1 or self.boats[choosenBoat].escale[p] == 0 :
                                        time = (self.distanceMatriceBP[ self.boats[choosenBoat].indMatrice ][ inter.numLine[ self.boats[choosenBoat].indMatrice ] ][ self.ports[ p ].id[self.boats[choosenBoat].indMatrice] ]/self.boats[choosenBoat].vitesse)*2 + inter.time
                                        if inter.double:
                                            time += inter.time + 3
                                        if time <= TIMEBOAT and min(inter.nbJoursInt,4) * time < self.boats[choosenBoat].maxHourWeek:
                                            possibleIntPerHarbour[p][month].append(idInt)
                        nbCovered = 0                
                        for idInt in possibleIntPerHarbour[p][month]:
                            inter = self.interventions[idInt]
                            if inter.reste or inter.double:
                                nbCovered += self.interventions[idInt].nbJoursInt
                            else:
                                nbCovered += self.interventions[idInt].time / 14.0
                        
                        if nbCovered > maxPortMonth:
                            maxPortMonth = nbCovered
                            bestHarbour = p
                            bestMonth = month
                        if nbCovered > 0:
                            maxs.append([nbCovered, p, month])

            # diminuer le nombre de campagnes possibles pour ce bateau, recouvrir les interventions                         

            if bestHarbour == None and bestMonth == None:
                # Dans le cas ou le bateau n'a rien a traiter.

                nbCampaignLeft -= 1
                campaignLeft[choosenBoat] -= 1
                continue
                
            #print(maxs)
            items = [id for id in maxs if id[0] >= maxPortMonth]
            print(items)
            random.shuffle(items)
            #items.sort(key=lambda x:(x[2], -x[0]))
            bestHarbour = items[0][1]
            bestMonth = items[0][2]
#            print(bestHarbour,bestMonth,self.boats[choosenBoat].nom)
#            input()
            self.boats[choosenBoat].initiateListPosition(bestMonth,bestHarbour)
            # Traitement des Campagnes
            
            # Définir les déplacements pour aller jusqu'a son port de campagne
            
            TimeBoatPerWeek = [self.boats[choosenBoat].maxHourWeek for we in range (4) ] 
            
            
            if self.boats[choosenBoat].type == 2:
                nbJoursTaken,timeSpent,timeSpentRet = self.routeToCampaign(solution,choosenBoat,bestMonth,TIME = TIMEESCALE)
            else:
                nbJoursTaken,timeSpent,timeSpentRet = self.routeToCampaign(solution,choosenBoat,bestMonth,)
            
            
            TimeBoatPerWeek[0] -= timeSpent
            TimeBoatPerWeek[- 1] -= timeSpentRet
            
            
            departure = self.boats[choosenBoat].dictMonth[bestMonth]
            arrival = departure
            
            listInterventionsPossible = list()

            for idInt in possibleIntPerHarbour[bestHarbour][bestMonth]:
                inter = self.interventions[idInt]
                if self.listInterventionsDone[idInt] == False and inter.listBoat[choosenBoat] == 1 and self.interventions[idInt].reste != True and self.interventions[idInt].double != True:
                    if  (inter.beginning <= (bestMonth*16 + nbJoursTaken) and inter.end > (bestMonth*16+ nbJoursTaken) ) or (inter.beginning > bestMonth*16+ nbJoursTaken and inter.beginning < (bestMonth+1)*16- nbJoursTaken ):
                
                        if (self.distanceMatriceBP[ self.boats[choosenBoat].indMatrice ][ inter.numLine[ self.boats[choosenBoat].indMatrice ] ][ self.ports[ departure ].id[self.boats[choosenBoat].indMatrice] ]/self.boats[choosenBoat].vitesse)*2 + inter.time <= TIMEBOAT :
                            listInterventionsPossible.append(inter)

            allTimePoss = [ time for time in range (bestMonth*16+ nbJoursTaken,(bestMonth+1)*16- nbJoursTaken) ]
            self.boats[choosenBoat].affectationInterventionDayRandomLearning(listInterventionsPossible,self.nbJours,allTimePoss)
                
            
            for time in range ( bestMonth*16 + nbJoursTaken ,(bestMonth+1)*16 - nbJoursTaken):
                nbWeek = (time - bestMonth*16)//DAYWEEK
                
                route,timeSpent,nbIntRetrieve = self.compute_route(choosenBoat,time,TimeBoatPerWeek[nbWeek], departure , arrival, stock = self.boats[choosenBoat].stock,route = [])
                
                for idInt in route:
                    
                    self.interventions[idInt].setInterTime(time)
                    self.interventions[idInt].setInterBoat(choosenBoat)

                solution.addRoute(choosenBoat,time,route,departure,arrival)
                
                
                TimeBoatPerWeek[nbWeek] -= timeSpent              
            
            
            solution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP,batType = [1,2,3,4])

            solution = self.fusionPhase(solution, 100, boatNum = choosenBoat, timeWindow = [ bestMonth*16,(bestMonth+1)*16 ])
            

            solution = self.fusionDayPhase(solution, 100, boatNum = choosenBoat, timeWindow = [ bestMonth*16,(bestMonth+1)*16 ])

            self.placement2(solution, listInt = possibleIntPerHarbour[bestHarbour][bestMonth], boatNum = choosenBoat, timeWindow = [ bestMonth*16,(bestMonth+1)*16-1])

            solution = self.exchangePhase(solution, 0, listInt = possibleIntPerHarbour[bestHarbour][bestMonth], boatNum = choosenBoat, timeWindow = [ bestMonth*16,(bestMonth+1)*16 ])


            solution = self.fusionPhase(solution, 100, boatNum = choosenBoat, timeWindow = [ bestMonth*16,(bestMonth+1)*16 ])

#            for day in range ((bestMonth*16),((bestMonth+1)*16)):
#                print(solution.routeBatPerDay[choosenBoat][day],solution.portBatPerDay[choosenBoat][day])
#                input()
            solution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP,batType = [1,2,3,4])
            
            for idInt in possibleIntPerHarbour[bestHarbour][bestMonth]:
                inter = self.interventions[idInt]
                if self.listInterventionsDone[idInt] == True and listIntCovered[idInt] == False:
                    listIntCovered[idInt] = True
                    
                    if inter.reste or inter.double:
                        nbBoatCover[choosenBoat] += inter.nbJoursInt
                    else:
                        nbBoatCover[choosenBoat] += inter.time / 14.0    
            
            
            nbNotCovered = sum([1 for i in listIntCovered if i == False])
            nbCampaignLeft -= 1
            campaignLeft[choosenBoat] -= 1
            possibleMonth[choosenBoat][bestMonth] = False
            
            campaign[choosenBoat][bestMonth] = bestHarbour
            
#        for b in listBoatPossMove:
#            print(self.boats[b].nom,nbBoatCover[b])
#        listNum = list()
#        for idInt in range (len(self.interventions)):
#            if listIntCovered[idInt] == False:
#                listNum.append(self.interventions[idInt].numEsm)
#        solution.prepareToPlot()
#        solution.plotNotPossible(listNum)
#        
        print(campaign)
        #input()
        return
            
                
    def boatPosition(self,solution):
        """
        Essaye de trouver un déplacement des bateaux pour répondre à la demande avec une couverture des interventions par les bateaux.
        4 mouvements max de 4 semaines max.
        """

        listBoatPossMove = list()
        # L'ensemble des bateaux pouvant être échangés .
        
        for b in self.numBoatConfig:
            if self.boats[b].habitable and self.boats[b].type == 3:
                listBoatPossMove.append(b)
                
        # Definir l'ensemble des ports possibles pour les bateaux
        listPossPort = dict()
        
        for b in listBoatPossMove:
            listPossPort[b] = list()
            for p in self.ports.keys():
            
                if self.boats[b].escale == None :
                    if   (self.ports[p].indZone == self.ports[self.boats[b].nomPortAtt].indZone):
                        indPortAtt = self.ports[self.boats[b].nomPortAtt].id[self.boats[b].indMatrice]
                        indPortSuiv = self.ports[p].id[self.boats[b].indMatrice]
                        
                        if self.distanceMatricePP[self.boats[b].indMatrice][indPortAtt][indPortSuiv] / self.boats[b].vitesse < TIMEBOATAFFPORT:
                            listPossPort[b].append( p )
                            
                            
                            
                            
                            
                            
                            
        #Test PLOT HISTOGRAMME
        histoInt = list()
        for inter in self.interventions:

            for time in range (inter.beginning,inter.end+1):
                histoInt.append(time)
        plt.hist(histoInt, range = (0, self.nbJours), bins = self.nbJours, color = 'yellow',edgecolor = 'red')
        plt.show()
        
        histo = dict()
        
        for b in listBoatPossMove:
            histo[b] = dict()

            for p in self.boats[b].escale.keys():
                histo[b][p] = list()
                
                for idInt in range (len(self.interventions)):
                    inter = self.interventions[idInt]
                    if self.listInterventionsDone[idInt] == False:
                        if inter.listBoat[b] == 1 :
                            if inter.retrieve == False or self.boats[b].escale[p] == 1:
                                if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ p ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time <= TIMEBOAT :
                                
                                    for month in range ((inter.beginning)//16,((inter.end+1)//16)+1):
                                        histo[b][p].append(month)
                                        
        for b in listBoatPossMove:
            listHist = list()        
            labels = list()
            for p in self.boats[b].escale.keys():
                labels.append(p)
                listHist.append(histo[b][p])
                #plt.hist(histo[b][p], range = (0, self.nbJours), bins = self.nbJours, color = 'yellow',edgecolor = 'red')
                #plt.title(str(self.boats[b].nom) + "   "+p)
                #plt.show()
            plt.hist(listHist,range = (0, self.nbJours//13), bins = self.nbJours//13,edgecolor = 'red',label = labels, hatch = '/', histtype = 'bar' )
            plt.title(str(self.boats[b].nom) )
            plt.legend()
            plt.show()
        
                    
        
        
        # Definir le temps ou le bateau va/peut partir
        # Chaque bateau a 4 temps possible 0,52,104 et 156
        
        matriceA = dict()
        
        for b in listBoatPossMove:
            matriceA[b] = dict()
            for season in range (0,self.nbJours,SEASON):

                matriceA[b][season] = dict()
                if self.boats[b].escale == None :
                    
                    for p in listPossPort[b]:
                        matriceA[b][season][p] = list()
                        
                        for idInt in range (len(self.interventions)):
                            inter = self.interventions[idInt]
                            if self.listInterventionsDone[idInt] == False:
                                if inter.listBoat[b] == 1 and max(inter.beginning,season) + inter.nbJoursInt <= (season+SEASON-1):
                                    if inter.retrieve == False or (self.area == "ATL" and BASETECHNIQUENAVIREATL[p]) or (self.area == "MED" and BASETECHNIQUENAVIREMED[p]):
                                        if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ p ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time <= TIMEBOAT :
                                            matriceA[b][season][p].append(idInt)
                
                else:
                
                    for p in self.boats[b].escale.keys():
                        matriceA[b][season][p] = list()
                        
                        for idInt in range (len(self.interventions)):
                            inter = self.interventions[idInt]
                            if self.listInterventionsDone[idInt] == False:
                                if inter.listBoat[b] == 1 and max(inter.beginning,season) + inter.nbJoursInt <= (season+52):
                                    if inter.retrieve == False or self.boats[b].escale[p] == 1:
                                        if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ p ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time <= TIMEBOAT :
                                        
                                            matriceA[b][season][p].append(idInt)
        
        # Definir le programme linéaire
        
        prob=pu.LpProblem("",pu.LpMaximize)
        
        dictVarPortBat = dict()
        dictVarInt = dict()
        for inter in self.interventions:
            if self.listInterventionsDone[inter.idInt] == False:
            
                variable = pu.LpVariable(name='var_'+str(inter.idInt),cat='Binary')
                dictVarInt[inter.idInt] = variable

        for b in listBoatPossMove:
            for season in range (0,self.nbJours,SEASON):
                if self.boats[b].escale == None :
                    for p in listPossPort[b]:
                    
                        variable = pu.LpVariable(name='var_'+str(self.boats[b].nom)+"_"+p+"_"+str(season),cat='Binary')
                    
                        dictVarPortBat[(b,p,season)] = variable
                
                else:
        
                    for p in self.boats[b].escale.keys():
                        
                        variable = pu.LpVariable(name='var_'+str(self.boats[b].nom)+"_"+p+"_"+str(season),cat='Binary')
                    
                        dictVarPortBat[(b,p,season)] = variable
        
        
            
        
        # Contrainte 1            
        for b in listBoatPossMove:
            
            for season in range (0,self.nbJours, SEASON ):
                sumVariable = list()
                if self.boats[b].escale == None :
                    sumVariable += [(dictVarPortBat[(b,p,season)],1) for p in listPossPort[b] ]
                    
                    
                else:
                    sumVariable += [(dictVarPortBat[(b,p,season)],1) for p in self.boats[b].escale.keys() ]
                    
                prob += pu.LpConstraint(e=pu.LpAffineExpression( sumVariable ),sense=pu.LpConstraintEQ,name='',rhs = 1)
        
        #Contrainte 2
        
        
        for inter in self.interventions:
            if self.listInterventionsDone[inter.idInt] == False:
                sumVariable = list()
                sumVariable.append( (dictVarInt[inter.idInt],1) )
                
                
                for b in listBoatPossMove:
                    for season in range (0,self.nbJours, SEASON ):
                        if self.boats[b].escale == None :
                        
                            for p in listPossPort[b]:
                                if inter.idInt in matriceA[b][season][p]:
                                    sumVariable.append( (dictVarPortBat[(b,p,season)],-1) )
                        
                        else:
                            for p in self.boats[b].escale.keys():
                                if inter.idInt in matriceA[b][season][p]:
                                    sumVariable.append( (dictVarPortBat[(b,p,season)],-1) )
                    
                e = pu.LpAffineExpression( sumVariable )
                prob += pu.LpConstraint( e, sense=pu.LpConstraintLE,name='',rhs = 0)
            
            
        # Contrainte pour ne pas avoir deux bateaux sur un même port dans le même intervalle de temps.
        
        for indB in range (len(listBoatPossMove)):
            for indB2 in range (indB,len(listBoatPossMove)):
                b = listBoatPossMove[indB]
                b2 = listBoatPossMove[indB2]
                for season in range (0,self.nbJours, SEASON ):
                    
                    listPort = list()
                    
                    if self.boats[b].escale == None :
                        listPort = list( listPossPort[b] )
                        listPort.remove(self.boats[b].nomPortAtt)
                    
                    else:
                        listPort = list( self.boats[b].escale.keys() )
                        listPort.remove(self.boats[b].nomPortAtt)
                        
                    listPort2 = list()
                    
                    for p in listPort:
                        if self.boats[b2].escale == None:
                            if p in listPossPort[b2] and p != self.boats[b2].nomPortAtt:
                                listPort2.append(p)
                        else:
                            if p in self.boats[b2].escale.keys():
                                listPort2.append(p)
                        
                    for p in listPort2:
                        
                        lp = pu.LpAffineExpression( [ (dictVarPortBat[(b,p,season)],1) , (dictVarPortBat[(b2,p,season)],1) ] )
                        
                        prob += pu.LpConstraint(e = lp, sense = pu.LpConstraintLE, name = '', rhs = 1)
                      
        # Borne sur le nombre d'interventions traitées par un bateau par port.
        
        
#        for b in listBoatPossMove:
#            for day in range (self.nbJours):
#                if matriceDay[b][day] > 0:
#                    if self.boats[b].escale == None :
#                        for p in listPossPort[b]:
#                            sumVariable = list()
#                            requiredDay = 0
#                            totalHour = 0
#                            for idInt in matriceA[b][day][p]:
#                                inter = self.interventions[idInt]
#                                sumVariable.append( (dictVarInt[idInt],1) )                 
#                            
#                                if inter.reste or inter.double:
#                                    requiredDay += inter.nbJoursInt
#                                    
#                                else:
#                                    totalHour += inter.time
#                            requiredDay += math.ceil(totalHour/14)
#                            if requiredDay > 0:
#                                print(pu.LpAffineExpression( sumVariable ))
#                                input()
#                                prob += pu.LpConstraint(e=pu.LpAffineExpression( sumVariable ),sense=pu.LpConstraintLE,name='',rhs = math.ceil( matriceDay[b][day]*len(matriceA[b][day][p])/requiredDay ))        
#                                
#                    
#                    else:
#                        
#                        for p in self.boats[b].escale.keys():
#                            sumVariable = list()
#                            requiredDay = 0
#                            totalHour = 0
#                            for idInt in matriceA[b][day][p]:
#                                inter = self.interventions[idInt]
#                                sumVariable.append( (dictVarInt[idInt],1) )                 
#                            
#                                if inter.reste or inter.double:
#                                    requiredDay += inter.nbJoursInt
#                                    
#                                else:
#                                    totalHour += inter.time
#                            requiredDay += math.ceil(totalHour/14)
#                            if requiredDay > 0:
#                                prob += pu.LpConstraint(e=pu.LpAffineExpression( sumVariable ),sense=pu.LpConstraintLE,name='',rhs = math.ceil( matriceDay[b][day]*len(matriceA[b][day][p])/requiredDay ))
                
        
        # Objectif
        
        sumVariable = list()
        for idInt in range (len(self.interventions)):
            if self.listInterventionsDone[idInt] == False:
                sumVariable.append( (dictVarInt[idInt], self.interventions[idInt].nbJoursInt) )
        
        for b in listBoatPossMove:
            for season in range (0,self.nbJours, SEASON ):
            
                if self.boats[b].escale == None :
                    for p in listPossPort[b]:
                        if p != self.boats[b].nomPortAtt:
                            sumVariable.append( (dictVarPortBat[(b,p,season)], -1/2) )
                        
                else:
                    for p in self.boats[b].escale.keys():
                        if p != self.boats[b].nomPortAtt:
                            sumVariable.append( (dictVarPortBat[(b,p,season)], -1/2) )                    
                            
                    
                    
        prob.setObjective(pu.LpAffineExpression( sumVariable ) )
        
        pu.solvers.GUROBI_CMD(path=None, keepFiles=0, mip=1, msg=0, options=[]).actualSolve(prob)
        print("Status:", pu.LpStatus[prob.status])
        
        positionBoatPerSeason = dict()
        for b in listBoatPossMove:
            positionBoatPerSeason[b] = dict()
            for season in range (0,self.nbJours, SEASON ):
                 
                if self.boats[b].escale == None :
                
                    for p in listPossPort[b]:
                    
                        if dictVarPortBat[(b,p,season)].varValue == 1:
                            positionBoatPerSeason[b][season] = p
                            
                
                else:
                    
                    for p in self.boats[b].escale.keys():
                                    
                        if dictVarPortBat[(b,p,season)].varValue == 1:
                            positionBoatPerSeason[b][season] = p
            self.boats[b].initiateListPosition(positionBoatPerSeason[b])
                      
#        for idInt in range (len(self.interventions)):
#            if self.listInterventionsDone[idInt] == False:
#                print(idInt, dictVarInt[idInt].varValue,self.interventions[idInt].beginning, self.interventions[idInt].end )
#                print(self.interventions[idInt].nbJoursInt)
#                print(self.interventions[idInt].nomClosestPort)

        print(positionBoatPerSeason)
        
        
        
        
        return

############################################################# ROUTE DESIGN ##############################################################################
    def compute_time_route_simple(self,departure,arrival,boat,route):
        """
        """
        if len(route) ==0:
            return self.distanceMatricePP[boat.indMatrice][ self.ports[departure].id[boat.indMatrice]][self.ports[arrival].id[boat.indMatrice]]/boat.vitesse
        distance = 0 
        
        distance += self.distanceMatriceBP[boat.indMatrice][ self.interventions[route[0]].numLine[boat.indMatrice] ][ self.ports[departure].id[boat.indMatrice] ]
        distance += self.distanceMatriceBP[boat.indMatrice][ self.interventions[route[-1]].numLine[boat.indMatrice] ][ self.ports[arrival].id[boat.indMatrice] ]
        
        for ind in range (0,len(route)-1):
            distance += self.distanceMatriceBB[boat.indMatrice][ self.interventions[route[ind]].numLine[boat.indMatrice] ][ self.interventions[route[ind+1]].numLine[boat.indMatrice] ]
        
        time = (distance / boat.vitesse) + sum([self.interventions[idInt].time for idInt in route ])
        
        return time

    def compute_time_route_v2(self,departure,arrival,insertPos,idInt,boat,route):
        """
        Version 2 de la fonction compute_distance qui est plus lente mais recalcul l'ensemble de la route.
        """
        time = 0
        
        if len(route)==0:
            return (self.distanceMatriceBP[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice]][self.ports[departure].id[boat.indMatrice]] + self.distanceMatriceBP[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice]][self.ports[arrival].id[boat.indMatrice]])/boat.vitesse + self.interventions[idInt].time
            

        
            
        time += self.interventions[idInt].time 
        
        for ind in range (len(route)):
            time += self.interventions[ route[ind] ].time
        distance = 0    
        if insertPos != 0 and insertPos != len(route):
            distance += self.distanceMatriceBP[boat.indMatrice][ self.interventions[route[0]].numLine[boat.indMatrice] ][ self.ports[departure].id[boat.indMatrice] ]
            distance += self.distanceMatriceBP[boat.indMatrice][ self.interventions[route[-1]].numLine[boat.indMatrice] ][ self.ports[arrival].id[boat.indMatrice] ]
            
            for ind in range (insertPos-1):
                distance += self.distanceMatriceBB[boat.indMatrice][ self.interventions[route[ind]].numLine[boat.indMatrice] ][ self.interventions[route[ind+1]].numLine[boat.indMatrice] ]
                
            distance += self.distanceMatriceBB[boat.indMatrice][ self.interventions[route[insertPos-1]].numLine[boat.indMatrice] ][ self.interventions[idInt].numLine[boat.indMatrice] ]
            
            distance += self.distanceMatriceBB[boat.indMatrice][ self.interventions[route[insertPos]].numLine[boat.indMatrice] ][ self.interventions[idInt].numLine[boat.indMatrice] ]
            
            for ind in range (insertPos,len(route)-1):
                distance += self.distanceMatriceBB[boat.indMatrice][ self.interventions[route[ind]].numLine[boat.indMatrice] ][ self.interventions[route[ind+1]].numLine[boat.indMatrice] ]
                
        elif insertPos == 0:
            
            distance += self.distanceMatriceBP[boat.indMatrice][ self.interventions[idInt].numLine[boat.indMatrice] ][ self.ports[departure].id[boat.indMatrice] ]
            
            distance += self.distanceMatriceBB[boat.indMatrice][ self.interventions[idInt].numLine[boat.indMatrice] ][ self.interventions[route[0]].numLine[boat.indMatrice] ]
            
            for ind in range ( len(route)-1 ):
                distance += self.distanceMatriceBB[boat.indMatrice][ self.interventions[route[ind]].numLine[boat.indMatrice] ][ self.interventions[route[ind+1]].numLine[boat.indMatrice] ]

            distance += self.distanceMatriceBP[boat.indMatrice][ self.interventions[route[-1]].numLine[boat.indMatrice] ][ self.ports[arrival].id[boat.indMatrice] ]
            
        elif insertPos == len(route):
            distance += self.distanceMatriceBP[boat.indMatrice][ self.interventions[route[0]].numLine[boat.indMatrice] ][ self.ports[departure].id[boat.indMatrice] ]
            
            for ind in range ( len(route)-1 ):
                distance += self.distanceMatriceBB[boat.indMatrice][ self.interventions[route[ind]].numLine[boat.indMatrice] ][ self.interventions[route[ind+1]].numLine[boat.indMatrice] ]
            distance += self.distanceMatriceBB[boat.indMatrice][ self.interventions[idInt].numLine[boat.indMatrice] ][ self.interventions[route[-1]].numLine[boat.indMatrice] ]
            distance += self.distanceMatriceBP[boat.indMatrice][ self.interventions[idInt].numLine[boat.indMatrice] ][ self.ports[arrival].id[boat.indMatrice] ]
            
        time += (distance / boat.vitesse) 
        
        return time 

            
            
    
    
    # ATTENTION CETTE FONCTION NE MARCHE PAS CAR IMPRECISIONS
    def compute_time_route(self,departure,arrival,time,insertPos,idInt,boat,route):
        """
        route une liste de integer, les identifiants des interventions dans la route.
        """

        if len(route)==0:
            return (self.distanceMatriceBP[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice]][self.ports[departure].id[boat.indMatrice]] + self.distanceMatriceBP[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice]][self.ports[arrival].id[boat.indMatrice]])/boat.vitesse + self.interventions[idInt].time
    
        time += self.interventions[idInt].time
        if insertPos != 0 and insertPos != len(route):
            time -= (self.distanceMatriceBB[boat.indMatrice][self.interventions[route[insertPos-1]].numLine[boat.indMatrice] ][self.interventions[route[insertPos]].numLine[boat.indMatrice] ])/boat.vitesse
            time += (self.distanceMatriceBB[boat.indMatrice][self.interventions[route[insertPos-1]].numLine[boat.indMatrice] ][self.interventions[idInt].numLine[boat.indMatrice] ])/boat.vitesse
            time += (self.distanceMatriceBB[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice] ][self.interventions[route[insertPos]].numLine[boat.indMatrice] ])/boat.vitesse

        if insertPos == 0:
            time -= (self.distanceMatriceBP[boat.indMatrice][self.interventions[route[0]].numLine[boat.indMatrice] ][self.ports[departure].id[boat.indMatrice]])/boat.vitesse
            time += (self.distanceMatriceBP[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice] ][self.ports[departure].id[boat.indMatrice]])/boat.vitesse
            time += (self.distanceMatriceBB[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice] ][self.interventions[route[0]].numLine[boat.indMatrice] ])/boat.vitesse
            
        if insertPos == len(route):
            time -= (self.distanceMatriceBP[boat.indMatrice][self.interventions[route[-1]].numLine[boat.indMatrice] ][self.ports[arrival].id[boat.indMatrice]])/boat.vitesse
            print((self.distanceMatriceBP[boat.indMatrice][self.interventions[route[-1]].numLine[boat.indMatrice] ][self.ports[arrival].id[boat.indMatrice]])/boat.vitesse)
            time += (self.distanceMatriceBP[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice] ][self.ports[arrival].id[boat.indMatrice]])/boat.vitesse
            print((self.distanceMatriceBP[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice] ][self.ports[arrival].id[boat.indMatrice]])/boat.vitesse)
            time += (self.distanceMatriceBB[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice] ][self.interventions[route[-1]].numLine[boat.indMatrice] ])/boat.vitesse
            print((self.distanceMatriceBB[boat.indMatrice][self.interventions[idInt].numLine[boat.indMatrice] ][self.interventions[route[-1]].numLine[boat.indMatrice] ])/boat.vitesse)
        return time
        

    
    
    def greedy(self,departure,arrival,boat,listPossibleInterventions,route=[],timeRoute=0,timeAvailable = math.inf,stock = 0, tempsBat = TIMEBOAT):
        '''
        
        Demarre d'une liste route, calcul le temps de la route avec ajout de chaque intervention une à une , ajoute l'intervention qui fait 
        le moins augmenter le temps de la route.  stock la place libre sur le pont
        '''
        # Choisir la plus proche du départ
        # Le port a un id pour son rang dans la matrice, une balise a numLine
        listPossibleInterventionsDone = dict()
        for idInt in listPossibleInterventions:
            listPossibleInterventionsDone[idInt] = False
        closestInter = -1
        nbIntRetrieve = 0
##        for idInt in route:
##            if self.interventions[idInt].retrieve :
        finalTime = self.compute_time_route_simple(departure,arrival,boat,route)
 
        if len(route) != 0 and self.interventions[route[0]].double:
            indDeb = 1
            indFin = 0
        else:
            indDeb = 0
            indFin = 1
        
        while (closestInter != None):
            closestInter = None
            ind = 0
            time = math.inf
            for idInt in listPossibleInterventions:
                inter = self.interventions[idInt]
                if listPossibleInterventionsDone[idInt]==False:
                    if len(route)==0:
                        currTime = self.compute_time_route_v2(departure,arrival,0,idInt,boat,route)
                        if currTime < time and currTime <= tempsBat and currTime <= timeAvailable:
                            if inter.retrieve:
                                if nbIntRetrieve+1 <= stock:
                                    time = currTime
                                    closestInter = idInt
                                    indInsert = 0
                                else:
                                    #print(boat.nom,route,idInt,stock,nbIntRetrieve+1,departure,arrival)
                                    pass
                            else:
                                time = currTime
                                closestInter = idInt
                                indInsert = 0
                                            
                    else:
                    
                    
                        for node in range (indDeb,len(route)+indFin):
                            currTime = self.compute_time_route_v2(departure,arrival,node,idInt,boat,route)
                            if currTime < time and currTime <= tempsBat and currTime <= timeAvailable:
                                # Permet de limité à 2 le nombre de balises chargées sur le pont
                                if inter.retrieve:
                                    if nbIntRetrieve+1 <= stock:
                                        time = currTime
                                        closestInter = idInt
                                        indInsert = node
                                    else:
                                        #print(boat.nom,route,idInt,stock,nbIntRetrieve+1,departure,arrival)
                                        pass
                                else:
                                    time = currTime
                                    closestInter = idInt
                                    indInsert = node
                
                                    
                                
            
            if closestInter != None:
                if inter.retrieve:
                    nbIntRetrieve += 1
                route.insert(indInsert,closestInter)
                listPossibleInterventionsDone[closestInter] = True
                finalTime = time
                
                # Pas besoin de faire un 2-opt normalement
                #route = self.two_opt(departure,arrival,route,boat.vitesse,nbIter)
            else:

                break
            
        if len(route) == 0:
            # Aucunes interventions effectuées
            finalTime = self.distanceMatricePP[boat.indMatrice][self.ports[departure].id[boat.indMatrice]][self.ports[arrival].id[boat.indMatrice]]/boat.vitesse

        return route,finalTime,nbIntRetrieve
            


    def glouton(self,departure,arrival,boat,listPossibleInterventionsPriority,timeAvailable,stock = 0,route = [], tempsBat = TIMEBOAT):
    
        stockBeg = stock
        # On fait un premier cycle en ne prenant que les interventions prioritaires. 
        for i in range(len(listPossibleInterventionsPriority)):
            route,timeSpent,nbIntRetrieve = self.greedy(departure,arrival,boat,listPossibleInterventionsPriority[ i ],route,timeRoute=0,timeAvailable=timeAvailable,stock=stock, tempsBat = tempsBat)
            stock = stock-nbIntRetrieve
        
        return route, timeSpent, stockBeg - stock
        
    def compute_route_semaine(self,numBat,time,timeAvailable,departure = None, arrival = None, listInt = list(), check = True, stock = 0, route = [],tempsBat = TIMEBOATCAMPAGNE):
        if departure == None and arrival == None:
            departure = self.boats[numBat].nomPortAtt
            arrival = departure
            

            
        listPossibleInterventionsPriority = [[] for i in range (3)]
        for idInt in listInt:
            inter = self.interventions[idInt]
            if (inter.beginning <= time and inter.end >= time) or (inter.beginning >= time and inter.end < time) or  (inter.beginning < time and inter.end >= time ):
                if (self.distanceMatriceBP[ self.boats[numBat].indMatrice ][ inter.numLine[self.boats[numBat].indMatrice] ][ self.ports[departure].id[self.boats[numBat].indMatrice ] ] + self.distanceMatriceBP[ self.boats[numBat].indMatrice ][ inter.numLine[self.boats[numBat].indMatrice] ][ self.ports[arrival].id[self.boats[numBat].indMatrice] ]) /self.boats[numBat].vitesse <= tempsBat:
                    if self.interventions[idInt].listBoat[numBat] == 1 :
                        listPossibleInterventionsPriority[self.interventions[idInt].type-1].append(idInt)
                        
        #print(listPossibleInterventionsPriority)
        #input()
        route,timeSpent,nbIntRetrieve = self.glouton(departure,arrival,self.boats[numBat],listPossibleInterventionsPriority,timeAvailable,stock = stock,route = [],tempsBat = tempsBat)
        if check:
            for ind in range (len(self.interventions)):
                if self.interventions[ind].idInt in route:
                    self.listInterventionsDone[ind] = True
        return route,timeSpent,nbIntRetrieve    
        
    def compute_route(self,numBat,time,timeAvailable, departure = None , arrival = None, check = True, stock = 0,route = [],tempsBat = TIMEBOAT):
        # Determiner le port de départ et de retour au jour donné.
        if departure == None:
            departure = self.boats[numBat].nomPortAtt
            arrival = departure
        listPossibleInterventions = list(self.boats[numBat].listInterventionsPerDay[time])
            
        listPossibleInterventionsPriority = [[] for i in range (208)]
        
        #print("bastia",self.ports["BASTIA"].listInterventionsPerDay[time])
        #print("bon",self.ports["BONIFACIO"].listInterventionsPerDay[time])
        #print("ajac",self.ports["AJACCIO"].listInterventionsPerDay[time] )
        #print(listPossibleInterventions)
        
        for idInt in listPossibleInterventions:
            if self.listInterventionsDone[idInt]==False :
                if self.interventions[idInt].listBoat[numBat]==1 :
                    listPossibleInterventionsPriority[self.interventions[idInt].beginning].append(idInt)
        #Construit une route pour le bateau au temps time
        #print(self.boats[numBat].nom,listPossibleInterventionsPriority)
        route,timeSpent,nbIntRetrieve = self.glouton(departure,arrival,self.boats[numBat],listPossibleInterventionsPriority,timeAvailable,stock = stock,route = route,tempsBat = tempsBat)

        minRoute = list()

        minTime = math.inf
        if len(route) <= 8 and len(route) >= 2 and self.interventions[route[0]].double == False:
            # Tester toutes les possibilitées.
            for possRoute in itertools.permutations(route) :
                time = self.compute_time_route_simple(departure,arrival,self.boats[numBat],possRoute)
                if time < minTime:
                    minTime = time
                    minRoute = list(possRoute)
        #print("Temps dépensé:",timeSpent)

        if minTime < timeSpent:
            print(minTime,timeSpent)
            print(minRoute,route)
#            input()

        if check:
            for ind in range (len(self.interventions)):
                if self.interventions[ind].idInt in route:
                    self.listInterventionsDone[ind] = True


        return route,timeSpent,nbIntRetrieve

        
    def longRoute(self,solution):
        """
        Algorithme pour inserer les routes de 2 jours des bateaux habitables dans la solution,
        """
#        inter = self.interventions[371]
#        print(inter.nomEsm,inter.type,inter.typeEsm,inter.listBoat,inter.beginning,inter.end)   
#        input()     
        # Trier les interventions en fonction des bateaux qui peuvent les faire
        listIntPerBoat = [list() for i in range (len(self.boats))]
        listInt = list()
        for idInt in range (len(self.listInterventionsDone)) :

            inter = self.interventions[idInt]
            if self.listInterventionsDone[idInt] == False and inter.double == False and inter.reste == False:
                boolPossible = False
                for b in self.numBoatConfig:
                        boat = self.boats[b]
                        portAtt = self.ports[boat.nomPortAtt]
                        
                        if self.boats[b].habitable and self.interventions[idInt].listBoat[b] == 1 and self.ports[inter.nomClosestPort].indZone >= self.ports[self.boats[b].nomPortAtt].indZone :
                            
                            listIntPerBoat[b].append(idInt)
                            if idInt not in listInt:
                                listInt.append(idInt)
                            boolPossible = True
                if boolPossible == False:          
                    inter = self.interventions[idInt]
                    print(inter.nomEsm,inter.type,inter.typeEsm,inter.listBoat,inter.beginning,inter.end)   
                    
        # Liste des disponibilités pour chaque mois
        disponibilityMonth = dict()
        possibleAmeliorationMonth = dict()
        possibleBoat = dict()
        nbLongRouteReste = dict() 
        for b in range (len(self.boats)):
            if self.boats[b].habitable:
                possibleBoat[b] = True
                nbLongRouteReste[b] = MAXLONGROUTE
                disponibilityMonth[b] = list()
                possibleAmeliorationMonth[b] = list()
                # On suppose que 1 mois est composé de 4 semaines, on a donc 13 mois.
                for month in range (nbJours//(4*DAYWEEK)):
                    disponibilityMonth[b].append(ROADPERMONTH)
                    possibleAmeliorationMonth[b].append(True)
        cptTries = 0

        while( cptTries < 50 ):
#            print("La liste:",len(listInt),listInt)
            change = False
            # Prendre le bateau ayant le moins travaillé
            for b in self.numBoatConfig:
                if self.boats[b].habitable:
                    if len(listIntPerBoat[b]) > 0 :
                        if nbLongRouteReste[b] > 0 and possibleBoat[b] :
                            
                            if change == False:
                                choosenBoatNum = b
                                
                            if solution.evalBatNbJour[b] >= solution.evalBatNbJour[choosenBoatNum] :
                                choosenBoatNum = b
                                change = True
                    
            if change == False:
                # Pas de bateaux trouvés.
                break
                
            listInterventionsPerMonth = [list() for i in range (nbJours//(4*DAYWEEK))]
            
            for month in range (nbJours//(4*DAYWEEK)):
                
                for idInt in listIntPerBoat[choosenBoatNum]:
                    inter = self.interventions[idInt]
                    if (inter.beginning <= month*4*DAYWEEK and inter.end >= (month)*4*DAYWEEK ) or (inter.beginning >= month*4*DAYWEEK and inter.end < (month+1)*4*DAYWEEK) or  (inter.beginning < (month+1)*4*DAYWEEK and inter.end >= (month)*4*DAYWEEK ) and inter.reste == False and inter.double == False:
                        # Atteignable en deux jours par le bateau.
                        boat = self.boats[choosenBoatNum]
                        if month in boat.dictMonth.keys():
                            portMonth =  boat.dictMonth[month]
                        else:
                            portMonth = boat.nomPortAtt 
                        if (self.distanceMatriceBP[boat.indMatrice][ inter.numLine[boat.indMatrice] ][ self.ports[portMonth].id[boat.indMatrice] ]*2)/boat.vitesse + inter.time < TIMEBOATLONG:
                        
                            listInterventionsPerMonth[month].append(idInt)                         
            # Tri de la liste des mois préférables
            listMonthNbInt = list()
            for month in range (nbJours//(4*DAYWEEK)):
                listMonthNbInt.append( (month,len(listInterventionsPerMonth[month]) ))
                
            listMonthNbInt.sort(key = sortSecond ,reverse = True)
            
            
            choosenMonth = 0
            change = False
            for month,nbInt in listMonthNbInt:
                if disponibilityMonth[choosenBoatNum][month] >0 and possibleAmeliorationMonth[choosenBoatNum][month]:
                    # Choix en fonction de la liste des interventions qui peuvent être faites ce mois mais egalement le temps disponible.
                    choosenMonth = month
                    
                    
                    boat = self.boats[choosenBoatNum]
                    if choosenMonth in self.boats[choosenBoatNum].dictMonth.keys():
                        departure = self.boats[choosenBoatNum].dictMonth[choosenMonth]
                        arrival = departure
                    
                    else:
                        departure = self.boats[choosenBoatNum].nomPortAtt
                        arrival =   departure
                    
                    
                    listIntPerDay = dict()
                    for day in range (choosenMonth * 16,(choosenMonth+1) * 16 -2):
                        if (day // DAYWEEK) != ((day+1) // DAYWEEK) or boat.disponibility[day]==False or boat.disponibility[day+1]==False:
                            # Si ce ne sont pas des jours de la même semaine.
                            continue
                            
                        if len(solution.routeBatPerDay[choosenBoatNum][day]) > 0:
                            if self.interventions[solution.routeBatPerDay[choosenBoatNum][day][0]].double :
                                continue 
                        if len(solution.routeBatPerDay[choosenBoatNum][day+1]) > 0:
                            if self.interventions[solution.routeBatPerDay[choosenBoatNum][day+1][0]].double:
                                continue
                            
                        if (solution.evalBatTimeRoute[choosenBoatNum][day] == 0 and solution.evalBatNbJour[choosenBoatNum] <1) or (solution.evalBatTimeRoute[choosenBoatNum][day +1] == 0 and solution.evalBatNbJour[choosenBoatNum] <1) or (solution.evalBatTimeRoute[choosenBoatNum][day] == 0 and solution.evalBatTimeRoute[choosenBoatNum][day +1] == 0 and solution.evalBatNbJour[choosenBoatNum] <2):
                            continue                    
                         
                        # Ici on a trouvé 2 jours disponibles, on essaye de faire les routes.

                        
                        route,timeSpent,nbIntRetrieve = self.compute_route_semaine(choosenBoatNum,day,boat.maxHourWeek - solution.evalBatTimeSemaine[choosenBoatNum][ day//DAYWEEK ] + solution.evalBatTimeRoute[choosenBoatNum][day]+solution.evalBatTimeRoute[choosenBoatNum][day+1] , departure , arrival , listInterventionsPerMonth[choosenMonth]+ solution.routeBatPerDay[choosenBoatNum][day]+solution.routeBatPerDay[choosenBoatNum][day+1],stock = boat.stock,tempsBat = TIMEBOATLONG, check = False)

                        possible = True
                        for idInt in solution.routeBatPerDay[choosenBoatNum][day]+solution.routeBatPerDay[choosenBoatNum][day+1]:
                            if idInt not in route:
                                possible = False
                                break
                        cptInt = 0
                        if possible:
                            for idInt in route:
                                if idInt in listInterventionsPerMonth[choosenMonth]:
                                    cptInt += 1
                            listIntPerDay[day] = cptInt
                    
                    maxInt = 0
                    bestDay = None
                    for day in listIntPerDay.keys():
                        if listIntPerDay[day] > maxInt: 
                            maxInt = listIntPerDay[day]
                            bestDay = day
                
                    if maxInt == 0:
                        possibleAmeliorationMonth[choosenBoatNum][choosenMonth]
                        continue
                        
                    disponibilityMonth[choosenBoatNum][choosenMonth] -= 1
                    nbLongRouteReste[choosenBoatNum] -= 1

                    route,timeSpent,nbIntRetrieve = self.compute_route_semaine(choosenBoatNum,bestDay,boat.maxHourWeek - solution.evalBatTimeSemaine[choosenBoatNum][ bestDay//DAYWEEK ] + solution.evalBatTimeRoute[choosenBoatNum][bestDay]+solution.evalBatTimeRoute[choosenBoatNum][bestDay+1] , departure , arrival , listInterventionsPerMonth[choosenMonth]+ solution.routeBatPerDay[choosenBoatNum][bestDay]+solution.routeBatPerDay[choosenBoatNum][bestDay+1],stock = boat.stock,tempsBat = TIMEBOATLONG, check = True)
                    solution.addRoute(choosenBoatNum,bestDay,route,departure,arrival)
                    solution.addRoute(choosenBoatNum,bestDay+1,[],departure,arrival)
                    solution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP,batType = [1,2,3,4]) 
                    
                    for idInt in route:
                        if idInt in listInt: 
                            listInt.remove(idInt)
                            for b in range (len(listIntPerBoat)):
                                if idInt in listIntPerBoat[b]:
                                    listIntPerBoat[b].remove(idInt)

                    change = True
                    break
            if change == False:
                possibleBoat[choosenBoatNum] = False
                            
            

            
            
    def possiblePlacement(self,idInt,day,boatNum):
        """
        return True si il est possible de placer l'intervention d'id IdInt sur le bateau d'idi boatNumau temps day, False sinon
        """       
        inter = self.interventions[idInt]
        dayPossible = True
        for dayInt in range (inter.nbJoursInt):
            if self.boats[boatNum].disponibility[day + dayInt] == False or solution.evalBatTimeRoute[boatNum][day + dayInt] != 0 :
                dayPossible = False
                break

        if dayPossible:
                
            firstWe = day // DAYWEEK
            lastWe =  (day + inter.nbJoursInt) // DAYWEEK
            dayBeg = (firstWe + 1) * DAYWEEK - day
            dayEnd = (day + inter.nbJoursInt) - (lastWe * DAYWEEK)
        
            temps = (self.distanceMatriceBP[ self.boats[boatNum].indMatrice ][ inter.numLine[ self.boats[boatNum].indMatrice ] ][ self.ports[ self.boats[boatNum].listPosition[day] ].id[self.boats[boatNum].indMatrice] ]/self.boats[boatNum].vitesse)*2 + inter.time 
            
            if temps*dayBeg + solution.evalBatTimeSemaine[boatNum][firstWe] < self.boats[boatNum].maxHourWeek and temps*dayEnd + solution.evalBatTimeSemaine[boatNum][lastWe] < self.boats[boatNum].maxHourWeek:
                
                dayPossible = True
            else:
                dayPossible = False
                
        return dayPossible
        
    def attributeFixe(self, solution, batType = [1,2,3,4]):
    
        interventionsRD = list()
        dictAttributeInter = dict()
        dictAttributeBoat = dict()
        dictDayBoat = dict()
        cptImpossible = 0
        
        for b in self.numBoatConfig:
            dictAttributeBoat[b] = list()
            dictDayBoat[b] = self.boats[b].maxDayYear
            print(b,self.boats[b].nom)
        
        for idInt in range (len(self.interventions)):
        
            if (self.interventions[idInt].double or self.interventions[idInt].reste)  :
                interventionsRD.append( (idInt, self.interventions[idInt].nbJoursInt, self.interventions[idInt].score))
                dictAttributeInter[idInt] = False 
        interventionsRD.sort(key = lambda x:(-x[2], x[1]),reverse = True)
        dictIntPossBoat = dict()
        dictIntBoatTime = dict()
        
        for ob in interventionsRD:   
            idInt = ob[0]
            inter = self.interventions[idInt]
            dictIntBoatTime[idInt] = dict()
            dictIntPossBoat[idInt] = list()
            for b in range (len(inter.listBoat)):
                    if inter.listBoat[b] == 1:  
                        temps = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                        if inter.double:
                            temps += inter.time + 3
                        
                        if temps <= TIMEBOAT and  ( temps * min(inter.nbJoursInt,4) <= self.boats[b].maxHourWeek):
                            
                            dictIntPossBoat[idInt].append(b)
                        dictIntBoatTime[idInt][b] = temps
        allDone = False
        existUniqBoat = True
        
        solution.prepareToPlot()
        
        while (allDone == False):
            while(existUniqBoat):
            
                for ob in interventionsRD:
                    idInt = ob[0]   
                    if dictAttributeInter[idInt] == False:
                        if len(dictIntPossBoat[idInt]) == 1:
                            if dictDayBoat[dictIntPossBoat[idInt][0]] > ob[1]:
                                dictAttributeInter[idInt] = True
                                dictAttributeBoat[dictIntPossBoat[idInt][0]].append(idInt)
                                dictDayBoat[dictIntPossBoat[idInt][0]] -= ob[1]

                                
                            else:
                                dictIntPossBoat[idInt] = list()
                                dictAttributeInter[idInt] = True
                                cptImpossible += 1
                                
                        if len(dictIntPossBoat[idInt]) == 0:
                            dictIntPossBoat[idInt] = list()
                            dictAttributeInter[idInt] = True
                            cptImpossible += 1
                
                for b in self.numBoatConfig:
                    for ob in interventionsRD:
                        idInt = ob[0]
                        if dictAttributeInter[idInt] == False:
                            if dictDayBoat[b] < ob[1]:

                                if b in dictIntPossBoat[idInt]:
                                    dictIntPossBoat[idInt].remove(b)
                                    
                                
                existUniqBoat = False                           
                for ob in interventionsRD:
                    idInt = ob[0] 
                    if dictAttributeInter[idInt] == False:
                          
                        if len(dictIntPossBoat[idInt]) <= 1:
                            existUniqBoat = True
                            break
                
            for ob in interventionsRD:
                idInt = ob[0]   
                if dictAttributeInter[idInt] == False:
                    dictBatUse = dict()
                    bestBoat = None
                    minDay = - math.inf
                    for b in dictIntPossBoat[idInt]:
                        dictBatUse[b] = dictDayBoat[b]
                        # compter le nombre d'interventions qu'il peut faire
                        
                        for ob in interventionsRD:
                            idInt2 = ob[0]                               
                            if dictAttributeInter[idInt2] == False:
                                if b in dictIntPossBoat[idInt2]:
                                    dictBatUse[b] -= ob[1]
                        if dictBatUse[b] > minDay and dictDayBoat[b] > ob[1]:
                             bestBoat = b
                    if bestBoat == None:
                        cptImpossible += 1
                        dictAttributeInter[idInt] = True
#                        print("retire")
#                        print(dictIntPossBoat[idInt])
#                        input()
                    else:
                        dictAttributeInter[idInt] = True
                        dictAttributeBoat[bestBoat].append(idInt)
                        dictDayBoat[bestBoat] -= ob[1]


                    break
            allDone = True
            
            for ob in interventionsRD:
                idInt = ob[0]
                if dictAttributeInter[idInt] == False:     
                   allDone = False
                   break                      
            existUniqBoat = True
        print(dictAttributeBoat,cptImpossible)
        allInt = list()
        for b in self.numBoatConfig:
            allInt += dictAttributeBoat[b]
            print(b,self.boats[b].nom, dictDayBoat[b])
        cptDay = 0
        cptInt = 0
        cptDayBre = 0
        notInt = list()
        for ob in interventionsRD:
            if ob[0] not in allInt:
                if self.interventions[ob[0]].numEsm[:2] == "2B" or self.interventions[ob[0]].numEsm[:2] == "2A"  :
                    self.interventions[ob[0]].printInt()
                    input()
                    cptDayBre += self.interventions[ob[0]].nbJoursInt 
                cptDay += self.interventions[ob[0]].nbJoursInt
                notInt.append(self.interventions[ob[0]].numEsm)
                cptInt += 1
        
        solution.plotNotPossible(notInt )
        

        
        print("cptInt:",cptInt)
        print("cptDay:",cptDay)
        print("cptDayBre:",cptDayBre)
        
        for b in self.numBoatConfig:
            print(self.boats[b].nom,dictDayBoat[b],self.boats[b].maxDayYear)
        
        input()
        
        
        # TODO deplacement ICI (boatPositiontest)
        
    
        # Attribuer une position à l'intervention sachant qu'on l'a deja attribuée à un bateau.
        
        
        for b in self.numBoatConfig:
        
            dictAttributeBoat[bestBoat].append(idInt)
            
            listSortIntBoat = list()
            for idInt in dictAttributeBoat[bestBoat]:
                listSortIntBoat.append( (idInt,self.interventions[idInt].nbJoursInt) )
            listSortIntBoat.sort(key = sortSecond ,reverse = True)
                
            
            # Trouver une position pour l'intervention.
            
            for idInt,nbJours in listSortIntBoat:
                inter = self.interventions[idInt]
                
                cptIntDay = math.inf
                choosenDay = None
                for day in range (inter.beginning,inter.end):
                    if self.possiblePlacement(idInt,day,b):
                    
                        cptIntNbjoursPoss = 0
                        for idInt2,nbJours2 in listSortIntBoat:
                            inter2 = self.interventions[idInt2]
                            possibly = True
                            
                            for day2 in range (day+1 - nbJours2,day+1):
                                if self.possiblePlacement(idInt2,day2,b):
                                    possibly = True
                            if possibly:
                                cptIntNbjoursPoss += nbJours2 
                                
                        if cptIntNbjoursPoss < cptIntDay:
                            cptIntDay = cptIntNbjoursPoss
                            choosenDay = day
                if choosenDay == None:
                    # On a pas réussi à placer l'interventions
                    continue
                else:
                    # On place l'interventions
                    self.listInterventionsDone[inter.idInt] = True
                
                
                    for d in range (choosenDay,choosenDay+inter.nbJoursInt):
                        if inter.reste:
                            self.boats[choosenBoat].disponibility[d] = False
                            solution.addRoute(choosenBoat,d,[inter.idInt])       
                        else:
                            solution.addRoute(choosenBoat,d,[inter.idInt,inter.idInt])
                    
                    # Les evaluations des semaines qui ont changées
                    for we in range (choosenDay//DAYWEEK, ((choosenDay+inter.nbJoursInt)//DAYWEEK )+1):
                        solution.evaluateWeek(b,we,self.distanceMatriceBB[self.boats[b].indMatrice],self.distanceMatriceBP[self.boats[b].indMatrice],self.distanceMatricePP[self.boats[b].indMatrice])
            
                    # regarder combien d'interventions on arrive a placer.
                    
        
        return                    
                                    
          
    def placement2(self,solution,batType = [1,2,3,4], listInt = list(), boatNum = None, timeWindow = list()):
        """
        Placement des interventions "reste" (qui durent plus d'un jour). apres avoir placé les 
        autres interventions.
        
        """  
        
        if listInt == list():
            listInt = [idInt for idInt in range (len(self.interventions))]
        
        if timeWindow == list():
            timeWindow = [0,self.nbJours-1]
        interventionsRD = list()
        
        for idInt in listInt:

            if (self.interventions[idInt].double or self.interventions[idInt].reste) and self.interventions[idInt].notPossible == False and self.listInterventionsDone[idInt] == False :
                inter = self.interventions[idInt]
                
                interventionsRD.append( (idInt, self.interventions[idInt].nbJoursInt,  self.interventions[idInt].score))
            
            
        # Tri de la liste d'interventions considérée.

        interventionsRD.sort(key = lambda x : (-x[2], -x[1]))

        
        dictIntPossBoat = dict()


        for b in self.numBoatConfig:
            if self.boats[b].type in batType:
                dictIntPossBoat[b] = list()

                for ind in range (len(interventionsRD)):

                    inter = self.interventions[interventionsRD[ind][0]]


                    if inter.listBoat[b] == 1 and (boatNum == None or b == boatNum):

                        if self.boats[b].habitable or self.boats[b].type == 2 :
                            for day in range (timeWindow[0],timeWindow[1]+1):
                                temps = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].listPosition[day] ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                                if inter.double:
                                    temps += inter.time + 3

                                if temps <= TIMEBOAT and  ( temps * min(inter.nbJoursInt,4) <= self.boats[b].maxHourWeek):

                                    dictIntPossBoat[b].append(interventionsRD[ind])
                                    break

                        else:
                            temps = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                            if inter.double:
                                temps += inter.time + 3

                            if temps <= TIMEBOAT and  ( temps * min(inter.nbJoursInt,4) <= self.boats[b].maxHourWeek):
                                dictIntPossBoat[b].append(interventionsRD[ind])


        potentialChargePerBoat = dict()


        for b in dictIntPossBoat.keys():
            sumNbJours = 0
            for idInt,nbJours,score in dictIntPossBoat[b]:
                sumNbJours += nbJours
            potentialChargePerBoat[b] = sumNbJours

        for ind in range (len(interventionsRD)):
            
            inter = self.interventions[interventionsRD[ind][0]]
            
            possibleBoatPerType = dict()
            
            for i in batType:
                possibleBoatPerType[i] = list()
            
            possibleDayPerBoat = dict()
            for b in range (len(inter.listBoat)):
                
                if self.boats[b].type in batType:
            
                    if inter.listBoat[b] == 1 and (boatNum == None or b == boatNum):
                    
                        if self.boats[b].habitable or self.boats[b].type == 2 :
                            possibleDayPerBoat[b] = list()
                            possibleBoat = False
                            for day in range (timeWindow[0],timeWindow[1]+1):
                                temps = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].listPosition[day] ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                                if inter.double:
                                    temps += inter.time + 3
                                
                                if temps <= TIMEBOAT and  ( temps * min(inter.nbJoursInt,4) <= self.boats[b].maxHourWeek):
                                
                                    if possibleBoat==False:
                                        possibleBoatPerType[self.boats[b].type].append(b)
                                        possibleBoat = True
                                    possibleDayPerBoat[b].append(day)
                                
                                
                        else:
                            temps = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                            if inter.double:
                                temps += inter.time + 3
                            
                            if temps <= TIMEBOAT and  ( temps * min(inter.nbJoursInt,4) <= self.boats[b].maxHourWeek):
                                
                                if self.boats[b].type in batType:
                                    possibleBoatPerType[self.boats[b].type].append(b)
            
            
            
            
            
                            
            sumLen = 0
            for i in possibleBoatPerType.keys():
                sumLen += len(possibleBoatPerType[i])
            
            if sumLen == 0:
                continue
            notPosed = True
            
            listTabuBoat = deque()  

            while (notPosed):
                nbDaysLeft = -math.inf
                choosenBoat = None
#                choosenBoat = random.choice(possibleVedette+possibleNavire)
#                if choosenBoat not in listTabuBoat:
#                    listTabuBoat.append(choosenBoat)
#                else:
#                    break
#                if solution.evalBatNbJour[choosenBoat] < inter.nbJoursInt:
#                    continue
                for tyB in batType:
                    for b in possibleBoatPerType[tyB]:

                        
                        
                        
                        if solution.evalBatNbJour[b]-potentialChargePerBoat[b] > nbDaysLeft and solution.evalBatNbJour[b] > inter.nbJoursInt and b not in listTabuBoat:
                            choosenBoat = b
                            nbDaysLeft = solution.evalBatNbJour[b]

                    if choosenBoat != None: 
                        break
                        
                if choosenBoat != None: 
                    listTabuBoat.append(choosenBoat)
                    
                else:
                    # On a pas réussi a placer l'intervention.
                    break

                daysPossible = list()
                for day in range (max(timeWindow[0],inter.beginning), min(timeWindow[1]+1,inter.end + 1 ) - inter.nbJoursInt):
                    dayPossible = True
                    
                    if (self.boats[choosenBoat].habitable or self.boats[b].type == 2 ) and day not in possibleDayPerBoat[choosenBoat]:
                        # Si le mois est impossible.
                        continue
                    
                    
                    for dayInt in range (inter.nbJoursInt):
                        if self.boats[choosenBoat].disponibility[day + dayInt] == False or solution.evalBatTimeRoute[choosenBoat][day + dayInt] != 0 :
                            dayPossible = False
                            break

                    if dayPossible:
                            
                        firstWe = day // DAYWEEK
                        lastWe =  (day + inter.nbJoursInt) // DAYWEEK
                        dayBeg = (firstWe + 1) * DAYWEEK - day
                        dayEnd = (day + inter.nbJoursInt) - (lastWe * DAYWEEK)
                    
                        temps = (self.distanceMatriceBP[ self.boats[choosenBoat].indMatrice ][ inter.numLine[ self.boats[choosenBoat].indMatrice ] ][ self.ports[ self.boats[choosenBoat].listPosition[day] ].id[self.boats[choosenBoat].indMatrice] ]/self.boats[choosenBoat].vitesse)*2 + inter.time 
                        
                        if temps*dayBeg + solution.evalBatTimeSemaine[choosenBoat][firstWe] < self.boats[choosenBoat].maxHourWeek and temps*dayEnd + solution.evalBatTimeSemaine[choosenBoat][lastWe] < self.boats[choosenBoat].maxHourWeek:
                            
                            daysPossible.append(day)
                                
                                
                # Choisir un jour dans la liste dayPossible avec le moins d'amplitude.
                
                if len(daysPossible) == 0:
                    continue
                    
                bestDays = list()
                minDayAfter = math.inf
                for day in daysPossible:
                
                    nbDaysAfter = 0
                    dayAfter = day + inter.nbJoursInt
                    while (dayAfter <= timeWindow[1] and solution.evalBatTimeRoute[choosenBoat][dayAfter] == 0  ):
                        nbDaysAfter += 1	
                        dayAfter += 1
                        
                    nbDaysBefore = 0
                    dayBefore = day - 1
                    
                    while (dayBefore >= timeWindow[0] and solution.evalBatTimeRoute[choosenBoat][dayBefore] == 0  ):
                        nbDaysBefore += 1
                        dayBefore -= 1
                    
                    bestDays.append( (day, nbDaysAfter, nbDaysBefore) )
                # Pour chaque jours regarder l'intersection des jours pris avec l'intersection des interventions non encore faites.
                
#                cptIntersectInt = dict()
#                for day,nbJoursAf in bestDays:
#                    cptIntersectInt[day] = 0
#                    for idInt in range (len(self.interventions)):
#                        if self.listInterventionsDone[idInt] == False:
#                            inter2 = self.interventions[idInt]
#                            
#                            if inter2.reste and inter2.listBoat[choosenBoat] == 1:
#                                if (day <= inter2.beginning and day + inter.nbJoursInt <= inter2.end) or (day > inter2.beginning and day <= inter2.end):
#                                    cptIntersectInt[day] += 1
                  
                # Calculer le nombre de semaine que l'on va perdre en choisissant chaque temps
                cptWeekTaken = dict()
                dayPossibleRestraint = list()
                for day in daysPossible:
                    cptWeekTaken[day] = 0
                    if (day+inter.nbJoursInt) % DAYWEEK == 0:
                        for we in range (day//DAYWEEK, (day+inter.nbJoursInt)//DAYWEEK):
                            
                            if solution.evalBatTimeSemaine[choosenBoat][we] == 0:
                                cptWeekTaken[day] += 1
                    else:
                        for we in range (day//DAYWEEK, ((day+inter.nbJoursInt)//DAYWEEK)+1):
                            
                            if solution.evalBatTimeSemaine[choosenBoat][we] == 0:
                                cptWeekTaken[day] += 1
                
                    if solution.evalBatNbSemaine[choosenBoat] - cptWeekTaken[day] > 0:
                        dayPossibleRestraint.append(day)
                    
                    
                
                

                if len(dayPossibleRestraint) == 0:
                    continue
                else:
                    notPosed = False
                    # Update du potentiel de charge des bateaux
                    for b in potentialChargePerBoat.keys():
                        if inter.idInt in dictIntPossBoat[b]:
                            potentialChargePerBoat[b]-= inter.nbJoursInt

                
                # Garder uniquement les temps non Pareto-dominés et choisir aléatoirement dans ces solutions.
                listBestDaysPareto = list(dayPossibleRestraint)
                for day,nbJoursAf,nbJoursBf in bestDays:
                
                    for day2,nbJoursAf2,nbJoursBf2 in bestDays:
                        
                        if day != day2:
                            
                            if min(nbJoursAf,nbJoursBf) >= min(nbJoursAf2,nbJoursBf2) and cptWeekTaken[day] >= cptWeekTaken[day2] :
                                
                                if min(nbJoursAf,nbJoursBf) > min(nbJoursAf2,nbJoursBf2) or cptWeekTaken[day] > cptWeekTaken[day2] :
                                    if day in listBestDaysPareto: 
                                        listBestDaysPareto.remove(day)
                                        
                                elif nbJoursAf+nbJoursBf > nbJoursAf2+nbJoursBf2:
                                    if day in listBestDaysPareto: 
                                        listBestDaysPareto.remove(day)

                bestDay = random.choice(listBestDaysPareto)
                
                
                              
                # Positionner l'intervention
                
                self.listInterventionsDone[inter.idInt] = True
                
                
                for d in range (bestDay,bestDay+inter.nbJoursInt):
                    if inter.reste:
                        self.boats[choosenBoat].disponibility[d] = False
                        solution.addRoute(choosenBoat,d,[inter.idInt])       
                    else:
                        solution.addRoute(choosenBoat,d,[inter.idInt,inter.idInt])
                
                # Les evaluations des semaines qui ont changées
                for we in range (bestDay//DAYWEEK, ((bestDay+inter.nbJoursInt)//DAYWEEK )+1):
                    solution.evaluateWeek(choosenBoat,we,self.distanceMatriceBB[self.boats[choosenBoat].indMatrice],self.distanceMatriceBP[self.boats[choosenBoat].indMatrice],self.distanceMatricePP[self.boats[choosenBoat].indMatrice])
            
                                        
    def placement(self, solution, batType = [1,2,3,4]):
        """
        Placement des interventions double(qu'on doit effectuer 2 fois) .
        
        """
        # liste de la disponibilité des bateaux  pour ne pas placer deux interventions double le même jour.      
        disponibility = list()
        for b in range (len(self.boats)):
            disponibility.append([])
            disponibility[b] = [True for d in range (self.nbJours)]
        
        interventionsRD = list()
        
        for idInt in range (len(self.interventions)):

            if self.interventions[idInt].double  and self.interventions[idInt].notPossible == False :
                interventionsRD.append( (idInt, self.interventions[idInt].nbJoursInt,sum(self.interventions[idInt].listBoat),self.interventions[idInt].score))
            
        # Tri de la liste d'interventions considérée.
        interventionsRD.sort(key = lambda v:(-v[3],-v[2],v[1]))

       
        
        nbJoursTaken = [self.boats[b].maxDayYear for b in self.numBoatConfig]
        
        # Choix du jour et du bateau qui fera l'intervention.    
        for ind in range (len(interventionsRD)):
            
            inter = self.interventions[interventionsRD[ind][0]]
            
            # On selectionne les Vedettes si possible et ensuite les navires de travaux.
            
            possibleBoatPerType = dict()
            
            for i in batType:
                possibleBoatPerType[i] = list()

            for b in range (len(inter.listBoat)):
                if inter.listBoat[b] == 1 and self.boats[b].type in batType:
                    
                    possibleBoatPerType[self.boats[b].type].append(b)
              
            randBat = random.random()
            
            weightBatCopy = [ inter.weightBat[b] for b in range (len(self.boats)) ]
            
            for b in range (len(self.boats)):
            
                if self.boats[b].type in possibleBoatPerType.keys() \
                        and b in possibleBoatPerType[self.boats[b].type] \
                        and (self.boats[b].habitable or self.boats[b].type == 2) :
                    boatPossible = False
                    for month in range (self.nbJours//16):
                        temps = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].listPosition[month*16] ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                
                        temps += inter.time+3
                    
                        if temps <= TIMEBOAT and  ( min(inter.nbJoursInt,4)*temps <= self.boats[b].maxHourWeek) and nbJoursTaken[b] > inter.nbJoursInt:
                            boatPossible = True
                            break
                    if boatPossible == False:
                        weightBatCopy[b] = 0
                
                else:
            
                    temps = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                
                    temps += inter.time+3
                    
                    if temps >= TIMEBOAT or  ( min(inter.nbJoursInt,4)*temps >= self.boats[b].maxHourWeek) or nbJoursTaken[b] < inter.nbJoursInt:
                        weightBatCopy[b] = 0
            
            
            foundBoat = False
            choosenBoat = None
            for i in batType:
                sumWeightBatType =  sum([weightBatCopy[b] for b in possibleBoatPerType[i]]) 
                
                
                partialSum = 0
                if sumWeightBatType != 0:
                    for b in possibleBoatPerType[i]:
                        partialSum += weightBatCopy[b]/sumWeightBatType 
                        
                        
                        if randBat <= partialSum:
                            choosenBoat = b
                            foundBoat = True
                            break
                            
                if foundBoat == True:
                    break
                    
                
            if foundBoat == False :
                # pas de bateau trouvé pour faire cette intervention, necessite un déplacement.
                continue
            randTime = random.random()
            weightDayCopy = list()
            
            for day in range (inter.beginning,inter.end+1-inter.nbJoursInt):
                possible = True
                
                # Verification du temps
                temps = (self.distanceMatriceBP[ self.boats[choosenBoat].indMatrice ][ inter.numLine[ self.boats[choosenBoat].indMatrice ] ][ self.ports[ self.boats[choosenBoat].listPosition[day] ].id[self.boats[choosenBoat].indMatrice] ]/self.boats[choosenBoat].vitesse)*2 + inter.time
                
                temps += inter.time+3
                        
                if temps >= TIMEBOAT or  ( min(inter.nbJoursInt,4)*temps >= self.boats[b].maxHourWeek) or nbJoursTaken[b] < inter.nbJoursInt:
                    possible = False
                    continue

                
                for d in range (day,day + inter.nbJoursInt):
                    if disponibility[choosenBoat][d] == False or self.boats[b].disponibility[d] == False:
                        # Verification du temps
                        
                        possible = False
                        break
                        
                    
                
                if possible:
                    weightDayCopy.append( [inter.weightDay[choosenBoat][day],day] )
                
            sumWeightD = sum([ i[0] for i in weightDayCopy  ])
            partialSum = 0
            if len(weightDayCopy) == 0 or sumWeightD == 0:
                # Pas de solution possible pour cette intervention avec ces positions.
                continue
            
            
            for day in range (len(weightDayCopy)):
                partialSum += weightDayCopy[day][0]/sumWeightD 
                
                if randTime <= partialSum:
                    choosenDay = weightDayCopy[day][1]
                    break
                    
            self.listInterventionsDone[interventionsRD[ind][0]] = True
  
            nbJoursTaken[choosenBoat] -= inter.nbJoursInt
            
            solution.evalBatNbJour[choosenBoat] -= inter.nbJoursInt
            
            for d in range (choosenDay,choosenDay+inter.nbJoursInt):
                disponibility[choosenBoat][d] = False
                
                solution.addRoute(choosenBoat,d,[inter.idInt,inter.idInt])

    def routeVedette(self, randomWeight, solution, TimeBoatPerWeek):
        tempVedette = list(self.batParType[0])
        random.shuffle(tempVedette)
        for b in tempVedette:

            # Intervention par jour
            listInterventionsPossible = list()
            for idInt in range(len(self.interventions)):
                inter = self.interventions[idInt]
                if self.listInterventionsDone[idInt] == False and inter.listBoat[b] == 1 and self.interventions[
                    idInt].reste != True and self.interventions[idInt].double != True:
                    # if abs(self.ports[inter.nomClosestPort].indPlace - self.ports[self.boats[b].nomPortAtt].indPlace ) <=1:
                    if (self.distanceMatriceBP[self.boats[b].indMatrice][inter.numLine[self.boats[b].indMatrice]][
                            self.ports[self.boats[b].nomPortAtt].id[self.boats[b].indMatrice]] / self.boats[
                            b].vitesse) * 2 + inter.time <= TIMEBOAT:
                        # if abs(self.ports[inter.nomClosestPort].indPlace - self.ports[self.boats[b].nomPortAtt].indPlace ) <=1:

                        listInterventionsPossible.append(inter)

            self.boats[b].initialize()
            if randomWeight:
                self.boats[b].affectationInterventionDayRandomLearning(listInterventionsPossible, self.nbJours)
            else:
                self.boats[b].affectationInterventionDay(listInterventionsPossible, self.nbJours)
            for time in range(self.nbJours):
                # bateaux de type 1

                nbWeek = time // DAYWEEK
                departure = self.boats[b].nomPortAtt
                arrival = departure
                route, timeSpent, nbIntRetrieve = self.compute_route(b, time, TimeBoatPerWeek[b][nbWeek], departure,
                                                                     arrival, stock=0,
                                                                     route=solution.routeBatPerDay[b][time])
                for idInt in route:
                    self.interventions[idInt].setInterTime(time)
                    self.interventions[idInt].setInterBoat(b)
                solution.addRoute(b, time, route)

                TimeBoatPerWeek[b][nbWeek] -= timeSpent

        cptNonVisited = 0
        for i in range(len(self.interventions)):
            if self.listInterventionsDone[i] == True:
                cptNonVisited += 1
        print("après les bateaux de type 1 on a fait : ", cptNonVisited, " interventions")

        solution.setNonVisited(self.listInterventionsDone)
        solution.evaluate(self.distanceMatriceBB, self.distanceMatriceBP, self.distanceMatricePP, batType=[1])

        solution = self.exchangePhase(solution, 0, [1])
        solution = self.fusionDayPhase(solution, 1000, [1])
        solution = self.fusionPhase(solution, 1000, [1])

        cptNonVisited = 0
        for i in range(len(self.interventions)):
            if self.listInterventionsDone[i] == True:
                cptNonVisited += 1
        print("après les bateaux de type 1 on a fait : ", cptNonVisited, " interventions")

    def routeNT(self, randomWeight, solution, TimeBoatPerWeek, interventionsNav):
        tempNavire = list(self.batParType[1])
        random.shuffle(tempNavire)
        # Filtrer les interventions pour ne prendre que celle que eux peuvent faire

        for inter in self.interventions:
            onlyNavire = True
            for b in range(len(inter.listBoat)):
                if self.boats[b].type != 2 and self.boats[b].type != 1 and self.boats[b].type != 4:
                    if inter.listBoat[b] == 1:
                        onlyNavire = False
                        break
            if onlyNavire:
                interventionsNav.append(inter)

        for b in tempNavire:
            listInterventionsPossible = list()
            for inter in interventionsNav:
                if self.listInterventionsDone[inter.idInt] == False and inter.listBoat[b] == 1 and self.interventions[
                    inter.idInt].reste != True and self.interventions[inter.idInt].double != True:
                    if (self.distanceMatriceBP[self.boats[b].indMatrice][inter.numLine[self.boats[b].indMatrice]][
                            self.ports[self.boats[b].nomPortAtt].id[self.boats[b].indMatrice]] / self.boats[
                            b].vitesse) * 2 + inter.time <= TIMEBOAT:
                        listInterventionsPossible.append(inter)

            if randomWeight:
                # Choisir un nombre précis de semaine
                # ici on peut ignorer les temps ou le bateau part en campagne.

                nbCampaignBoat = len(self.boats[b].dictMonth.keys())
                allWeekPoss = [t for t in range(self.nbJours // DAYWEEK) if
                               self.boats[b].listPosition[t * DAYWEEK] == self.boats[b].nomPortAtt]
                random.shuffle(allWeekPoss)

                allTimePoss = list()
                for we in allWeekPoss:
                    for d in range(0, 4):
                        allTimePoss.append(we * DAYWEEK + d)

                self.boats[b].affectationInterventionDayRandomLearning(listInterventionsPossible, self.nbJours,
                                                                       allTimePoss[
                                                                       :self.boats[b].maxDayYear - nbCampaignBoat * 16])
            else:
                listInterventionsPossible = []
                self.boats[b].affectationInterventionDay(listInterventionsPossible, self.nbJours)

            for time in allTimePoss:
                # bateaux de type 2
                nbWeek = time // DAYWEEK
                departure = self.boats[b].nomPortAtt
                arrival = departure

                route, timeSpent, nbIntRetrieve = self.compute_route(b, time, TimeBoatPerWeek[b][nbWeek], departure,
                                                                     arrival, stock=self.boats[b].stock,
                                                                     route=solution.routeBatPerDay[b][time])

                for idInt in route:
                    self.interventions[idInt].setInterTime(time)
                    self.interventions[idInt].setInterBoat(b)
                solution.addRoute(b, time, route)

                TimeBoatPerWeek[b][nbWeek] -= timeSpent

        cptNonVisited = 0
        for i in range(len(self.interventions)):
            if self.listInterventionsDone[i] == True:
                cptNonVisited += 1
        print("après les bateaux de type 2 on a fait : ", cptNonVisited, " interventions")

        solution.setNonVisited(self.listInterventionsDone)

        solution.evaluate(self.distanceMatriceBB, self.distanceMatriceBP, self.distanceMatricePP, batType=[2])

        solution = self.fusionPhase(solution, 1000, [2])

        solution = self.fusionDayPhase(solution, 1000, [2])

        self.placement2(solution, batType=[1, 2])

        solution = self.exchangePhase(solution, 0, [2])

        solution = self.fusionPhase(solution, 1000, [2])

        cptNonVisited = 0
        for i in range(len(self.interventions)):
            if self.listInterventionsDone[i] == True:
                cptNonVisited += 1
        print("après les bateaux de type 2 on a fait : ", cptNonVisited, " interventions")

    def routeBaliseur(self,randomWeight, solution, TimeBoatPerWeek):
        # Traitement des bateaux baliseur

        # Choix de la position du baliseur pour chaque saison
        # self.boatPosition(solution)

        # self.placement(solution,batType = [3])

        tempBaliseur = list(self.batParType[2])
        random.shuffle(tempBaliseur)

        for b in tempBaliseur:

            listInterventionsPossible = list()

            for idInt in range(len(self.interventions)):
                inter = self.interventions[idInt]
                if self.listInterventionsDone[idInt] == False and inter.listBoat[b] == 1 and self.interventions[
                    idInt].reste != True and self.interventions[idInt].double != True:
                    if (self.distanceMatriceBP[self.boats[b].indMatrice][inter.numLine[self.boats[b].indMatrice]][
                            self.ports[self.boats[b].nomPortAtt].id[self.boats[b].indMatrice]] / self.boats[
                            b].vitesse) * 2 + inter.time <= TIMEBOAT:
                        listInterventionsPossible.append(inter)
            if randomWeight:
                # ici on peut ignorer les temps ou le bateau part en campagne.

                nbCampaignBoat = len(self.boats[b].dictMonth.keys())
                allWeekPoss = [t for t in range(self.nbJours // DAYWEEK) if
                               self.boats[b].listPosition[t * DAYWEEK] == self.boats[b].nomPortAtt]
                random.shuffle(allWeekPoss)

                allTimePoss = list()
                for we in allWeekPoss:
                    for d in range(0, 4):
                        allTimePoss.append(we * DAYWEEK + d)

                self.boats[b].affectationInterventionDayRandomLearning(listInterventionsPossible, self.nbJours,
                                                                       allTimePoss[
                                                                       :self.boats[b].maxDayYear - nbCampaignBoat * 16])
            else:
                self.boats[b].affectationInterventionDay(listInterventionsPossible, self.nbJours)

            for time in allTimePoss:
                nbWeek = time // DAYWEEK
                departure = self.boats[b].nomPortAtt
                arrival = departure
                route, timeSpent, nbIntRetrieve = self.compute_route(b, time, TimeBoatPerWeek[b][nbWeek], departure,
                                                                     arrival, stock=self.boats[b].stock, route=[])

                for idInt in route:
                    self.interventions[idInt].setInterTime(time)
                    self.interventions[idInt].setInterBoat(b)

                solution.addRoute(b, time, route, departure, arrival)

                TimeBoatPerWeek[b][nbWeek] -= timeSpent

        cptNonVisited = 0
        for i in range(len(self.interventions)):
            if self.listInterventionsDone[i] == True:
                cptNonVisited += 1
        print("après les bateaux de type 3 on a fait : ", cptNonVisited, " interventions")

        solution.setNonVisited(self.listInterventionsDone)
        solution.evaluate(self.distanceMatriceBB, self.distanceMatriceBP, self.distanceMatricePP, batType=[3])

        solution = self.fusionPhase(solution, 1000, [3])

        solution = self.fusionDayPhase(solution, 1000, [3])

        solution.evaluate(self.distanceMatriceBB, self.distanceMatriceBP, self.distanceMatricePP, batType=[3])

        self.placement2(solution, batType=[3])

        solution = self.exchangePhase(solution, 0, [3])

        solution = self.fusionPhase(solution, 1000, [3])

        solution.evaluate(self.distanceMatriceBB, self.distanceMatriceBP, self.distanceMatricePP, batType=[3])

        for b in self.numBoatConfig:
            if self.boats[b].type == 3 and solution.evalBatNbJour[b] < 0:
                print("apEval", b)
                input()

        cptNonVisited = 0
        for i in range(len(self.interventions)):
            if self.listInterventionsDone[i] == True:
                cptNonVisited += 1
        print("après les bateaux de type 3 on a fait : ", cptNonVisited, " interventions")

    def routeBC(self,randomWeight, solution, TimeBoatPerWeek,interventionsNav):
        tempBaliseurCotier = list(self.batParType[3])
        random.shuffle(tempBaliseurCotier)

        for b in tempBaliseurCotier:

            listInterventionsPossible = list()

            for inter in interventionsNav:
                idInt = inter.idInt
                if self.listInterventionsDone[idInt] == False and inter.listBoat[b] == 1 and self.interventions[
                    idInt].reste != True and self.interventions[idInt].double != True:
                    if (self.distanceMatriceBP[self.boats[b].indMatrice][inter.numLine[self.boats[b].indMatrice]][
                            self.ports[self.boats[b].nomPortAtt].id[self.boats[b].indMatrice]] / self.boats[
                            b].vitesse) * 2 + inter.time <= TIMEBOAT:
                        listInterventionsPossible.append(inter)
            if randomWeight:
                # ici on peut ignorer les temps ou le bateau part en campagne.

                nbCampaignBoat = len(self.boats[b].dictMonth.keys())
                allWeekPoss = [t for t in range(self.nbJours // DAYWEEK) if
                               self.boats[b].listPosition[t * DAYWEEK] == self.boats[b].nomPortAtt]

                random.shuffle(allWeekPoss)

                allTimePoss = list()
                for we in allWeekPoss:
                    for d in range(0, 4):
                        allTimePoss.append(we * DAYWEEK + d)

                self.boats[b].affectationInterventionDayRandomLearning(listInterventionsPossible, self.nbJours,
                                                                       allTimePoss[
                                                                       :self.boats[b].maxDayYear - nbCampaignBoat * 16])
            else:
                self.boats[b].affectationInterventionDay(listInterventionsPossible, self.nbJours)

            for time in allTimePoss:
                nbWeek = time // DAYWEEK
                departure = self.boats[b].nomPortAtt
                arrival = departure
                route, timeSpent, nbIntRetrieve = self.compute_route(b, time, TimeBoatPerWeek[b][nbWeek], departure,
                                                                     arrival, stock=self.boats[b].stock, route=[])

                for idInt in route:
                    self.interventions[idInt].setInterTime(time)
                    self.interventions[idInt].setInterBoat(b)

                solution.addRoute(b, time, route, departure, arrival)

                TimeBoatPerWeek[b][nbWeek] -= timeSpent

        cptNonVisited = 0
        for i in range(len(self.interventions)):
            if self.listInterventionsDone[i] == True:
                cptNonVisited += 1
        print("après les bateaux de type 4 on a fait : ", cptNonVisited, " interventions")

        solution.setNonVisited(self.listInterventionsDone)

        solution.evaluate(self.distanceMatriceBB, self.distanceMatriceBP, self.distanceMatricePP, batType=[4])

        solution = self.fusionPhase(solution, 1000, [4])

        solution = self.fusionDayPhase(solution, 1000, [4])

        self.placement2(solution, batType=[4])

        solution = self.exchangePhase(solution, 0, [4])

        solution = self.fusionPhase(solution, 1000, [4])

        solution.evaluate(self.distanceMatriceBB, self.distanceMatriceBP, self.distanceMatricePP, batType=[4])

        cptNonVisited = 0
        for i in range(len(self.interventions)):
            if self.listInterventionsDone[i] == True:
                cptNonVisited += 1
        print("après les bateaux de type 4 on a fait : ", cptNonVisited, " interventions")



    def routeDesign(self,solution,randomWeight = False ):
        """
        Construction des routes pour chaque bateau à chaque temps de l'algorithme.
        """


        # Liste pour chaque bateau à chaque semaine le temps restant car les bateaux ne peuvent pas faire plus d'un certain nombre d'heures par semaines.

        TimeBoatPerWeek = [[] for b in  self.boats]

        nbSemaine = self.nbJours//DAYWEEK


        for idInt in range (len(self.interventions)):
            self.interventions[idInt].setInterTime(0)
            self.interventions[idInt].setInterBoat(0)

        for b in self.numBoatConfig:
            self.boats[b].initialize()
            self.boats[b].setDisponibility([True for d in range (self.nbJours)])

            for week in range (nbSemaine):
                # Initialisation du nombre d'heure disponible par jour.
                TimeBoatPerWeek[b].append(self.boats[b].maxHourWeek)


        # Placement des interventions Doubles
        self.placement(solution,batType = [1])


        cptNonVisited = 0
        cptReste = 0

        for i in range (len(self.interventions)):
            if self.listInterventionsDone[i] == True:
                cptNonVisited += 1
            if self.interventions[i].reste or self.interventions[i].double:
                cptReste += 1

        for b in self.numBoatConfig:
            for week in range (nbSemaine):
                totalTimeWeek = 0
                for day in range (week*DAYWEEK, (week+1)*DAYWEEK):
                    if len(solution.routeBatPerDay[b][day]) != 0:
                        departure,arrival = solution.portBatPerDay[b][day]
                        totalTimeWeek += self.compute_time_route_simple(departure,arrival,self.boats[b],solution.routeBatPerDay[b][day])

                TimeBoatPerWeek[b][week] -= totalTimeWeek

        self.routeVedette(randomWeight=randomWeight, solution=solution, TimeBoatPerWeek=TimeBoatPerWeek)
        # Traitement Navire de travaux
        self.boatPositionTest(solution,batType = [2,3,4])
        interventionsNav = list()


        self.routeNT(randomWeight = randomWeight, solution = solution, TimeBoatPerWeek = TimeBoatPerWeek,interventionsNav=interventionsNav)


        self.routeBaliseur(randomWeight = randomWeight, solution = solution, TimeBoatPerWeek = TimeBoatPerWeek)

        self.routeBC(randomWeight=randomWeight, solution=solution, TimeBoatPerWeek=TimeBoatPerWeek, interventionsNav=interventionsNav)



#        listEsm = list()
#        for idInt in range (len(self.interventions)):
#            if self.interventions[idInt].reste or self.interventions[idInt].double:
#                if self.listInterventionsDone[idInt] == False:
#                    listEsm.append(self.interventions[idInt].numEsm)
#
#        solution.prepareToPlot()
#        solution.plotNotPossible(listEsm)


        # Baliseur côtier, même principe que Baliseur normaux

        #self.placement(solution,batType = [4])

        #        solution.check([3])
#        print("Checker 5")
#        input()
#        solution.check()
#        input()
        self.longRoute(solution)

        

                    
############################################################## Mise à Jour des paramètres ################################################################
    def addSolutionSort(self,currSolution):
        boolReplace = False
        for idSol in range (len(self.solutionKept)):
            if currSolution.compare(self.solutionKept[idSol]) :
                self.solutionKept.insert(idSol,currSolution)
                boolReplace = True
                break
        if boolReplace == False:
            self.solutionKept.append(currSolution)

    
    def update(self,pourcentage,currSolution):
    
        acceptSolution = False
        
        # Les solution sont triées de la meilleure à la moins bonne
        if len(self.solutionKept)< self.nbBestSolutionkept:
            self.addSolutionSort(currSolution)
            
        else :
            # Retire la moins bonne solution et rajoute la solution courante
            indSolMin = 0
            boolReplace = False
            
            if currSolution.compare(self.solutionKept[indSolMin]):
                boolReplace = True
            
            for indSol in range (len(self.solutionKept)):
                if self.solutionKept[indSolMin].compare(self.solutionKept[indSol]):
                    indSolMin = indSol
                    boolReplace = True
                    
            if boolReplace and currSolution.compare(self.solutionKept[indSolMin]):
                del self.solutionKept[indSolMin]
                acceptSolution = True
            if acceptSolution:
                self.addSolutionSort(currSolution)
                
            
            #else:
                # on peut peut être l'inserer dans les solutions diverses
        
        
#        for inter in self.interventions :
#            # poids initialisé à 1
#            inter.normalWeight()
        # Update des differents poids des interventions.
#        for idInt in currSolution.nonVisited:
#            if self.interventions[idInt].double != True and self.interventions[idInt].reste != True and self.interventions[idInt].notPossible != True :
#                self.interventions[idInt].updateBadWeightDay(self.beta)
        
        
    
#        for indSol in range (len(self.solutionKept)):
#        #for indSol in range (0,1):
#            for batNum in self.numBoatConfig:
#                for time in range (self.nbJours):
#                    for idInt in self.solutionKept[indSol].routeBatPerDay[batNum][time]:
#                        self.interventions[idInt].updateWeightDay2(self.beta * len(self.solutionKept)-indSol,self.alpha * len(self.solutionKept)-indSol,time,batNum )
#                        
#        for idInt in range (0,1):
#            print(self.interventions[idInt].weightDay)
#        input()
            
#        for batNum in self.numBoatConfig:
#            for time in range (self.nbJours):
#                for idInt in currSolution.routeBatPerDay[batNum][time]:
#                    self.interventions[idInt].updateWeightDay2(self.beta+1, self.alpha+1, time,batNum )

        # On met a jour les temps des interventions:
        updatedInt = [False for i in range (len(self.interventions))]
        
        for b in self.numBoatConfig:
            for time in range (self.nbJours):
                for idInt in currSolution.routeBatPerDay[b][time]:
                    if self.interventions[idInt].double == False and self.interventions[idInt].reste == False: 
                        self.interventions[idInt].setInterTime(time)
                        self.interventions[idInt].setInterBoat(b)   
                    
        for idInt in range (len(self.interventions)):
        
            if idInt not in currSolution.nonVisited and updatedInt[idInt] == False:
                updatedInt[idInt] = True
                cptSol = 0
                for sol in self.solutionKept:
                    if self.interventions[idInt].idInt in sol.routeBatPerDay[self.interventions[idInt].numBoat][self.interventions[idInt].interTime]:
                        cptSol += 1
                        continue
                                             
                #print(cptSol,len(self.solutionKept),(self.nbBestSolutionkept / 100) * pourcentage)
                if cptSol >= (self.nbBestSolutionkept / 100) * pourcentage:
                    print("UPDATE")
                    self.interventions[idInt].updateWeightDay2(self.beta+1, self.alpha+1, self.interventions[idInt].interTime,self.interventions[idInt].numBoat)
                    
                    
#        else:
#            for ind in range (len(self.interventions)):
#                cptSol = 0
#                for sol in self.solutionKept:
#                    for b in sol.numBoatConfig:
#                        if self.interventions[ind].idInt in sol.routeBatPerDay[self.interventions[ind].numBoat][self.interventions[ind].interTime]:
#                            cptSol += 1
#                            break
#                if cptSol <= (self.nbBestSolutionkept / 100) * 30:
#                    self.interventions[ind].updateBadWeightDay(self.beta)
            
            
        
        # Update de la mémoire sur les interventions
        

        return
        
        
############################################################# Fonctions Principales ####################################################################

                
    def fusionDayPhase(self,solution,nbIter,batType = [1,2,3,4], boatNum = None, timeWindow = list()):
        # Prendre un jour court et essayer de le fusionner avec un autre jour de sa propre semaine ou d'une semaine utilisée.
        cptAlgo = 0
        tabuLenght = nbIter//3
        listTabu = deque()
        listNumBoat = list()
        
        if boatNum!= None:
            listNumBoat.append(boatNum)
        else:            
            for b in self.numBoatConfig:
                if self.boats[b].type in batType:
                    listNumBoat.append(b)
            
        if len(listNumBoat) == 0:
            return solution
        
        if timeWindow == list():
            timeWindow = [0,self.nbJours]
        
        while (cptAlgo < nbIter):
            
            #choisir un bateau aléatoirement
            alea = random.randint(0,len(listNumBoat)-1)
            choosenBoatNum = listNumBoat[alea]
            # Regarder son pire jour ou un jour aléatoire
            
            aleaDay = random.randint(0,10)
            if aleaDay < 5 :
                worstDay = 0
                timeWorstDay = math.inf
                for time in range (timeWindow[0],timeWindow[1]):
                    if solution.evalBatTimeRoute[choosenBoatNum][time] < timeWorstDay and len(solution.routeBatPerDay[choosenBoatNum][time])>0 and self.boats[choosenBoatNum].disponibility[time] and self.interventions[solution.routeBatPerDay[choosenBoatNum][time][0]].double == False:
                        timeWorstDay = solution.evalBatTimeRoute[choosenBoatNum][time]
                        worstDay = time
            else:
                worstDay = random.randint(0,self.nbJours - 1)
                cpt = 0
                while ((len(solution.routeBatPerDay[choosenBoatNum][worstDay]) == 0 or self.boats[choosenBoatNum].disponibility[worstDay]== False or self.interventions[solution.routeBatPerDay[choosenBoatNum][worstDay][0]].double) and cpt < 100):
                    worstDay = random.randint(0,self.nbJours - 1)
                    cpt += 1
                if cpt == 100:
                    # On a pas trouvé de route pour ce bateau
                    cptAlgo += 1
                    continue
                
            #print("worstDay:",worstDay,solution.routeBatPerDay[choosenBoatNum][worstDay],choosenBoatNum)
            currRoute = list(solution.routeBatPerDay[choosenBoatNum][worstDay])
            for idInt in currRoute:
                inter = self.interventions[idInt]
                
                # On prend un jour utilisé, dans la même semaine et différent du pire jour.
                timeWindowloc = [i for i in range (DAYWEEK*(worstDay//DAYWEEK),((worstDay//DAYWEEK)+1)*(DAYWEEK)-1) if solution.evalBatTimeRoute[choosenBoatNum][i]> 0 and self.boats[choosenBoatNum].disponibility[i]  and i != worstDay and i>=inter.beginning and i<=inter.end ]
                
                for possibleTime in timeWindowloc:
                    # On essaye d'introduire l'intervention.
                    boat = self.boats[choosenBoatNum]
                    departure,arrival = solution.portBatPerDay[choosenBoatNum][possibleTime]
                    newRoute,timeSpent,nbIntRetrieve = self.greedy(departure,arrival,boat,listPossibleInterventions = [idInt],route = list(solution.routeBatPerDay[choosenBoatNum][possibleTime]),timeRoute = 0,timeAvailable = boat.maxHourWeek-solution.evalBatTimeSemaine[choosenBoatNum][possibleTime // DAYWEEK] + solution.evalBatTimeRoute[choosenBoatNum][possibleTime],stock = boat.stock - solution.countIntRetrieveWeek(choosenBoatNum,possibleTime))
                    if idInt in newRoute:
                        # On a réussi a introduire l'intervention
                        # On enlève l'intervention de l'ancienne route
                        #print("REUSSITE",newRoute,solution.routeBatPerDay[choosenBoatNum][possibleTime])
                        solution.routeBatPerDay[choosenBoatNum][worstDay].remove(idInt)
                        inter.setInterTime(possibleTime)
                        #On enlève l'intervention de son ancien temps et on la rajoute à son nouveau
                        for day in range (self.nbJours):
                            if idInt in self.boats[choosenBoatNum].listInterventionsPerDay[day]:
                                self.boats[choosenBoatNum].listInterventionsPerDay[day].remove(idInt)
                        inter.setInterBoat(choosenBoatNum)
                        self.boats[inter.numBoat].listInterventionsPerDay[inter.interTime].append(idInt)
                        
                        #self.interventions[idInt].idClosestPort = self.ports[departure].id
                        # On réévalue la solution
                        solution.routeBatPerDay[choosenBoatNum][possibleTime] = newRoute
                        solution.evaluateWeek(choosenBoatNum,possibleTime//DAYWEEK,self.distanceMatriceBB[self.boats[choosenBoatNum].indMatrice],self.distanceMatriceBP[self.boats[choosenBoatNum].indMatrice],self.distanceMatricePP[self.boats[choosenBoatNum].indMatrice])
                        #print(solution.routeBatPerDay[choosenBoatNum][possibleTime],solution.routeBatPerDay[choosenBoatNum][worstDay])
                        break
            cptAlgo +=1                    
                
                
                    
        
        return solution
    
    def fusionPhase(self,solution,nbIter,batType = [1,2,3,4], boatNum = None, timeWindow = list()):
        '''
        Fusion des routes d'un bateau qui sont isolées dans une semaine afin de diminuer le nombre de semaine de travail d'un bateau.
        '''       
        cptAlgo = 0
        
        tabuLenght = nbIter//15
        listTabu = deque()
        listNumBoat = list()
        
        
        if timeWindow == list():
            timeWindow = [0,self.nbJours-1]
        
        if boatNum != None:
            listNumBoat.append(boatNum)
        
        else:
            for b in self.numBoatConfig:
                if self.boats[b].type in batType:
                    listNumBoat.append(b)
                
        if len(listNumBoat) == 0:
            return solution
        
        while (cptAlgo < nbIter):

            # Identifier les bateaux qui travaillent plus que le nombre de semaines maximal.
            listBatNumTooMuchWork = list()
            
            for batNum in listNumBoat:
                if solution.evalBatNbSemaine[batNum] > self.boats[batNum].armement:
                    listBatNumTooMuchWork.append(batNum)
                    
            aleaBoat = random.randint(0,10)
            # choisir un bateau aléatoirement dans les bateaux qui travaillent trop.
            if  len(listBatNumTooMuchWork) >0 and aleaBoat < 3:
                alea = random.randint(0,len(listBatNumTooMuchWork)-1)
                choosenBoatNum = listBatNumTooMuchWork[alea]
            else:
                alea = random.randint(0,len(listNumBoat)-1)
                choosenBoatNum = listNumBoat[alea]
            # trouver sa semaine la moins remplie.
            minTimePerWeek = self.boats[choosenBoatNum].maxHourWeek
            weekMin = 0
            chan = False
            for we in range (timeWindow[0]//DAYWEEK, timeWindow[1]//DAYWEEK):
                if solution.evalBatTimeSemaine[choosenBoatNum][we] != 0 and solution.evalBatTimeSemaine[choosenBoatNum][we] < minTimePerWeek and (choosenBoatNum,we) not in listTabu:
                    dispo = True
                    for day in range (we*DAYWEEK, (we+1)*DAYWEEK):
                        if self.boats[choosenBoatNum].disponibility[day] == False:
                            dispo = False
                            listTabu.append((choosenBoatNum,weekMin))
                            if len(listTabu)>tabuLenght:
                                listTabu.popleft()
                            
                            break
                        if len(solution.routeBatPerDay[choosenBoatNum][day]) != 0 and  self.interventions[ solution.routeBatPerDay[choosenBoatNum][day][0] ].double:
                            dispo = False
                            listTabu.append((choosenBoatNum,weekMin))
                            if len(listTabu)>tabuLenght:
                                listTabu.popleft()
                            
                            break
                        
                    if dispo:
                        minTimePerWeek = solution.evalBatTimeSemaine[choosenBoatNum][we]
                        weekMin = we
                        chan = True
                    
            if chan == False:
                cptAlgo += 1
                continue

            listTabu.append((choosenBoatNum,weekMin))
            if len(listTabu)>tabuLenght:
                listTabu.popleft()
            # Essayer d'introduire les tournées de cette semaine dans d'autres semaines
            
            for time in range (weekMin*DAYWEEK,(weekMin+1)*DAYWEEK):
                currRoute = list(solution.routeBatPerDay[choosenBoatNum][time])
                #print("La route choisie:",solution.routeBatPerDay[choosenBoatNum][time])
                if len(currRoute)==0:
                    # pas de routes à traiter
                    continue
                # time window représente les temps ou la route peut être replacée. l'intersection des intervalles de temps des interventions composant 
                # la route
                timeWindowloc = list()
                firstIntervention = self.interventions[currRoute[0]]
                
                for possTime in range (firstIntervention.beginning, firstIntervention.end +1):
                
                    if self.boats[choosenBoatNum].disponibility[possTime] == False:
                        continue
                
                    timePossibleForAllInt = True
                    for idInt in currRoute:
                        if possTime < self.interventions[idInt].beginning or possTime > self.interventions[idInt].end or (possTime >= (weekMin*DAYWEEK) and possTime <= ((weekMin+1)*DAYWEEK)):
                            timePossibleForAllInt = False
                            break
                    if timePossibleForAllInt and solution.evalBatTimeRoute[choosenBoatNum][possTime] > 0 and timePossibleForAllInt >= timeWindow[0] and timePossibleForAllInt <= timeWindow[1]:
                        timeWindowloc.append(possTime)
                if len(timeWindowloc) != 0:
                    # On a trouvé des temps possibles on regarde si il existe un temps ou le bateau fait déjà une route
                    random.shuffle(timeWindowloc)
                    for t in timeWindowloc:
                        #print('le temps:',t, t//DAYWEEK)
                        if solution.evalBatTimeSemaine[choosenBoatNum][ t //DAYWEEK ] > 0:
                            # Si le bateau travaille pendant ce temps t on essaye d'introduire la route.
                            boat = self.boats[choosenBoatNum]
                            departure,arrival = solution.portBatPerDay[choosenBoatNum][t]
                            #print(solution.evalBatTimeRoute[choosenBoatNum][t],solution.routeBatPerDay[choosenBoatNum][t])
                            newRoute,timeSpent,nbIntRetrieve = self.greedy(departure,arrival,boat,listPossibleInterventions = currRoute,route = list(solution.routeBatPerDay[choosenBoatNum][t]),timeRoute = 0,timeAvailable = boat.maxHourWeek-solution.evalBatTimeSemaine[choosenBoatNum][t //DAYWEEK] + solution.evalBatTimeRoute[choosenBoatNum][t],stock = boat.stock - solution.countIntRetrieveWeek(choosenBoatNum,t))
                            #greedy(self,departure,arrival,boat,listPossibleInterventions,route=[],timeRoute=0,timeAvailable = math.inf)
                            
                            if len(newRoute) > len(solution.routeBatPerDay[choosenBoatNum][t]):
                                # On a réussi a introduire au moins une intervention
                                for idInt in newRoute:
                                    if idInt not in solution.routeBatPerDay[choosenBoatNum][t]:
                                        inter = self.interventions[idInt]
                                        # On enlève l'intervention de l'ancienne route
                                        currRoute.remove(idInt)
                                        solution.routeBatPerDay[choosenBoatNum][time].remove(idInt)
                                        
                                        inter.setInterTime(t)
                                        inter.setInterBoat(choosenBoatNum)
                                        #On enlève l'intervention de son ancien temps et on la rajoute à son nouveau
                                        for day in range (self.nbJours):
                                            if idInt in self.boats[choosenBoatNum].listInterventionsPerDay[day]:
                                                self.boats[choosenBoatNum].listInterventionsPerDay[day].remove(idInt)
                                        
                                        self.boats[inter.numBoat].listInterventionsPerDay[inter.interTime].append(idInt)
                                        
                                        
                                        #self.interventions[idInt].idClosestPort = self.ports[departure].id
                                # On réévalue la solution
                                solution.routeBatPerDay[choosenBoatNum][t] = newRoute
                                

                                
                                #print(solution.routeBatPerDay[choosenBoatNum][time])
                                #print(solution.routeBatPerDay[choosenBoatNum][t]) 
                                #print("indice bateau:",self.boats[choosenBoatNum].indMatrice,timeSpent,choosenBoatNum,t,departure,arrival,"temps echange:",time)                    
                                #solution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP,batType)          
                                solution.evaluateWeek(choosenBoatNum,t // DAYWEEK,self.distanceMatriceBB[self.boats[choosenBoatNum].indMatrice],self.distanceMatriceBP[self.boats[choosenBoatNum].indMatrice],self.distanceMatricePP[self.boats[choosenBoatNum].indMatrice])
                                solution.evaluateWeek(choosenBoatNum,time // DAYWEEK,self.distanceMatriceBB[self.boats[choosenBoatNum].indMatrice],self.distanceMatriceBP[self.boats[choosenBoatNum].indMatrice],self.distanceMatricePP[self.boats[choosenBoatNum].indMatrice])
                                
#                                if solution.evalBatTimeSemaine[choosenBoatNum][t // DAYWEEK] > self.boats[choosenBoatNum].maxHourWeek:
#                                    print("Erreur")
#                                    input()
                                
                                break
                            
                    # essayer d'introduire les interventions 1 par 1.
                    for idInt in currRoute:
                        inter = self.interventions[idInt]
                        
                        listPossTime = [ i for i in range (inter.beginning,inter.end+1) if solution.evalBatTimeRoute[choosenBoatNum][i] > 0 and i >=timeWindow[0] and i <= timeWindow[1]]
                        random.shuffle(listPossTime)
                        
                        for possTime in listPossTime:
                            
                            if self.boats[choosenBoatNum].disponibility[possTime] and (possTime < (weekMin*DAYWEEK) or possTime >= ((weekMin+1)*DAYWEEK)) and solution.evalBatTimeSemaine[choosenBoatNum][ possTime //DAYWEEK ] > 0:
                                boat = self.boats[choosenBoatNum]
                                departure,arrival = solution.portBatPerDay[choosenBoatNum][possTime]
                                newRoute,timeSpent,nbIntRetrieve = self.greedy(departure,arrival,boat,listPossibleInterventions = [idInt],route = list(solution.routeBatPerDay[choosenBoatNum][possTime]),timeRoute = 0,timeAvailable = boat.maxHourWeek-solution.evalBatTimeSemaine[choosenBoatNum][possTime //DAYWEEK] + solution.evalBatTimeRoute[choosenBoatNum][possTime],stock = boat.stock - solution.countIntRetrieveWeek(choosenBoatNum,possTime))
                                
                                if idInt in newRoute:
                                    #print(solution.evalBatTimeRoute[choosenBoatNum][possTime],solution.routeBatPerDay[choosenBoatNum][possTime],solution.routeBatPerDay[choosenBoatNum][time])
                                    # On a réussi a placer l'intervention
                                    #print(inter.interTime,solution.routeBatPerDay[choosenBoatNum][inter.interTime],inter.idInt)
                                    solution.routeBatPerDay[choosenBoatNum][time].remove(idInt)
                                    
                                    
                                    # identifier le port
                                    #On enlève l'intervention de son ancien temps et on la rajoute à son nouveau
                                    for day in range (self.nbJours):
                                        if idInt in self.boats[choosenBoatNum].listInterventionsPerDay[day]:
                                            self.boats[choosenBoatNum].listInterventionsPerDay[day].remove(idInt)
                                    
                                    self.interventions[idInt].setInterTime(possTime)
                                    self.interventions[idInt].setInterBoat(choosenBoatNum)
                                    self.boats[inter.numBoat].listInterventionsPerDay[inter.interTime].append(idInt)
                                    
                                    #self.interventions[idInt].idClosestPort = self.ports[departure].id
                                    # On réévalue la solution
                                    solution.routeBatPerDay[choosenBoatNum][possTime] = newRoute
                                    
                                    #print("indice bateau 2:",self.boats[choosenBoatNum].indMatrice,timeSpent,choosenBoatNum,possTime,departure,arrival,"temps echange:",time)
                                    solution.evaluateWeek(choosenBoatNum,possTime // DAYWEEK,self.distanceMatriceBB[self.boats[choosenBoatNum].indMatrice],self.distanceMatriceBP[self.boats[choosenBoatNum].indMatrice],self.distanceMatricePP[self.boats[choosenBoatNum].indMatrice])
                                    solution.evaluateWeek(choosenBoatNum,time // DAYWEEK,self.distanceMatriceBB[self.boats[choosenBoatNum].indMatrice],self.distanceMatriceBP[self.boats[choosenBoatNum].indMatrice],self.distanceMatricePP[self.boats[choosenBoatNum].indMatrice])
                                    #solution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP,batType)
#                                    if solution.evalBatTimeSemaine[choosenBoatNum][t // DAYWEEK] > self.boats[choosenBoatNum].maxHourWeek:
#                                        print("Erreur")
#                                        input()
                                    
                                    break

   
            
            
            # On met a jour les temps des interventions:
            for b in listNumBoat:
                for time in range (self.nbJours):
                    for idInt in solution.routeBatPerDay[b][time]:
                        self.interventions[idInt].setInterTime(time)
                        self.interventions[idInt].setInterBoat(b)
        

            cptAlgo += 1

        return solution
    
    def exchangePhaseBetweenBoats(self, solution, nbIter):
        '''
        Tente de rééquilibrer le nombre de jours et de semaine de travail des bateaux en échangeant des tournées/interventions.
        
        '''
        tabuLenght = self.nbJours * 3
        listTabu = deque()
        
        cptAlgo = 0
        while (cptAlgo < nbIter):
            choosenBoat = None
            nbDaysLeft = self.nbJours
            
            listBoat = [ (i,solution.evalBatNbJour[b]) for b in self.numBoatConfig ]
            listBoat.sort(key = sortSecond ,reverse = False)
            
            
                
            # Identifier le bateaux qui travaille le plus en jours
            for b in self.numBoatConfig:
                if solution.evalBatNbJour[b] < nbDaysLeft:
                    choosenBoat = b
                    nbDaysLeft = solution.evalBatNbJour[b]
            
            
            # Choisir un jour aleatoire
            
            daysPossibleChange = list(self.boats[choosenBoat].disponibility)

            for time in range (self.nbJours):
                
                if solution.evalBatTimeRoute[choosenBoat][time] == 0:
                    # Le jours n'est pas utilisé
                    daysPossibleChange[time] = False
                    
            dayPossible = [time for time in range (self.nbJours) if daysPossibleChange[time] ]
            if len(dayPossible) == 0:
                cptAlgo += 1
                break
            
            choosenDay = random.choice(dayPossible)
            print("Ensemble des jours possilbe:",dayPossible,daysPossibleChange)
            print("Le bateau et jour choisit",self.boats[choosenBoat].nom,choosenDay,self.boats[choosenBoat].disponibility[choosenDay],self.boats[choosenBoat].disponibility)
            input()            
            # On doit traiter les interventions séparement car on ne sait pas quels bateaux peuvent les faire
            currRoute = list( solution.routeBatPerDay[choosenBoat][choosenDay] )
            for idInt in currRoute:
                
                if self.interventions[idInt].reste or self.interventions[idInt].double : 
                    print(idInt,self.interventions[idInt].reste,self.interventions[idInt].double)
                    input()
                
                inter = self.interventions[idInt]

                # Definir la liste des bateaux pouvant faire cette intervention
                listPossibleBoat = list()
                for b in self.numBoatConfig:
                    if b != choosenBoat:
                        if inter.listBoat[b] == 1 and inter.double != True:
                            if self.boats[b].habitable == False:
                                if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time <= TIMEBOAT :
                                    listPossibleBoat.append(b)
                            else:
                                if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time <= TIMEBOATLONG :
                                    listPossibleBoat.append(b)
                                
                if len(listPossibleBoat) == 0:
                    continue
                # Choisir le bateau avec le plus de jours disponibles
                choosenBoatReceive = None
                nbDaysLeft = -math.inf
                for b in listPossibleBoat:
                    if solution.evalBatNbJour[b] > nbDaysLeft:
                        choosenBoatReceive = b
                        nbDaysLeft = solution.evalBatNbJour[b]
                
                
                listPossibleDay = [time for time in range (inter.beginning,inter.end+1) if self.boats[choosenBoatReceive].disponibility[time] ]
                    
                if len(listPossibleDay) == 0:
                    continue
                
                choosenDayReceive = random.choice( listPossibleDay )
                
                departure,arrival = solution.portBatPerDay[choosenBoatReceive][choosenDayReceive]
                
                
                newRoute,timeSpent,nbIntRetrieve = self.greedy(departure,arrival,self.boats[choosenBoatReceive],listPossibleInterventions = [inter.idInt],route = list(solution.routeBatPerDay[choosenBoatReceive][choosenDayReceive]),timeRoute = 0,timeAvailable = self.boats[choosenBoatReceive].maxHourWeek-solution.evalBatTimeSemaine[choosenBoatReceive][choosenDayReceive //DAYWEEK] + solution.evalBatTimeRoute[choosenBoatReceive][choosenDayReceive],stock = self.boats[choosenBoatReceive].stock - solution.countIntRetrieveWeek(choosenBoatReceive,choosenDayReceive))         
                
                if inter.idInt in newRoute:
                    # On a réussi à déplacer cette intervention
                    for time in range (self.nbJours):
                        if inter.idInt in self.boats[choosenBoatReceive].listInterventionsPerDay[time]: 
                            self.boats[choosenBoatReceive].listInterventionsPerDay[time].remove(inter.idInt)
                    
                    solution.routeBatPerDay[choosenBoat][choosenDay].remove(idInt)

                    solution.routeBatPerDay[choosenBoatReceive][choosenDayReceive] = newRoute
                                
                    inter.setInterTime(choosenDayReceive)
                    inter.setInterBoat(choosenBoatReceive)
                    
                    
                    self.boats[inter.numBoat].listInterventionsPerDay[inter.interTime].append(idInt)
                    
                    solution.evaluateWeek(choosenBoat,choosenDay // DAYWEEK,self.distanceMatriceBB[self.boats[choosenBoat].indMatrice],self.distanceMatriceBP[self.boats[choosenBoat].indMatrice],self.distanceMatricePP[self.boats[choosenBoat].indMatrice])
                                
                    solution.evaluateWeek(choosenBoatReceive,choosenDayReceive // DAYWEEK,self.distanceMatriceBB[self.boats[choosenBoatReceive].indMatrice],self.distanceMatriceBP[self.boats[choosenBoatReceive].indMatrice],self.distanceMatricePP[self.boats[choosenBoatReceive].indMatrice])
            
            cptAlgo += 1


    
            
    def exchangePhase(self, solution, nbIter,batType=[1,2,3,4], listInt = list(), boatNum = None, timeWindow = list()):
        '''
         Dans cette phase on essaye d'améliorer la solution en changeant les interventions qui ne sont pas traitées de période.
        
        liste tabou sur les mouvements possibles pour ne pas faire des cycles
        '''
        tabuLenght = 10000
        listTabu = deque()
        
        listNumBoat = list()
        
        if listInt == list():
            listInt = solution.nonVisited
            
        if timeWindow == list(): 
            timeWindow = [0,self.nbJours-1]
            
        if boatNum != None:
            listNumBoat.append(boatNum)
        else:
            for b in self.numBoatConfig:
                if self.boats[b].type in batType:
                    listNumBoat.append(b)        
        

        if len(solution.nonVisited) == 0:
            return
  
        copyNonVisited = list()
        
        for idInter in listInt:
            if self.interventions[idInter].reste != True and self.interventions[idInter].double != True and self.listInterventionsDone[idInter] == False:
                boolBoat = False
                
                for b in listNumBoat:
                    if self.interventions[idInter].listBoat[b] == 1:
                        boolBoat = True
                        break
                if boolBoat:
                    copyNonVisited.append(idInter)
        
        random.shuffle(copyNonVisited)
        
        for idInter in copyNonVisited:
            listTabu = deque()
            inter = self.interventions[idInter]
            portInter = inter.nomClosestPort
            tempInterTime = inter.interTime
            
            # chercher les bateaux étant au port affecté à l'intervention dans les temps de l'intervention
            # Chercher une periode pour échanger, si pas trouvée, en prendre une aléatoirement.
            
            dictBoatPossible = dict()
            for b in listNumBoat:
                if self.boats[b].habitable == False and self.boats[b].type != 2:
                    if inter.listBoat[b]==1:
                         if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time <= TIMEBOAT :
                            for i in range (inter.beginning,inter.end+1):
                                if solution.evalBatTimeRoute[b][i] + inter.time < TIMEBOAT and solution.evalBatTimeSemaine[b][i//DAYWEEK]+ inter.time < self.boats[b].armement and self.boats[b].disponibility[i] and (solution.evalBatNbJour[b] > 0 or solution.evalBatTimeRoute[b][i] != 0  ) :
                                    if b not in dictBoatPossible.keys():
                                        dictBoatPossible[b] = list()
                                        
                                    dictBoatPossible[b].append(i)
                else:
                    if inter.listBoat[b]==1:
                        for i in range ( max(inter.beginning,timeWindow[0]), min(inter.end + 1,timeWindow[1]+1)):
                            if (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].listPosition[i] ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time <= TIMEBOAT :
                                if solution.evalBatTimeRoute[b][i] + inter.time < TIMEBOAT and solution.evalBatTimeSemaine[b][i//DAYWEEK]+ inter.time < self.boats[b].armement and self.boats[b].disponibility[i] and (solution.evalBatNbJour[b] > 0 or solution.evalBatTimeRoute[b][i] != 0  ) :
                                    if b not in dictBoatPossible.keys():
                                        dictBoatPossible[b] = list()
                                        
                                    dictBoatPossible[b].append(i)
            
            
            
            #print("Temps possible: ",inter.idInt,dictBoatPossible)   
            if len(dictBoatPossible.keys())!= 0:
            
                while (True):
                    bestTime = None
                    bestBoat = None
                    habitable = False
                    nbDaysLeft = -math.inf
                    # Prendre le meilleur temps possibe et meilleur bateau
                    timeMaxRoute = 0
                    for b in dictBoatPossible.keys():
                        for time in dictBoatPossible[b]:
                            if solution.evalBatTimeRoute[b][time] >= timeMaxRoute and (inter.idInt,time,b) not in listTabu and solution.evalBatNbJour[b]> nbDaysLeft:
                                if (solution.evalBatTimeSemaine[b][time//DAYWEEK] > 0) or (solution.evalBatNbSemaine[b] > 0):
                                    # Eviter si plus de semaines disponibles.
                                    if (solution.evalBatTimeRoute[b][time] > 0) or (solution.evalBatNbJour[b] > 0):
                                        # Eviter les interventions si plus de jours disponibles.
                                    
                                        timeMaxRoute = solution.evalBatTimeRoute[b][time]
                                        bestTime = time
                                        bestBoat = b
                                        nbDaysLeft = solution.evalBatNbJour[b]
                                        habitable = self.boats[b].habitable

            
                        
                    if bestTime != None:
                        listTabu.append((inter.idInt,bestTime,bestBoat))

                        if len(listTabu)>tabuLenght:
                            listTabu.popleft()
                        
                        # regarder si il est possible de mettre cette intervention a cette date.
                        departure = solution.portBatPerDay[bestBoat][bestTime][0]
                        arrival = solution.portBatPerDay[bestBoat][bestTime][1]
                        route = solution.routeBatPerDay[bestBoat][bestTime]
                        timeRoute = solution.evalBatTimeRoute[bestBoat][bestTime]
                        timeAvailable = self.boats[bestBoat].maxHourWeek - solution.evalBatTimeSemaine[bestBoat][bestTime//DAYWEEK] + timeRoute
                        boat = self.boats[bestBoat]
                        #print(self.boats[b].maxHourWeek , solution.evalBatTimeSemaine[b][bestTime//DAYWEEK] , timeRoute)
                        newRoute,timeSpent,nbIntRetrieve = self.greedy(departure,arrival,self.boats[bestBoat],[inter.idInt],list(route),0,timeAvailable,stock = boat.stock - solution.countIntRetrieveWeek(bestBoat,bestTime) )
                        #print(timeSpent,timeRoute)
                        #print("Nouvelle route:",newRoute,route,timeSpent,"temps ",bestTime,"id int",inter.idInt )
                        if len(newRoute) > len(route):
                            #On a réussi à placer l'intervention 
                            for time in range (self.nbJours):
                                if inter.idInt in self.boats[bestBoat].listInterventionsPerDay[time]: 
                                     self.boats[bestBoat].listInterventionsPerDay[time].remove(inter.idInt)
                            inter.setInterTime(bestTime)
                            inter.setInterBoat(bestBoat)
                            self.boats[bestBoat].listInterventionsPerDay[inter.interTime].append(inter.idInt)
                            
                            solution.routeBatPerDay[bestBoat][bestTime] = newRoute
                            self.listInterventionsDone[idInter] = True
                            solution.setNonVisited(self.listInterventionsDone)
                            

                            solution.evaluateWeek(bestBoat,bestTime//DAYWEEK,self.distanceMatriceBB[self.boats[bestBoat].indMatrice],self.distanceMatriceBP[self.boats[bestBoat].indMatrice],self.distanceMatricePP[self.boats[bestBoat].indMatrice])
                            

                            break

                    else:
                        break
            #print(len(solution.nonVisited))
            
        return solution
    
    

    def firstPhase(self):
    
        if self.area == "MED":
            tempsEscale = TIMEESCALE
        else:
            tempsEscale = TIMEBOATNT
        
        for b in self.numBoatConfig:
            boat = self.boats[b]
            if self.boats[b].escale == None:
                escale = dict()
                
                for p in self.ports.keys():
                    
                    if self.boats[b].type == 2 :
                        if self.distanceMatricePP[boat.indMatrice][ self.ports[boat.nomPortAtt].id[boat.indMatrice] ][ self.ports[p].id[boat.indMatrice] ]/boat.vitesse < tempsEscale :
                            escale[p] = 1
                                 
                    else:
                        if self.distanceMatricePP[boat.indMatrice][ self.ports[boat.nomPortAtt].id[boat.indMatrice] ][ self.ports[p].id[boat.indMatrice] ]/boat.vitesse < TIMEBOAT:
                            escale[p] = 1
                        
                self.boats[b].escale = escale
                
            if self.boats[b].type == 2 :
                escale2 = dict()
                
                
                for p in self.ports.keys():
                    
                    if self.distanceMatricePP[boat.indMatrice][ self.ports[boat.nomPortAtt].id[boat.indMatrice] ][ self.ports[p].id[boat.indMatrice] ]/boat.vitesse < TIMEBOAT:
                        escale2[p] = 1
                        
                self.boats[b].escale2 = escale2
                print(self.boats[b].nom,self.boats[b].escale)

      
        
        #liste pour savoir si une intervention appartient deja a une tournée ou pas.
        self.listInterventionsDone = [False for i in self.interventions]
        
        for inter in self.interventions:
            inter.firstAffectation = False    
        
        # Creation d'une solution

        newSolution = so.Solution(self.nbJours,self.boats,self.numBoatConfig,self.ports,self.interventions,self.area)

#        self.attributeFixe(newSolution)
#        input()

        
        # Calcul le meilleur port pour chaque balise
        for inter in self.interventions:
            inter.ComputeClosestPort(self.distanceMatriceBP[0][inter.numLine[0]],self.nomPort)



        self.listInterventionsImpossible = list()
        for idInt in range (len(self.interventions)):
            # identifier le bateau le plus adapté pour cette intervention.
            inter = self.interventions[idInt]
            notPossible = True
            
            minTime = -math.inf
            minBoat = None
            minPort = None
            
            for b in self.numBoatConfig:
                if inter.listBoat[b] == 1:
                    if self.boats[b].habitable or self.boats[b].type == 2 :
                    
                        minTimePort = -math.inf
                        minPortint = None
                        
                        for p in self.boats[b].escale.keys():
                        
                            temps = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ p ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                            if inter.double:
                                temps += inter.time + 3
                                
                                
                                
                            if TIMEBOAT - temps >= minTimePort and (self.boats[b].habitable == False or inter.double or inter.reste):
                                minTimePort = TIMEBOAT - temps
                                minPortint = p
                            elif TIMEBOATLONG - temps >= minTimePort and self.boats[b].habitable and inter.double==False and inter.reste==False:
                                minTimePort = TIMEBOATLONG - temps
                                minPortint = p
                                
                                
                        if minTimePort >= minTime:
                            minTime = minTimePort
                            minPort = minTimePort
                            minBoat = b
                        
                        
                                        
                                        
                    else:
                        temps = (self.distanceMatriceBP[ self.boats[b].indMatrice ][ inter.numLine[ self.boats[b].indMatrice ] ][ self.ports[ self.boats[b].nomPortAtt ].id[self.boats[b].indMatrice] ]/self.boats[b].vitesse)*2 + inter.time
                        if inter.double:
                            temps += inter.time + 3
                            
                        if TIMEBOAT - temps >= minTime:
                            minTime = TIMEBOAT - temps
                            minPort = self.boats[b].nomPortAtt
                            minBoat = b 
                            
                        
            if minTime >= 0:

                
                
                if (inter.double or inter.reste):
                    if (-minTime + TIMEBOAT) * min(inter.nbJoursInt,4) <= self.boats[minBoat].maxHourWeek:
                        notPossible = False
                else:
                    notPossible = False
                    
            
            inter.setNotPossible(notPossible)
            
            if inter.notPossible:
                self.listInterventionsImpossible.append(inter.idInt)

#                bestBoat = 0
#                timeToDoIntervention = math.inf
#                for b in self.numBoatConfig  :
#                    
#                    boat = self.boats[b]
#                    if self.interventions[idInt].listBoat[b]==1:
#                        

#                        timeRequiered = (self.distanceMatriceBP[self.boats[b].indMatrice][inter.numLine[boat.indMatrice]][self.ports[self.boats[b].nomPortAtt].id[boat.indMatrice]]*2)/self.boats[b].vitesse + inter.time
#                        if timeRequiered < timeToDoIntervention:
#                            bestBoat = b
#                            timeToDoIntervention = timeRequiered
#                if timeToDoIntervention > TIMEBOAT :
#                    
#                    listEsmNum.append(inter.numEsm)
#                    inter.printInt()
#                    
#                    for b in self.numBoatConfig:
#                        if self.interventions[idInt].listBoat[b] == 1:
#                            print(self.boats[b].nom) 
#                        
#                    print(timeToDoIntervention)
#                    newSolution.plotNotPossible(listEsmNum)
#                    
#                    input()
#                    listEsmNum = list()
#                    inter.setTooFar(True)
#                
#                #print("L'intervention:",inter.idInt," dont le meilleur bateau est : ",self.boats[bestBoat].nom," a besoin de :",timeToDoIntervention," temps pour faire cette intervention",inter.numEsm,inter.typeEsm,inter.type)
#                else:
#                    inter.setTooFar(False)
#        print("Afiichage des impossibles")
#              
#        newSolution.plotNotPossible(listEsmNum)
#        
        self.listAffectationPortIntervention = list()
        
        for i in range (len(self.ports.keys())):
            self.listAffectationPortIntervention.append([])
        
        # Regroupe les interventions en fonction de leur port le plus proche.
        for inter in self.interventions:
            self.listAffectationPortIntervention[inter.idClosestPort].append(inter)
        
        for b in self.numBoatConfig:
            self.boats[b].restartListPosition()


        self.routeDesign(newSolution,randomWeight = True)

        newSolution.setNonVisited(self.listInterventionsDone)

        return newSolution
        
        
         
    def algorithmExecution(self,iterAlgo,boatfile,inst):
        """
        Fonction principale, itère en créant des nouvelles solutions.
        """
        print("DEBUT ALGORITHME")
        cptAlgo = 0
        cptRandomGen = 0
        
        self.memoryInt = list()
#        while cptRandomGen < self.nbBestSolutionkept*2 :
#        
#            # generation des poids aléatoires pour chaque intervention.
#            for inter in self.interventions :
#                inter.randomiseWeight()
#            # Initialisation des centroids
#            for p in self.ports.values():
#                p.update()
#            
#            currSolution=self.firstPhase()
#            currSolution.setNonVisited(self.listInterventionsDone)
#            currSolution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP)
#            
#            currSolution = self.exchangePhase(currSolution,500)
#            currSolution.setNonVisited(self.listInterventionsDone)
#            currSolution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP)
#            
#            currSolution = self.fusionPhase(currSolution,50)
#            
#            currSolution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP)

#            currSolution.check()
#            
#            self.update(pourcentage = 60,currSolution = currSolution)
#            
#            for i in range (len(self.solutionKept)):
#                print(i,"     :",self.solutionKept[i].cost," nbNonVisited: ",len(self.solutionKept[i].nonVisited)," MinWeekAvailable: ",self.solutionKept[i].minWeekAvailable )
#            
#            cptRandomGen += 1
                
        
        
        while(cptAlgo < iterAlgo ):

            
            # generation des poids normales pour chaque intervention.
            
            # Initialisation des centroids

            
            
            currSolution = self.firstPhase()



            print(len(self.interventions))
            # On va essayer de plot la densité par port

            dictIntperPort = dict()
            
            for p in self.ports.keys():
                dictIntperPort[p] = 0
                
            for inter in self.interventions:
                if inter.reste or inter.double:
                    dictIntperPort[inter.nomClosestPort] += inter.nbJoursInt
                


            totalDay = 0
            for b in self.numBoatConfig:
                print(b,self.boats[b].nom)
                print(self.boats[b].type)
                
                if self.boats[b].type == 2 or self.boats[b].type == 1:
                    totalDay += self.boats[b].maxDayYear
            print(totalDay)
            #input()
                
            
            
            nbDay = 0
            nbInt = 0
            nbTime = 0
            listNum = list()
            for inter in self.interventions:
                    onlyNavire = True
                    for b in range (len(inter.listBoat)):
                        if self.boats[b].type == 3:
                            if inter.listBoat[b] == 1:
                                onlyNavire = False
                                break
                    if onlyNavire:
                        nbInt += 1 
                        
                        if inter.reste or inter.double :
                            nbDay += inter.nbJoursInt
                            listNum.append(inter.numEsm)
                        else:
                            if inter.retrieve:
                                nbDay += 1
            
            print(nbDay,nbInt)
#            currSolution.prepareToPlot() 
#            currSolution.plotNotPossible(listNum)
            

            

            
            # e la liste des esms.
            
                        
            currSolution.setNonVisited(self.listInterventionsDone)

            currSolution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP)
            

            listValSolution = list()


            
            if cptAlgo >= 0:
                
                print("DEBUT FUSION")
                currSolution = self.fusionPhase(currSolution,1000,[1,2,3,4])

                print("DEBUT ECHANGE")
                currSolution = self.exchangePhase(currSolution,0,[1,2,3,4])
            
                currSolution = self.fusionPhase(currSolution,1000,[1,2,3,4])
                
                currSolution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP)




                # ICI on prend les interventions non traitées et on recommence

                newListInt = list()

                count = 0
                for idInt in range(len(self.interventions)):
                    if not self.listInterventionsDone[idInt]:
                        count += 1
                print("Non visités: ", count)
                

                currSolution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP)
                for b in self.numBoatConfig:
                    print(currSolution.evalBatNbJour[b])
                print("Le nombre de jours total restant:",sum(currSolution.evalBatNbJour))
                sumNBJour = 0
                tosave = []
                for idInt in range (len(self.interventions)):
                    inter = self.interventions[idInt]
                    if self.listInterventionsDone[idInt] == False:
                        self.interventions[idInt].updateScore(False)
                        tosave.append(self.interventions[idInt].tojson())
                        if inter.reste:
                            sumNBJour += inter.nbJoursInt
                    else:
                        self.interventions[idInt].updateScore(True)
                print("Somme des jours a traiter:",sumNBJour)
#                input()
                print(json.dumps(tosave))


                listNum = list()
                for idInt in range (len(self.interventions)):
                    inter = self.interventions[idInt]
                    if self.listInterventionsDone[idInt] == True:
                            if inter.reste :
                                listNum.append(inter.numEsm)
#                currSolution.prepareToPlot()
#                currSolution.plotNotPossible(listNum)
                         

                currSolution.evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP)
                
                currSolution.setNonVisited(self.listInterventionsDone)
                
                currSolution.setIterAlgo(cptAlgo)
                
                for b in self.numBoatConfig:
                    if self.boats[b].type == 2:
                        print(currSolution.evalBatTimeSemaine[b])
#                input()
                
                print("Le nombre de jours total restant:",sum(currSolution.evalBatNbJour))
                for b in self.numBoatConfig:
                    print(self.boats[b].nom, currSolution.evalBatNbJour[b] )
#                input()
                
                #print("DEBUT FUSION DES JOURS")
                #currSolution = self.fusionDayPhase(currSolution,500)

            currSolution.check()
            self.update(pourcentage = 50,currSolution = currSolution)
            listValSolution.append([len(currSolution.nonVisited),currSolution.minWeekAvailable,currSolution.minDayAvailable])
            print(currSolution.cost,len(currSolution.nonVisited),currSolution.minWeekAvailable,currSolution.minDayAvailable,currSolution.iteration)
            #print("Final non visited :",currSolution.nonVisited)
            
            print("Le nombre d'impossible:",len(self.listInterventionsImpossible))
            moyenne = 0
            for i in range (len(self.solutionKept)):
                moyenne += self.solutionKept[i].cost
                print(i,"     :",self.solutionKept[i].cost," nbNonVisited: ",len(self.solutionKept[i].nonVisited)," MinWeekAvailable: ",self.solutionKept[i].minWeekAvailable,"nbJours :",self.solutionKept[i].minDayAvailable,"L'iteration :" ,self.solutionKept[i].iteration)
            print("La moyenne ",moyenne / len(self.solutionKept))
            print("L'iteration:",cptAlgo)

            cptAlgo += 1
            
            
            #self.plot()
            #print(self.reward)
        
        bestSolInd = 0
        for indSol in range (len(self.solutionKept)):
            if self.solutionKept[indSol].compare(self.solutionKept[bestSolInd]):
                #print(indSol , bestSolInd)
                bestSolInd = indSol
        
        self.solutionKept[bestSolInd].evaluate(self.distanceMatriceBB,self.distanceMatriceBP,self.distanceMatricePP)
        self.solutionKept[bestSolInd].check()
        print("Le cout final de la solution :",self.solutionKept[bestSolInd].cost)
        
        
        #self.solutionKept[bestSolInd].prepareToPlot()
        #self.solutionKept[bestSolInd].createRealRoute()
        #self.solutionKept[bestSolInd].toJson2()
        listBoat = list()
        for b in self.numBoatConfig:
            if self.boats[b].type == 3:
                listBoat.append(b)
        #self.solutionKept[bestSolInd].plot3(self.numBoatConfig)
    
    
#        for route in self.solutionKept[bestSolInd].routeBatPerDay[4]:
#            print("route")
#            for idInt in route:
#                print(idInt,self.interventions[idInt].reste or self.interventions[idInt].double )
#        input()
    
        # for i in range (len(listValSolution)):
        #     print("à l'iteration ",i," : ",listValSolution[i])
        # self.solutionKept[bestSolInd].plotNonVisited()
        
        if self.area == "MED":
            self.solutionKept[bestSolInd].toCsv(donneeInt = './donnée/donneeInterventionsMed'+str(inst+1)+'hyp'+str(boatfile+1)+'.csv',donneeBat = './donnée/donneeBoatsMed'+str(inst+1)+'hyp'+str(boatfile+1)+'.csv')
        
        if self.area == "ATL":
            self.solutionKept[bestSolInd].toCsv(donneeInt = './donnée/donneeInterventionsAtl'+str(inst+1)+'hyp'+str(boatfile+1)+'.csv',donneeBat = './donnée/donneeBoatsAtl'+str(inst+1)+'hyp'+str(boatfile+1)+'.csv')


if __name__ == '__main__':

    
    traits = json.load(open("../../../../../data/clean/traitsv2.geojson"))
    traits20NM=json.load(open("../../../../../data/clean/Export Limite.geojson"))
    esm = pd.read_excel("../../../../../data/clean/extraction_Aladin_v4.xls", sheet_name="ECOBJECTS_PASSIVEEOBJECTS")
    
    # Génération avec des balises situées en Corse et faible nombre d'Esms
    
    area="MED"
    
    
    data = sel.DataSelector(traits,esm,traits20NM,depthFile="../../../../../data/resultat/depthNM.txt")
    tr, e ,t20= data.select(area)
    
    
    
    p=pd.read_csv("../../../../../data/clean/portMed.csv")
    
    nbJours = 208
    nbEsm = 752

    area = "ATL"
    
    
    if area == "MED":
        for i in [0,1,3]:
            for j in [0,1,2,3]:
                algo = MHalgorithm(nbJours,area)
                algo.readInstancesFile(
                filenameIntervention="./instanceTest3/interventionsTestMed"+str(j+1)+"_hyp"+str(i+1)+".json",
                filenameBoat="../boatSheet"+str(i)+"_V5.json",
                filenameDistanceBB="../../../../../data/clean/matriceDistance/distanceMedBB.csv",
                filenameDistanceBP="../../../../../data/clean/matriceDistance/distanceMedBP.csv",
                filenameDistancePP="../../../../../data/clean/matriceDistance/distanceMedPP.csv",
                filenameDistance20NMBB="../../../../../data/clean/matriceDistance/distanceMedBB20NM.csv",
                filenameDistance20NMBP="../../../../../data/clean/matriceDistance/distanceMedBP20NM.csv",
                filenameDistance20NMPP="../../../../../data/clean/matriceDistance/distanceMedPP20NM.csv",
                filenamePort="../../../../../data/clean/instanceMedV2/portMed.csv")
                algo.initiateAlgo(alpha = 1.8, beta = 1.2, nbBestSolutionkept = 10, nbDiverseSolutionkept = 10, config = None)
            
                algo.algorithmExecution( 30 ,i,j)
        
    else:

        for i in [0]:#, 1, 3, 5]:
            for j in [1]:
                algo = MHalgorithm(nbJours,area)
                algo.readInstancesFile(
                filenameIntervention="./instanceTest3/interventionsTestAtl"+str(j+1)+"_hyp"+str(i+1)+".json",
                filenameBoat="../boatSheet"+str(i)+"_V5.json",
                filenameDistanceBB="../../../../../data/clean/matriceDistance/distanceAtlBB.csv",
                filenameDistanceBP="../../../../../data/clean/matriceDistance/distanceAtlBP.csv",
                filenameDistancePP="../../../../../data/clean/matriceDistance/distanceAtlPP.csv",
                filenameDistance20NMBB="../../../../../data/clean/matriceDistance/distanceAtlBB20NM.csv",
                filenameDistance20NMBP="../../../../../data/clean/matriceDistance/distanceAtlBP20NM.csv",
                filenameDistance20NMPP="../../../../../data/clean/matriceDistance/distanceAtlPP20NM.csv",
                filenamePort="../../../../../data/clean/portAtl.csv")
                
            
            
                algo.initiateAlgo(alpha = 1.8, beta = 1.2, nbBestSolutionkept = 10, nbDiverseSolutionkept = 10, config = None)

            
                
                algo.algorithmExecution( 30 ,i,j)
                
        
    
    #cp = cProfile.Profile()
    #cp.enable()
    #cp.disable()
    #cp.print_stats()


        
        
        
        
