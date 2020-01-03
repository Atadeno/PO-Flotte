import io
import numpy as np
import pandas as pd
import json
import Port as po
import sys
import math
import matplotlib.pyplot as plt
from shapely.geometry import asShape


import Interventions as it
import Port as po

sys.path.append('../../')
import selector as sel
import computeDistance as cd
import balise as bs
import config as cfg
# prix du litre d'essence
PRIX_ESSENCE = cfg.PRIX_ESSENCE
DAYWEEK = cfg.DAYWEEK
TIMEBOAT = cfg.TIMEBOAT
class Solution:
    """
    Une solution est définie par une liste d'interventions, liste de bateaux, pour chaque bateaux, pour chaque temps la route du bateau.
    """


    def __init__(self,nbJours,boats,numBoatConfig,ports,interventions,area):
        
        self.nbJours = nbJours
        self.boats = boats
        self.ports = ports
        self.numBoatConfig = numBoatConfig
        self.interventions = interventions
        self.area = area
        
        self.routeBatPerDay = list()
        self.realRouteBatPerDay = list()
        self.portBatPerDay = list()
        self.longRoute = list()
        
        for b in range (len(self.boats)):
            self.routeBatPerDay.append([])
            self.realRouteBatPerDay.append([])
            self.portBatPerDay.append([])
            self.longRoute.append([])
            
            
            for time in range (nbJours):
                self.routeBatPerDay[b].append([])
                self.realRouteBatPerDay[b].append([])
                self.portBatPerDay[b].append([self.boats[b].nomPortAtt, self.boats[b].nomPortAtt])
                
        self.evalBatTimeRoute = list()
        self.evalBatTimeSemaine = list()
        # liste du nb de semaines qu'un bateau travaille. 
        self.evalBatNbSemaine = list()
        self.evalBatNbJour = list()
        
        for b in range (len(self.boats)):
            self.evalBatTimeRoute.append([])
            self.evalBatTimeSemaine.append([])
            self.evalBatNbSemaine.append(self.boats[b].armement)
            self.evalBatNbJour.append(self.boats[b].maxDayYear)
            
            for time in range (self.nbJours):
                self.evalBatTimeRoute[b].append(0)
                        
                if time%DAYWEEK==0:
                    self.evalBatTimeSemaine[b].append(0)
                
    def setIterAlgo(self,iteration):   
        self.iteration = iteration
      
    def prepareToPlot(self):
        traits = json.load(open("../../../../../data/clean/traitsv2.geojson")) # from data folder.
        esm = pd.read_excel("../../../../../data/clean/extraction_Aladin_v4.xls", sheet_name="ECOBJECTS_PASSIVEEOBJECTS")
        traits20NM=json.load(open("../../../../../data/clean/Export Limite.geojson"))
        data = sel.DataSelector(traits,esm,traits20NM,depthFile="../../../../../data/resultat/depthNM.txt",depthPBMAfile = "../../../../../data/resultat/profondeurPBMASup1Atl.csv")
        self.t, self.e ,self.t20= data.select(self.area)

            
    def setNonVisited(self,listInterventionsDone):
        """
        Input: liste des identifiants des interventions non visité par la solution
        """
        nonVisited = list()
        cptInterNoVisit = 0
        cpt=0
        for boolInter in listInterventionsDone:
            if boolInter == False :
                nonVisited.append(cpt)
                cptInterNoVisit += 1
                #print("Une des interventions non faite:",
                #cpt,self.interventions[cpt].type,self.interventions[cpt].coord,self.interventions[cpt].interTime)

            cpt+=1
        self.nonVisited = nonVisited



    def countIntRetrieveWeek(self,batNum,day):
        """
        Retourne le nombre de slots restants sur le pont du bateau
        """
        nbIntRetrieve = 0
        

        for idInt in self.routeBatPerDay[batNum][day]:
            if self.interventions[idInt].retrieve:
                nbIntRetrieve +=1


        return nbIntRetrieve
        
    


    def evaluateWeek(self,batNum,week,distanceMatriceBB,distanceMatriceBP,distanceMatricePP):
        """
        Idem que pour evaluate mais que pour une semaine et un bateau
        """
        bat = self.boats[batNum]
        evalWeek = 0
        for day in range (week*DAYWEEK,(week+1)*DAYWEEK):
            route = self.routeBatPerDay[batNum][day]
            
            if len(route) == 0  :
                if self.evalBatTimeRoute[batNum][day] > 0.1 and self.portBatPerDay[batNum][day][0] == self.portBatPerDay[batNum][day][1] :
                    self.evalBatNbJour[batNum] += 1
                    self.evalBatTimeRoute[batNum][day] = 0
                    continue

                self.evalBatTimeRoute[batNum][day] = distanceMatricePP[self.ports[self.portBatPerDay[batNum][day][0]].id[bat.indMatrice]][self.ports[self.portBatPerDay[batNum][day][1]].id[bat.indMatrice]] / bat.vitesse
                evalWeek += self.evalBatTimeRoute[batNum][day]
                continue
            
                
            evalRoute = 0
            firstInt = self.interventions[route[0]]
            lastInt = self.interventions[route[-1]]
            
            
            

            
            evalRoute += ( distanceMatriceBP[firstInt.numLine[bat.indMatrice]][self.ports[self.portBatPerDay[batNum][day][0]].id[bat.indMatrice]]  / bat.vitesse ) 
            evalRoute += ( distanceMatriceBP[lastInt.numLine[bat.indMatrice]][ self.ports[self.portBatPerDay[batNum][day][1] ].id[bat.indMatrice]]  / bat.vitesse ) 
            
                
            for ind in range (len(route)-1):
                currInt = self.interventions[route[ind]]
                nextInt = self.interventions[route[ind+1]]
                evalRoute += ( distanceMatriceBB[currInt.numLine[bat.indMatrice]][nextInt.numLine[bat.indMatrice]]  / bat.vitesse )
                
            if self.evalBatTimeRoute[batNum][day] > 0 and evalRoute == 0:
                self.evalBatNbJour[batNum] += 1
                
            if self.evalBatTimeRoute[batNum][day] == 0 and evalRoute > 0:
                self.evalBatNbJour[batNum] -= 1
            
            self.evalBatTimeRoute[batNum][day] = evalRoute
            
            if self.evalBatTimeRoute[batNum][day] > TIMEBOAT + 0.5:
                self.evalBatNbJour[batNum] -= 1
                
            for idInt in route:
                self.evalBatTimeRoute[batNum][day] += self.interventions[idInt].time
            
                
            evalWeek += self.evalBatTimeRoute[batNum][day]
            
        if evalWeek == 0 and self.evalBatTimeSemaine[batNum][week] != 0:
            self.evalBatNbSemaine[batNum] += 1
        if evalWeek > 0 and self.evalBatTimeSemaine[batNum][week] == 0:
            self.evalBatNbSemaine[batNum] -= 1
        self.evalBatTimeSemaine[batNum][week] = evalWeek
        
        self.minWeekAvailable = min(self.evalBatNbSemaine)
        self.minDayAvailable = min(self.evalBatNbJour)
        self.sumWeekAvailable = sum(self.evalBatNbSemaine)
            
        return

            
        
    def evaluate(self,distanceMatriceBB,distanceMatriceBP,distanceMatricePP,batType = [1,2,3,4]):
        
        """
        Evalue le prix de l'ensemble des trajets effectués, TODO la durrée par route, le nombre de semaine par bateau et le nombre d'heures par semaine. 
        """
    
        evalSolution = 0
        
        
        listNumBoat = list()
        for b in self.numBoatConfig:
            if self.boats[b].type in batType:
                listNumBoat.append(b)
        
                for time in range (self.nbJours):
                    self.evalBatTimeRoute[b][time] = 0
                    if time%DAYWEEK==0:
                        self.evalBatTimeSemaine[b][time // DAYWEEK] = 0
                self.evalBatNbSemaine[b] = self.boats[b].armement
                self.evalBatNbJour[b] = self.boats[b].maxDayYear
            
        for batNum in listNumBoat:
            bat = self.boats[batNum]
            for time in range (self.nbJours):
                
                route = self.routeBatPerDay[batNum][time]
            
                if len(route) == 0:
                    if self.boats[batNum].habitable or self.boats[batNum].type == 2:
                        if time == self.nbJours-1 :
                            self.evalBatTimeRoute[batNum][time] = ( distanceMatricePP[bat.indMatrice][ self.ports[self.portBatPerDay[batNum][time][0]].id[bat.indMatrice] ][ self.ports[self.portBatPerDay[batNum][time][1]].id[bat.indMatrice] ] / bat.vitesse )
                            
                            evalSolution += self.evalBatTimeRoute[batNum][time] * bat.consommation * PRIX_ESSENCE
                        else :
                            self.evalBatTimeRoute[batNum][time] = ( distanceMatricePP[bat.indMatrice][ self.ports[self.portBatPerDay[batNum][time][0]].id[bat.indMatrice] ][ self.ports[self.portBatPerDay[batNum][time][1]].id[bat.indMatrice] ] /bat.vitesse )
                            
                            evalSolution += self.evalBatTimeRoute[batNum][time] * bat.consommation * PRIX_ESSENCE
                            
                    continue
                
                evalRoute = 0
                firstInt = self.interventions[route[0]]
                lastInt = self.interventions[route[-1]]
                
                if bat.habitable or bat.type == 2 :
                    
                    evalRoute += ( distanceMatriceBP[bat.indMatrice][firstInt.numLine[bat.indMatrice]][self.ports[self.portBatPerDay[batNum][time][0]].id[bat.indMatrice]]  / bat.vitesse ) 
                    evalRoute += ( distanceMatriceBP[bat.indMatrice][lastInt.numLine[bat.indMatrice]][ self.ports[self.portBatPerDay[batNum][time][1] ].id[bat.indMatrice]]  / bat.vitesse ) 
                    
                else :
                
                    evalRoute += ( distanceMatriceBP[bat.indMatrice][firstInt.numLine[bat.indMatrice]][bat.idPort]  / bat.vitesse ) 
                    evalRoute += ( distanceMatriceBP[bat.indMatrice][lastInt.numLine[bat.indMatrice]][bat.idPort]  / bat.vitesse ) 
                    
                for ind in range (len(route)-1):
                    currInt = self.interventions[route[ind]]
                    nextInt = self.interventions[route[ind+1]]
                    evalRoute += ( distanceMatriceBB[bat.indMatrice][currInt.numLine[bat.indMatrice]][nextInt.numLine[bat.indMatrice]]  / bat.vitesse )
                    
                self.evalBatTimeRoute[batNum][time] = evalRoute
                for idInt in route:
                    self.evalBatTimeRoute[batNum][time] += self.interventions[idInt].time
                
                evalSolution += evalRoute * bat.consommation * PRIX_ESSENCE
                    

                
        self.cost = evalSolution
        #calcul heures par semaine et calcul nombre de semaine restante ou le bateau peut travailler, calcul du nombre de jour ou il travaille
        for batNum in listNumBoat:
            nbSemaine = self.nbJours // DAYWEEK
            
            for we in range (nbSemaine):
                totalHeurSemaine = 0
                for time in range (0,DAYWEEK):
                    if self.evalBatTimeRoute[batNum][DAYWEEK*we + time] > 0:
                        self.evalBatNbJour[batNum] -= 1
                    if self.evalBatTimeRoute[batNum][DAYWEEK*we + time] > TIMEBOAT + 0.5:
                        self.evalBatNbJour[batNum] -= 1
                        self.longRoute[batNum].append(time)

                    totalHeurSemaine += self.evalBatTimeRoute[batNum][DAYWEEK*we + time]
                self.evalBatTimeSemaine[batNum][we] = totalHeurSemaine
                
                if totalHeurSemaine != 0:
                    # Si travaillé au moins une fois
                    self.evalBatNbSemaine[batNum] -= 1
                    
        # Le minimum de semaine qu'il reste a un bateau, on essaye de maximiser ce nombre
        self.minWeekAvailable = min(self.evalBatNbSemaine)
        self.minDayAvailable = min(self.evalBatNbJour)
        self.sumWeekAvailable = sum(self.evalBatNbSemaine)            
            
            
            
            
    def compare(self,solution):
        """
        Compare 2 solutions pour savoir laquelle est la meilleure.
        Ordre lexicographique sur nbInterventions, nb Semaines Disponibles,nombre de jours disponibles,somme des semaines utilisées pour chaque bateau , coût de la solution
        retourne True si self est la meilleur False sinon
        """
        if len(self.nonVisited) < len(solution.nonVisited):
            return True
        elif len(self.nonVisited) > len(solution.nonVisited):
            return False
            
        if self.minDayAvailable > solution.minDayAvailable:
            return True
        elif self.minDayAvailable < solution.minDayAvailable:
            return False
            
        if self.minWeekAvailable > solution.minWeekAvailable:
            return True
        elif self.minWeekAvailable < solution.minWeekAvailable:
            return False
        
            
        if self.sumWeekAvailable > solution.sumWeekAvailable:
            return True
        elif self.sumWeekAvailable < solution.sumWeekAvailable:
            return False
        
        if self.cost < solution.cost:
            return True
        else:
            return False
        
            
            
            
            
        

                
    def addRoute(self,batNum,time,route,departure = None,arrival = None, nbJours = 1):
        """
        Input le numero du bateau, time le jour ou il fait la route .route une liste d'identifiant d'esm
        addroute permet de calculer le chemin emprunté par le bateau et l'ajoute à routeBatPerDay
        """

        if departure == None and arrival == None:        
            departure = self.boats[batNum].listPosition[time]
            arrival = self.boats[batNum].listPosition[time]
        else:
            departure = departure
            arrival = arrival

        self.portBatPerDay[batNum][time] = (departure,arrival)
        self.routeBatPerDay[batNum][time] = route
        
        # TODO une liste du nombre de jours des tournées ou une tournée qui prend 2 slots 
        
        
    def createRealRoute(self):

        for batNum in self.numBoatConfig:
            bat = self.boats[batNum]
            for time in range (self.nbJours):
                route = self.routeBatPerDay[batNum][time]

                realRoute = list()
                
                arrival = self.portBatPerDay[batNum][time][1]
                departure = self.portBatPerDay[batNum][time][0]
                

                        


                
                if len(route)==0 :
                    if arrival != departure :
                        baliseA = bs.balise(self.ports[departure].coord[0],self.ports[departure].coord[1],self.t)
                        baliseB = bs.balise(self.ports[arrival].coord[0],self.ports[arrival].coord[1],self.t)
                        
                        if bat.limite20NM:
                            chemin,dist = cd.compute_distance_20NM(baliseA,baliseB,self.t,self.t20)
                        else:
                            chemin,dist = cd.compute_distance(baliseA,baliseB,self.t)
                        
                        self.realRouteBatPerDay[batNum][time].append(chemin)
                    
                    
                    
                else:

                    firstIntervention = self.interventions[ route[0] ]
                    
                    baliseA = bs.balise(self.ports[departure].coord[0],self.ports[departure].coord[1],self.t)
                    baliseB = bs.balise(firstIntervention.coord[0],firstIntervention.coord[1],self.t)
                    
                    if bat.limite20NM:
                        chemin,dist = cd.compute_distance_20NM(baliseA,baliseB,self.t,self.t20)
                    else:
                        chemin,dist = cd.compute_distance(baliseA,baliseB,self.t)
                    #print("TEST AFFICHAGE :",self.ports[departure].coord,firstIntervention.coord,chemin)
                    #cd.plotMap(chemin,self.t,t20=None)
                    



                    realRoute.append(chemin)
                                    
                    for ind in range (1,len(route)):
                        firstIntervention = self.interventions[ route[ind-1] ]
                        secondIntervention = self.interventions[ route[ind] ]
                        
                        baliseA = bs.balise(firstIntervention.coord[0],firstIntervention.coord[1],self.t)
                        baliseB = bs.balise(secondIntervention.coord[0],secondIntervention.coord[1],self.t)
                        
                        if bat.limite20NM:
                            chemin,dist = cd.compute_distance_20NM(baliseA,baliseB,self.t,self.t20)
                        else:
                            chemin,dist = cd.compute_distance(baliseA,baliseB,self.t)
                            

                        #cd.plotMap(chemin,self.t,t20=None)
                        #print(chemin)

                        realRoute.append(chemin)
                            
                    lastIntervention = self.interventions[ route[-1] ]

                    baliseA = bs.balise(lastIntervention.coord[0],lastIntervention.coord[1],self.t)
                    baliseB = bs.balise(self.ports[arrival].coord[0],self.ports[arrival].coord[1],self.t)
                    
                    if bat.limite20NM:
                        chemin,dist = cd.compute_distance_20NM(baliseA,baliseB,self.t,self.t20)
                    else:
                        chemin,dist = cd.compute_distance(baliseA,baliseB,self.t)
                    #cd.plotMap(chemin,self.t,t20=None)


                    realRoute.append(chemin)
                    self.realRouteBatPerDay[batNum][time] = realRoute
                

############################################################ FICHIER JSON ET CSV ###########################################################################

    def toCsv(self,donneeInt = './donnée/donneeInterventions.csv',donneeBat = './donnée/donneeBoats.csv'):
    
        """
        Crée un/deux fichier sur les données de la solution
        """
        listnbInt = dict()
        listIntCounted = list()
        for b in self.numBoatConfig:
            cpt = 0
            for time in range (self.nbJours):
                for idInt in self.routeBatPerDay[b][time]:
                    if idInt not in listIntCounted:
                        cpt += 1
                        listIntCounted.append(idInt)
                        self.interventions[idInt].setInterBoat(b)
                        self.interventions[idInt].setInterTime(b)   
                        
                        
            listnbInt[b] = cpt
        
        listInt = list()
        
        for idInt in range (len(self.interventions)):
            inter = self.interventions[idInt]
            if idInt not in self.nonVisited:
                listInt.append( [idInt,inter.numEsm,inter.typeEsm,inter.type,self.boats[inter.numBoat].nom,self.boats[inter.numBoat].nomPortAtt] )
                
            
            else:      
                listInt.append( [idInt,inter.numEsm,inter.typeEsm,inter.type,None,None] )
            if inter.notPossible:
                listInt[idInt].append(False)
            else:
                listInt[idInt].append(True)
        
        
        listBat = list()
        
        for b in self.numBoatConfig:
            boat = self.boats[b]
            listBat.append( [ boat.nom, self.evalBatNbJour[b], listnbInt[b] ] )
        
        dfInt = pd.DataFrame( listInt,columns = ["id","numEsm","typeEsm","type","nomBateau","nomPort","traitable"] )
        dfBat = pd.DataFrame( listBat,columns = ["nom","joursRestants","NbInt"] )
        
        
        dfInt.to_csv(donneeInt,index = False,encoding='utf-8-sig')
        dfBat.to_csv(donneeBat,index = False,encoding='utf-8-sig')
                
        
                            
            
    
    def toJson(self,path = './solution.json'):
        """
        Crée un fichier json à partir d'une solution. pour chaque bateau l'ensemble des tournées sur ce bateau.
        """
        listToJson = dict()
        
        for b in self.numBoatConfig:
            listToJson[self.boats[b].nom] = [[] for i in range (self.nbJours)]
            for time in range (self.nbJours):
                listToJson[self.boats[b].nom][time].append(self.portBatPerDay[b][time][0])
                for ind in range (len(self.routeBatPerDay[b][time])):
                    inter = self.interventions[ self.routeBatPerDay[b][time][ind] ]
                    
                    listToJson[self.boats[b].nom][time].append( {"beginning":inter.beginning,"end":inter.end,"numEsm":inter.numEsm,"type":inter.type,"coord":inter.coord,"time":inter.time} )
                listToJson[self.boats[b].nom][time].append(self.portBatPerDay[b][time][1])
        
        f=open(path,"w")
        json.dump(listToJson,f)

    def toJson2(self,path = './solution.json'):
        """
        Crée un fichier json à partir d'une solution. pour chaque jour l'ensemble des tournée sur ce jour.
        """
        listToJson = list()
        
        
        for time in range (self.nbJours):
            listToJson.append( dict() )
            listToJson[time]["jour"] = str(time)
            
            for b in self.numBoatConfig:
                listToJson[time][self.boats[b].nom] = list()
                
                departure = self.portBatPerDay[b][time][0]
                arrival = self.portBatPerDay[b][time][1]
                listToJson[time][self.boats[b].nom].append({"nom":departure, "coord":self.ports[departure].coord})

                if len(self.routeBatPerDay[b][time]) > 0:

                    for ind in range (len(self.routeBatPerDay[b][time])):
                        route = self.realRouteBatPerDay[b][time][ind]
                        distance = cd.orthodromiqueDistancePath(route)
                        tempsRoute = distance / self.boats[b].vitesse
                        listToJson[time][self.boats[b].nom].append({"route": route, "distance":distance,"temps": tempsRoute })
                        inter = self.interventions[ self.routeBatPerDay[b][time][ind] ]
                        listToJson[time][self.boats[b].nom].append( {"beginning":inter.beginning,"end":inter.end,"numEsm":inter.numEsm,"type":inter.type,"coord":inter.coord,"time":inter.time} )

                    if departure != arrival :
                        route = self.realRouteBatPerDay[b][time][-1]
                        distance = cd.orthodromiqueDistancePath(route)
                        tempsRoute = distance / self.boats[b].vitesse
                        listToJson[time][self.boats[b].nom].append({"route": route, "distance":distance,"temps": tempsRoute })

                listToJson[time][self.boats[b].nom].append({"nom":arrival, "coord":self.ports[arrival].coord})
        
        f=open(path,"w")
        json.dump(listToJson,f)




############################################################## AFFICHAGE ############################################################################
    def plot(self):
        """
        Affiche la solution, les routes que prend chaque bateau chaque jour (couleur differente en fonction des jours)
        DEPRECATED
        """
        # TODO mettre à jour
        cmap=plt.cm.get_cmap('hsv', self.nbJours)
        for b in self.numBoatConfig:
            # Une figure par bateau de la config
            fig = plt.figure(figsize=[15, 15])
            ax = fig.gca(xlabel="Longitude", ylabel="Latitude",title= str(self.boats[b].nom))
            
            #Affichage des ports
            for p in self.ports.keys():
                ax.plot(self.ports[p].lon,self.ports[p].lat,'^',linewidth=4,color='Black')
            
            for time in range (self.nbJours):
                if len(self.routeBatPerDay[b][time])==0:
                    if self.boats[b].habitable :
                        if time<self.nbJours-1 and self.boats[b].listPosition[time]!=self.boats[b].listPosition[time+1]:
                            coordPortDeparture = self.ports[self.boats[b].listPosition[time]].coord
                            coordPortArrival = self.ports[self.boats[b].listPosition[time+1]].coord
                            ax.plot([coordPortDeparture[0] ,coordPortArrival[0]] ,[coordPortDeparture[1] ,coordPortArrival[1]],linewidth=4,color=cmap(time))
                        if time == self.nbJours-1 and self.boats[b].listPosition[time]!=self.boats[b].nomPortAtt:
                            coordPortDeparture = self.ports[self.boats[b].listPosition[time]].coord
                            coordPortArrival = self.ports[self.boats[b].nomPortAtt].coord
                            
                            ax.plot([coordPortDeparture[0] ,coordPortArrival[0]] ,[coordPortDeparture[1] ,coordPortArrival[1]],linewidth=4,color=cmap(time))
                    continue
                    
                coordPortDeparture=self.ports[ self.boats[b].nomPortAtt ].coord
                coordPortArrival = coordPortDeparture

                if self.boats[b].habitable: 
                    if time<self.nbJours-1 :
                        coordPortDeparture = self.ports[ self.boats[b].listPosition[time] ].coord
                        coordPortArrival = self.ports[ self.boats[b].listPosition[time+1] ].coord
                    else:
                        coordPortDeparture = self.ports[ self.boats[b].listPosition[time] ].coord

                
                    
                firstIntervention = self.interventions[ self.routeBatPerDay[b][time][0] ].coord
                
                ax.plot([coordPortDeparture[0] ,firstIntervention[0]] ,[coordPortDeparture[1] ,firstIntervention[1]],linewidth=4,color=cmap(time))
                
                for ind in range (1,len(self.routeBatPerDay[b][time])):
                    interventionOne = self.interventions[ self.routeBatPerDay[b][time][ind-1] ].coord  
                    interventionSecond = self.interventions[ self.routeBatPerDay[b][time][ind] ].coord
                    
                    
                    ax.plot([interventionOne[0] ,interventionSecond[0]] ,[interventionOne[1] ,interventionSecond[1]],linewidth=4,color=cmap(time))
                    
                lastIntervention = self.interventions[ self.routeBatPerDay[b][time][-1] ].coord
                
                ax.plot([coordPortArrival[0] ,lastIntervention[0]] ,[coordPortArrival[1] ,lastIntervention[1]],linewidth=4,color=cmap(time))
            plt.show()



    def plot2(self):
        """
        Affiche de la solution, les routes que prend chaque bateau chaque jour (couleur differente en fonction des jours)
        DEPRECATED
        """
        # TODO mettre à jour
        for b in self.numBoatConfig:
            # Une figure par bateau et par temps de la config
            
            
            for time in range (self.nbJours):
                fig = plt.figure(figsize=[10, 10])
                ax = fig.gca(xlabel="Longitude", ylabel="Latitude",title= str(self.boats[b].nom)+ " au temps " + str(time))
                
                #Affichage des ports
                for p in self.ports.keys():
                    ax.plot(self.ports[p].lon,self.ports[p].lat,'^',linewidth=4,color='Black')
                
                
                if len(self.routeBatPerDay[b][time])==0 and self.boats[b].habitable==False:
                    # pas de départ ce jour
                    continue
                elif self.boats[b].habitable and time==0 and self.boats[b].nomPortAtt==self.boats[b].listPosition[time]:
                    # pas de départ
                    continue
                elif self.boats[b].habitable and time>0 and self.boats[b].listPosition[time-1]==self.boats[b].listPosition[time]:
                    #pas de départ
                    continue
                else :
                    if len(self.routeBatPerDay[b][time])==0:
                        coordPortDeparture = self.ports[ self.boats[b].listPosition[time-1] ].coord
                        coordPortArrival = self.ports[ self.boats[b].listPosition[time] ].coord
                        ax.plot([coordPortDeparture[0] ,coordPortArrival[0]] ,[coordPortDeparture[1] ,coordPortArrival[1]],linewidth=1,color='Black')
                        continue


                    
                
                coordPortDeparture=self.ports[ self.boats[b].nomPortAtt ].coord
                coordPortArrival = coordPortDeparture
                if self.boats[b].habitable:
            
                    if time>0 :
                        coordPortDeparture = self.ports[ self.boats[b].listPosition[time-1] ].coord
                    coordPortArrival = self.ports[ self.boats[b].listPosition[time] ].coord
                
                     
                firstIntervention = self.interventions[ self.routeBatPerDay[b][time][0] ].coord
                
                ax.plot([coordPortDeparture[0] ,firstIntervention[0]] ,[coordPortDeparture[1] ,firstIntervention[1]],linewidth=1,color='Black')
                
                for ind in range (1,len(self.routeBatPerDay[b][time])):
                    interventionOne = self.interventions[ self.routeBatPerDay[b][time][ind-1] ].coord  
                    interventionSecond = self.interventions[ self.routeBatPerDay[b][time][ind] ].coord
                    
                    
                    ax.plot([interventionOne[0] ,interventionSecond[0]] ,[interventionOne[1] ,interventionSecond[1]],linewidth=1,color='Black')
                    
                lastIntervention = self.interventions[ self.routeBatPerDay[b][time][-1] ].coord
                
                ax.plot([coordPortArrival[0] ,lastIntervention[0]] ,[coordPortArrival[1] ,lastIntervention[1]],linewidth=1,color='Black')
            
                plt.show()
                
    def plot3(self,listBoat):

        #print(self.realRouteBatPerDay[0])
        for b in listBoat:
            # Une figure par bateau de la config
            print("Le bateaux :",self.boats[b].nom," a travaillé ",self.boats[b].armement-self.evalBatNbSemaine[b]," semaines, il lui reste :", self.evalBatNbSemaine[b]," semaines et il lui reste ",self.evalBatNbJour[b],"jours")
            print("Voici le nombre d'heures effectuées par semaine:",self.evalBatTimeSemaine[b])
            
            for time in range (self.nbJours):
                if len(self.realRouteBatPerDay[b][time])!=0:
                    #fig = plt.figure(figsize=[8, 8])  # create a figure to contain the plot elements
                    #ax = fig.gca(xlabel="Longitude", ylabel="Latitude")
                    print("au temps : ",time)
                    print("La route effecuée par le bateau :",self.routeBatPerDay[b][time],self.portBatPerDay[b][time])
                    print("Le temps pour effectuer cette route:",self.evalBatTimeRoute[b][time])
                    print("Les temps des interventions des routes :")
                    if len(self.routeBatPerDay[b][time]) == 0:
                        print("XXXXXX")
                    print(self.routeBatPerDay[b][time])
                    for ind in self.routeBatPerDay[b][time]:
                        print("id: ",ind," temps: ",self.interventions[ind].time, " coord : ",self.interventions[ind].coord )
                    
                    #for path in self.realRouteBatPerDay[b][time]:
                    #    cd.plotMap(path,self.t,self.t20,fig = fig, ax = ax)
                    #ax.set_title('%s' % (self.boats[b].nom))
                    plt.show()
                    
    def plotNotPossible(self,listEsmNum, dictNum = None):
    
        dataFrameEsmNonVisited = self.e[self.e.NUM.isin(listEsmNum)]
        
        if dictNum == None:
            cd.plotMap( [], self.t, self.t20, dataFrameEsm = dataFrameEsmNonVisited )
            
        else:
            for ind,row in dataFrameEsmNonVisited.iterrows():
                print("le type:",row["TYPE_SUPPORT"])
                print("num ",row["NUM"])
                
                print(dictNum[row["NUM"] ][0].listBoat)
                
                if row["NUM"] in dictNum.keys():
                    print("Le bateau :", self.boats[ dictNum[row["NUM"] ][1] ].nom)
                    
                    p =  self.boats[ dictNum[row["NUM"] ][1] ].nomPortAtt
                    
                    port = bs.balise(self.ports[p].lon,self.ports[p].lat,self.t)
                    
                    ob = bs.balise(row["LONGITUDE"],row["LATITUDE"],self.t)
                    
                    chemin,dist=cd.compute_distance_20NM(port,ob,self.t,self.t20)
                    
                    cd.plotMap(chemin,self.t,self.t20)
                    
                    
        return
        
                 
    def plotNonVisited(self):
        listNumToPlot = list()
        listNumToPlot2 = list()
        listNumToPlot3 = list()
        cptFixe = 0
        cptBouee = 0
        cptAutre = 0
        cptNonVisited = 0
        for idInt in self.nonVisited:
            cptNonVisited += 1
            inter = self.interventions[idInt]

            
            if self.interventions[idInt].reste or self.interventions[idInt].double:
                listNumToPlot.append(self.interventions[idInt].numEsm)
                cptFixe += 1
                
#                    self.interventions[idInt].printInt()
#                    print(self.interventions[idInt].type,self.interventions[idInt].typeEsm, self.interventions[idInt].coord,self.interventions[idInt].beginning,self.interventions[idInt].end )
#                    for b in self.numBoatConfig:
#                        if self.interventions[idInt].listBoat[b] == 1:
#                            print(self.boats[b].nom) 
                
            elif self.interventions[idInt].typeEsm == "Bouée":
                cptBouee += 1
                print("Bouée", self.interventions[idInt].listBoat)
#                print(self.interventions[idInt].listBoat)
#                self.interventions[idInt].printInt()
#                
#                for b in self.numBoatConfig:
#                    if self.interventions[idInt].listBoat[b] == 1:
#                        print(self.boats[b].nom) 
#                
                listNumToPlot2.append(self.interventions[idInt].numEsm)
                
            else:
                cptAutre += 1
                listNumToPlot3.append(self.interventions[idInt].numEsm)
                #inter.printInt()
        dataFrameEsmNonVisited = self.e[self.e.NUM.isin(listNumToPlot)]
        dataFrameEsmNonVisited2 = self.e[self.e.NUM.isin(listNumToPlot2)]
        dataFrameEsmNonVisited3 = self.e[self.e.NUM.isin(listNumToPlot3)]
        
        
        print("Nombre d'interventions fixe :",cptFixe ,"et nombre de bouée : ",cptBouee,"Les autres : ",cptAutre," Le nombre de non visité total: ",cptNonVisited )
        cd.plotMap( [], self.t, self.t20, dataFrameEsm = dataFrameEsmNonVisited )
        cd.plotMap( [], self.t, self.t20, dataFrameEsm = dataFrameEsmNonVisited2 )
        cd.plotMap( [], self.t, self.t20, dataFrameEsm = dataFrameEsmNonVisited3 )
        

        return
        
        
            
        
                        
################################################################## Check #################################################################################

    def check(self,batType = [1,2,3,4]):
        '''
        Verifie qu'une solution est correct
        '''
        # Verifie que toutes les interventions sont effectuées
        print("##############################################################################")
        
#        print("Le nombre d'interventions non effectuées: ",len(self.nonVisited))
#        print("Les intervention en question :")
#        for idInt in self.nonVisited:
#            inter = self.interventions[idInt]
#            print("L'intervention :",idInt,"de type :", inter.type, "qui doit être éffectuée dans les temps : [",inter.beginning,",",inter.end,"] "," sur esm de type :",inter.typeEsm, "par les bateaux :",inter.listBoat,"le temps:",inter.time)
#            for b in range(len(inter.listBoat)):
#                if inter.listBoat[b] == 1:
#                    boat = self.boats[b]
#                    print(boat.nom)
            
        listBat = list()
        for b in self.numBoatConfig:
            if self.boats[b].type in batType:
                listBat.append(b) 
        
        
        # Verifie qu'on effectue pas 2 fois la même intervention
        print("##############################################################################")
        listInterventionDone = list()
        for b in listBat:
            for time in range (self.nbJours):
                for r in self.routeBatPerDay[b][time]:
                    if r in listInterventionDone and self.interventions[r].double != True and self.interventions[r].reste != True:
                        print("Doublon, l'intervention :",r," est effectuée plus d'une fois de type ",self.interventions[r].type,self.interventions[r].interTime,self.interventions[r].numBoat,time)
                    elif self.interventions[r].double != True and self.interventions[r].reste != True:
                        listInterventionDone.append(r)
        
        print('#################################################################################')
        
        # Verifie qu'on effectue bien les interventions reste et double.
        print('#################################################################################')
        
        
        dictIntDoubleReste = dict()
        for b in listBat:
            for time in range (self.nbJours):
                for r in self.routeBatPerDay[b][time]:
                    if self.interventions[r].reste == True or self.interventions[r].double == True:
                    
                        if self.interventions[r].reste == True:
                            if len(self.routeBatPerDay[b][time]) > 1:
                                print("Une intervention reste est effectuée avec d'autres interventions")
                        
                        if self.interventions[r].double == True:
                            if self.routeBatPerDay[b][time][0] != self.routeBatPerDay[b][time][-1]:
                                print("Une intervention double non exacte")
                                
                                                   
                        if r in dictIntDoubleReste.keys():
                            dictIntDoubleReste[r] += 1
                        else:
                            dictIntDoubleReste[r] = 1
        
        for idInt in dictIntDoubleReste.keys():
            if (self.interventions[idInt].reste and self.interventions[idInt].nbJoursInt == dictIntDoubleReste[idInt]) or (self.interventions[idInt].double and self.interventions[idInt].nbJoursInt *2 == dictIntDoubleReste[idInt]):
                listInterventionDone.append(idInt)
            else:
                print("Interventions reste/double faite a moitié",dictIntDoubleReste[idInt])
        
        
        print("Le nombre d'intervention différentes éffectuée:",len(listInterventionDone))

        if len(listInterventionDone) != len(self.interventions)-len(self.nonVisited):

            print("Il y a :",len(listInterventionDone)," dans les routes alors que le nombre de non faite est :",len(self.nonVisited) )

            
        # Verifie que toutes les interventions s'effectuent bien dans leur intervalle de temps
        print("##############################################################################")
        
        for b in listBat:
            for time in range (self.nbJours):
                for r in self.routeBatPerDay[b][time]:
                    if self.interventions[r].beginning > time or self.interventions[r].end < time :
                        inter = self.interventions[r]
                        print("Une intervention non correcte trouvée:",r)
                        print("intervention effectuée au temps:",time," qui doit être éffectuée entre : [",inter.beginning,",",inter.end,"] , coordonnée: ",inter.coord," sur l'esm de type : ",inter.typeEsm, " et de nom: ",inter.nomEsm)

        # Verifie que pas plus de stock retrieve sont fait dans la même tournée
        print("##############################################################################")
        time = 0
        for b in listBat:
            if self.boats[b].habitable == False:
                for time in range (self.nbJours):
                    cpt = 0
                    for idInt in self.routeBatPerDay[b][time]:
                        if self.interventions[idInt].retrieve:
                            cpt += 1
                    if cpt > self.boats[b].stock:
                        print("Il y a plus de deux interventions de remplacement faite en même temps",self.boats[b].nom,time,self.routeBatPerDay[b][time],self.portBatPerDay[b][time])
                break
                    
#            cpt = 0
#            while self.boats[b].escale[self.portBatPerDay[b][time][1]] != 1:
#                for idInt in self.routeBatPerDay[b][time]:
#                    if self.interventions[idInt].retrieve:
#                        cpt += 1
#                time +=1
#                if time == self.nbJours:    
#                    break
#                if cpt > self.boats[b].stock :
#                    print("Il y a plus de deux interventions de remplacement faite en même temps")
            
        # Verifie que tout les bateaux travaillent un temps pas plus long que leurs limitations
        print("##############################################################################")
        
        for b in listBat:
            for time in range (self.nbJours):
                if self.portBatPerDay[b][time][0] != self.portBatPerDay[b][time][1] and self.evalBatTimeRoute[b][time]<1:
                    print("FAUX",self.portBatPerDay[b][time],self.evalBatTimeRoute[b][time])
                    input() 
                if self.evalBatTimeRoute[b][time] > 14:
                    print("La route :",self.routeBatPerDay[b][time]," du bateau ",self.boats[b].nom," au temps : ",time," dure ",self.evalBatTimeRoute[b][time],"; c'est une route de deux jours." )
                    #input()
                
        for b in listBat:
            for week in range (self.nbJours//DAYWEEK):
                if self.evalBatTimeSemaine[b][week] > self.boats[b].maxHourWeek:
                    print("Le bateau ",b, " travaille :",self.evalBatTimeSemaine[b][week], "la semaine :",week,"alors qu'il est limité à:",  self.boats[b].maxHourWeek," heures.")
                    for d in range (week*DAYWEEK, (week+1)*DAYWEEK):
                        print(self.routeBatPerDay[b][d],self.evalBatTimeRoute[b][d],self.portBatPerDay[b][d])
                        print(self.boats[b].listPosition)
                    #input()
        
        for b in listBat :
            if self.evalBatNbSemaine[b] < 0:
                print("Le bateau ",b, self.boats[b].nom," travaille :",self.boats[b].armement - self.evalBatNbSemaine[b]," semaine alors qu'il est supposé travailler :", self.boats[b].armement," semaines." )
                
                
                

                
                    
                    
                    
                    
                    
                    

            
