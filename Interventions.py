

import math
import random
from scipy.stats import norm
import json

alpha = 1.5

class Intervention:

    def __init__(self,numLine,numLine20NM,idInt,intervention,nbBat,nbJours):
        self.numLine = [numLine,numLine20NM]
        self.idInt = idInt
        self.nomEsm = intervention["nomEsm"]
        self.numEsm = intervention["esm"]
        self.type = intervention["type"]
        self.beginning = int(intervention["beginning"])
        self.end = int(intervention["end"])
        self.typeEsm = intervention["typeEsm"]
        self.time = intervention["time"]
        self.listBoat = intervention["listBoat"]
        self.retrieve = intervention["retrieve"]
        self.coord = intervention["coord"]
        self.nbJoursInt = int(intervention["nbJoursInt"])
        self.double = intervention["double"]
        self.reste = intervention["reste"]
        self.nbJours = nbJours
        self.nbBat = nbBat
        self.score = 1 - self.beginning/208



        self.firstAffectation = False


        #On assimile des poids aux bateaux possibles et aux temps.
        self.weightBat = [0 for i in range (nbBat)]


        self.weightDay = [[] for i in range(nbBat)]

        # Le poids associé à l'affectation à une période
        for b in range (nbBat):
            if self.listBoat[b] == 1:
                self.weightBat[b] = 1
                self.weightDay[b]=[1 for i in range (nbJours)]






            self.idClosestPort =- 1



    def ComputeClosestPort(self,distancePort,portNom):
        """
        distancePort un tableau des distances de l'intervention a chaque Port
        """
        minDist = math.inf
        idPort = -1
        for i in range (len(distancePort)):
            if (distancePort[i]<minDist):
                minDist = distancePort[i]
                idPort = i
        self.idClosestPort = idPort
        self.nomClosestPort = portNom[idPort]

    def normalWeight(self):


        self.weightDay = [[] for i in range(self.nbBat)]

        for p in range (self.nbBat):
            if  self.listBoat[p] == 1:
                self.weightBat[p] = 1
            self.weightDay[p]=[1 for i in range (self.nbJours)]

    def setTooFar(self,boolean):
        self.far = boolean

    def gaussianWeight(self):


        self.weightDay = [[] for i in range(self.nbBat)]

        for p in range (self.nbBat):
            # genère des poids entre 0 et 6
            #TODO la variance ?
            self.weightDay[p] = [0 for i in range (self.nbJours)]
            self.weightDay[p] = [norm(self.end-self.beginning,1).pdf(i) for i in range(self.beginning,self.end+1)]

    def randomiseWeight(self):


        self.weightDay = [[] for i in range(self.nbBat)]

        for p in range (self.nbBat):
            # genère des poids entre 0 et 6
            self.weightDay[p]=[random.random() for i in range (self.nbJours)]


    def updateWeightDay(self,beta):

        if self.weightDay[self.numBoat][self.interTime] > 100000:
            return
        self.weightDay[self.numBoat][self.interTime] /= beta

        for day in range (self.beginning,self.end+1):
            self.weightDay[self.numBoat][day] *= (beta/3)

        return

    def updateWeightDay2(self,beta,alpha,jour,batNum):
        self.weightBat[batNum] += alpha
        self.weightDay[batNum][jour] += beta

#        for day in range (self.beginning,self.end+1):
#            if day != jour:
#                self.weightDay[batNum][day] /= beta

        return


    def updateScore(self,scheduled):
        if not scheduled:
            self.score += 1
        self.score *= .9



#    def perturb(self):
#        random.randint(self.nbBat)
#        random.randint(self.nbJours)

    def updateBadWeightDay(self,beta):
        self.weightDay[self.numBoat][self.interTime] /= beta

    def setInterTime(self,time):
        self.interTime = time
        self.firstAffectation = True

    def setInterBoat(self,numBoat):
        self.numBoat = numBoat

    def setNotPossible(self,boolnP):
        self.notPossible = boolnP

    def printInt(self):

        print("nom ",self.nomEsm)
        print("num ",self.numEsm)
        print("type ",self.type )
        print("deb ",self.beginning)
        print("fin ",self.end )
        print("typeEsm ",self.typeEsm)
        print("temps ",self.time )
        print("listBat ",self.listBoat)
        print("coord ",self.coord )
        print("nb de jours ",self.nbJoursInt)
        print("double ",self.double )
        print("reste ",self.reste )

    def tojson(self):
        return {
            "beginning": self.beginning,
            "end": self.end,
            "type": self.type,
            "nbJoursInt": self.nbJoursInt,
            "nomEsm": self.nomEsm,
            "esm": self.numEsm,
            "typeEsm": self.typeEsm,
            "coord": self.coord,
            "time": self.time,
            "listBoat": self.listBoat,
            "retrieve": self.retrieve,
            "double": self.double,
            "reste": self.reste
        }
