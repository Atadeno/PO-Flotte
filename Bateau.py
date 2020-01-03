import sys
sys.path.append('../../')

import pulp as pu
import selector as sel
import InstanceGeneration as ig
import computeDistance as cd
import math
import random
import config as cfg

DAYWEEK = cfg.DAYWEEK
TIMEBOAT = cfg.TIMEBOAT
ORDEREDMED= cfg.ORDEREDMED
ORDEREDATL = cfg.ORDEREDATL
NBJOURS = cfg.NBJOURS


class Bateau:
    euclidean = lambda self,a, b, c, d: math.sqrt((a - c) ** 2 + (b - d) ** 2)
    
    def __init__(self,dictBat,num):
        self.num = num
        self.nom = dictBat["Nom"]
        self.nature = dictBat["Nature"]
        
        # limite20NM est True si le bateau est limité a la zone des 20NM, false sinon
        self.limite20NM = True
        if dictBat["Rayon"] == [[600,"port"]] or dictBat["Rayon"] == [[100,"port"]]:
            # matrice normale
            self.limite20NM = False
         
        self.dictMonth = dict()
        
        
        if "Baliseur" in self.nature :
            self.type = 3
            self.stock = 2
        elif "Navire" in self.nature and dictBat["Habitable"]=='N':
            self.type = 2
            self.stock = 1
        elif "baliseur" in self.nature or "Navire" in self.nature and dictBat["Habitable"]=='O':
            self.type = 4
            
            if "baliseur" in self.nature:
                self.stock = 2
            else:
                self.stock = 1

            
             
        else :
            self.type = 1
            self.stock = 0
            
        self.portPrimaire = dictBat["Port"]

        
        
        
        self.vitesse = dictBat["Vitesse"]
        if dictBat["Habitable"]=='O':
            self.habitable = True
        else:
            self.habitable = False
        
        self.consommation = dictBat["Conso"]
        self.cout = dictBat["Cout"]
        self.armement = dictBat["Nombre_Semaines"]
        if self.armement == None or type(self.armement)==type(" "):
            self.armement = 0
        
        self.maxHourWeek = dictBat["Nombre_Heures"]
        
        if self.maxHourWeek == None:
            self.maxHourWeek = 0
        self.maxDayYear = math.floor(dictBat["Jours_Disponibles_Reels"])
        
        if "Escale" in dictBat.keys():
            self.escale = dictBat["Escale"]
        else:
            self.escale = None
        
    def setDisponibility(self,disponibility):
        self.disponibility = disponibility
    
        
    def setPort(self,idPort,nomPort,coordPort,area):
        """
        
        """
#        if self.nom == "Le Lavagnon":
#            nomPort = "ARCACHON"
#        if self.nom == "Amfard":
#            nomPort = "LORIENT"
        
        self.nomPortAtt = nomPort
        
        self.listPosition = [self.nomPortAtt for i in range (NBJOURS)]
        self.coordPort = coordPort

        if area=="MED":
            for zone in range (len(ORDEREDMED)):
                if self.nomPortAtt in ORDEREDMED[zone]:
                    self.navigationArea = zone
                    
            if self.limite20NM and self.navigationArea == 1 :
                self.limite20NM = False            
                    
        else:
            for zone in range (len(ORDEREDATL)):
                if self.nomPortAtt in ORDEREDATL[zone]:
                    self.navigationArea = zone
                    
            if self.limite20NM and self.navigationArea == 1 :
                self.limite20NM = False       
            
        if self.limite20NM:
            self.indMatrice = 1
        else:
            self.indMatrice = 0 
            
        self.idPort = idPort[self.indMatrice]
        
    def initiateListPosition(self,month,harbour):
        self.dictMonth[month] = harbour
        
        for j in range (16):
            self.listPosition[ (month*16) + j ] =  harbour 
        
    def restartListPosition(self):
        self.dictMonth = dict()
        self.listPosition = [self.nomPortAtt for i in range (NBJOURS)]
        
    
    def addListPosition(self,position):
        self.listPosition += position

    def changeListposition(self,portName,firstDay,endDay):

        for i in range (firstDay,endDay+1):

            self.listPosition[i] = portName
        
        
    def update(self,listIntervention,nbJours):
        """
        On prend toute les interventions qui peuvent avoir lieu à un temps t et affectées à ce bateau et on calcule le nouveau centroid à 
        l'aide des weightDays de chaque intervention.
        """
        self.centroid = list()
        
        for jour in range (nbJours):
            self.centroid.append([])
        
        totalSumWeightPerDay = [0 for i in range (nbJours)]
        
        lonSumWeightDay = [0 for i in range (nbJours)]
        
        latSumWeightDay = [0 for i in range (nbJours)]
        
        for inter in listIntervention:
            for time in range (inter.beginning,inter.end+1):
                totalSumWeightPerDay[time] += inter.weightDay[self.num][time]
                lonSumWeightDay[time] += inter.weightDay[self.num][time] * inter.coord[0]
                latSumWeightDay[time] += inter.weightDay[self.num][time] * inter.coord[1]

        for day in range (nbJours):
            if totalSumWeightPerDay[day] == 0:
                self.centroid[day] = [0,0]
            else:
                self.centroid[day] = [1.0*lonSumWeightDay[day]/totalSumWeightPerDay[day], 1.0*latSumWeightDay[day]/totalSumWeightPerDay[day]]
        


    def affectationInterventionDayRandom(self,listIntervention,nbJours):
        self.listInterventionsPerDay = list()
        for time in range (nbJours):
            self.listInterventionsPerDay.append(list())
            
        for inter in listIntervention:
            possTime = [time for time in range (inter.beginning,inter.end+1) if self.disponibility[time] ]
            if len (possTime) == 0:
                continue
            randTime = random.choice( possTime )
            self.listInterventionsPerDay[randTime].append(inter.idInt)
            inter.setInterTime(randTime)
            inter.setInterBoat(self.num)
            
            
        
        
    def initialize(self):
        self.listInterventionsPerDay = list()
    
    def affectationInterventionDayRandomLearning(self,listIntervention,nbJours,listPossibleDay = None):
        
        
        #listPoss = [i for i in range (nbJours) if self.disponibility[i]]
        if listPossibleDay == None:
            listPossibleDay = [i for i in range (nbJours) ]
        for time in range (nbJours):
            self.listInterventionsPerDay.append(list())
            
        for inter in listIntervention:
            randNumber = random.random()
            
            weightCopy = [[day,inter.weightDay[self.num][day]] for day in range (inter.beginning,inter.end+1) if self.disponibility[day] and day in listPossibleDay ]
            sumWeight = sum([ i[1] for i in weightCopy ])
            partialSum = 0
            if sumWeight == 0:
                continue
            
            for weight in range (len(weightCopy)):
                weightCopy[weight][1] = partialSum + weightCopy[weight][1]/sumWeight 
                partialSum += weightCopy[weight][1]
                
                if randNumber <= weightCopy[weight][1]:
                    choosenDay = weightCopy[weight][0]
                    break
                
            #self.listInterventionsPerDay[ listPoss[0] ].append(inter.idInt)
            self.listInterventionsPerDay[ choosenDay ].append(inter.idInt)
            inter.setInterTime(choosenDay)
            inter.setInterBoat(self.num)
            
        
        
    def affectationInterventionDay(self,listIntervention,nbJours):
        """
        Input: listIntervention une liste des interventions possiblement traitables par ce bateau
        Permet de décider quelles interventions seront effectuées quel jour.
        """
        self.update(listIntervention,nbJours)
        
        
        self.listInterventionsPerDay = list()
        for time in range (nbJours):
            self.listInterventionsPerDay.append(list())
        
        
            
        #Initialisation problème
        
        prob=pu.LpProblem("",pu.LpMinimize)
        
        # Creation des variables et contraintes
        dictVar = dict()
        dictVarCentroid = dict()
        for inter in listIntervention:
            sumVariable = list()
            for time in range (inter.beginning,inter.end+1):
                if self.disponibility[time] == True:
                    variable = pu.LpVariable(name='var_'+str(inter.idInt)+'_'+str(time),cat='Binary')
                    dictVar[(inter.idInt,time)] = variable

                    sumVariable.append((variable,1))
            
            prob += pu.LpConstraint(e=pu.LpAffineExpression( sumVariable ) ,sense=pu.LpConstraintEQ,name='',rhs=1)
            

        
        for time in range(nbJours):
            if self.disponibility[time] == True:
                #rsh = 2 car on ne peut faire que 2 interventions de type 3 dans une même route.
                
                prob += pu.LpConstraint(e=pu.LpAffineExpression( [(dictVar[(inter.idInt,time)],1) for inter in listIntervention if (inter.end>=time and inter.beginning<=time and inter.retrieve == True) ] ),sense=pu.LpConstraintLE,name='',rhs = self.UB)
                
                prob += pu.LpConstraint(e=pu.LpAffineExpression( [(dictVar[(inter.idInt,time)],1) for inter in listIntervention if (inter.end>=time and inter.beginning<=time and inter.retrieve == True) ] ),sense=pu.LpConstraintGE,name='',rhs = self.LB)
                
                if self.type == 3:
                    # si le bateau est un baliseur
                    prob += pu.LpConstraint(e=pu.LpAffineExpression( [(dictVar[(inter.idInt,time)],1) for inter in listIntervention if (inter.end>=time and inter.beginning<=time and inter.retrieve == True) ] ),sense=pu.LpConstraintLE,name='',rhs = 2)
                if self.type == 2:
                    prob += pu.LpConstraint(e=pu.LpAffineExpression( [(dictVar[(inter.idInt,time)],1) for inter in listIntervention if (inter.end>=time and inter.beginning<=time and inter.retrieve== True) ] ),sense=pu.LpConstraintLE,name='',rhs = 1)
                
                
                prob += pu.LpConstraint(e=pu.LpAffineExpression( [(dictVar[(inter.idInt,time)],inter.time) for inter in listIntervention if (inter.end>=time and inter.beginning<=time) ] ),sense=pu.LpConstraintLE,name='',rhs = TIMEBOAT  )

                
        sumVariable=list()
        for inter in listIntervention:
            for time in range (inter.beginning,inter.end+1):
                if self.disponibility[time] == True:
                    sumVariable.append( (dictVar[(inter.idInt,time)],self.euclidean(inter.coord[0],inter.coord[1],self.centroid[time][0],self.centroid[time][1])) )
        prob.setObjective(pu.LpAffineExpression( sumVariable ) )
        
        # résolution
        # TODO attention ici on utilise un solveur commercial il faudra changer
        #prob.writeLP("Test.mst")
        pu.solvers.GUROBI_CMD(path=None, keepFiles=0, mip=1, msg=0, options=[]).actualSolve(prob)
        print("Status:", pu.LpStatus[prob.status])
        

        # prob.solve()
        # print("Total Cos = ", pu.value(prob.objective))
        
        # lecture de la résolution du problème
        
        for inter in listIntervention:
            sumVariable=list()
            for time in range (inter.beginning,inter.end+1):
                if self.disponibility[time] == True:
                    if ( dictVar[(inter.idInt,time)].varValue==1 ):
                        self.listInterventionsPerDay[time].append(inter.idInt)
                        inter.setInterTime(time)
                        inter.setInterBoat(self.num)
                        #print("Les affectation :" ,inter.idInt,time)
                        
                
        if pu.LpStatus[prob.status] == "Not Solved":
            print(self.listInterventionsPerDay)
                    
        return
      
            
            
            
            
