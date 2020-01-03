import sys
sys.path.append('../../')

import selector as sel
import InstanceGeneration as ig

import computeDistance as cd
import math
import random

class Port():
    
    
    def __init__(self,port,id,id20NM,nbJours):
        
        
        self.id = [id,id20NM]
        self.nom = port["NOM_SUBDI"]
        self.lon = port["LONGITUDE_SUBDI"]
        self.lat = port["LATITUDE_SUBDI"]
        self.coord = [self.lon,self.lat]
        self.nbJours = nbJours
        
        # Initialisation de la liste des Centroids
        self.centroid=list()
        for day in range (nbJours):
            self.centroid.append(self.coord)

            
            

    def setZone(self,indZone,indPlace):
        self.indZone = indZone
        self.indPlace = indPlace
        

        
        
        
        
        
        
        
        

