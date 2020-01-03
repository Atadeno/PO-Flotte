# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt # plotting data
from shapely.geometry import asShape
from collections import OrderedDict


class DataSelector:
    COLOR = {
        True: '#6699cc',
        False: '#ffcc33'
    }

    def v_color(self,ob):
        return self.COLOR[ob.is_simple]

    FATL = ['Atlantique', 'Seine', 'Garonne', 'Dordogne', 'Arcachon', 'Auray', 'Vannes', 'Rade Brest', 'Crozon',
            'Loire', 'Morlaix', 'Treguier', 'Lezardrieux', 'Rance']

    FMED = ['Sud Méditerranée', 'Etang de Thaux', 'Rhone', 'Etang de Berre', 'Rade de Toulon','Fos', 'Corse','Sud MÃ©diterranÃ©e']
    HEADER = ['NOM', 'LATITUDE', 'LONGITUDE','NOM_SUBDI' ,'DEPARTEMENT_SUBDI','ACCES','NUM','TYPE_SUPPORT','POIDS_CORPS_MORT_DANS_AIR',"ACTIF","STATUT"]
    
    DEPARTEMENT = {"AUDE":'11',"ALPES MARITIMES":'6',"BOUCHES DU RHONE":'13',"CALVADOS":'14',"CHARENTE MARITIME":'17',"CORSE DU SUD":'2A',"COTES D'ARMOR":'22',"FINISTERE":'29',"GARD":'30',"GIRONDE":'33',"HAUTE CORSE":'2B',"HERAULT":'34',"ILLE ET VILAINE":'35',"LANDES":'40',"LOIRE ATLANTIQUE":'44',"MANCHE":'50',"MORBIHAN":'56',"NORD":'59',"PAS DE CALAIS":'62',"PYRENEES ATLANTIQUES":'64',"PYRENEES ORIENTALES":'66',"SEINE MARITIME":'76',"SOMME":'80',"VAR":'83',"VENDEE":'85'}
    
    DOM = ['503', '971', '972', '973', '974', '975', '976']
    DME = ['6', '83', '13', '30', '34', '11', '66', '2A', '2B']

    DAT = ['62', '80', '76', '14', '50', '50A', '50B', '35', '22','29', '29A', '29B', '56', '44', '85', '17', '33', '40', '64','59']
    
    NAT = ['LORIENT','BAYONNE','BOULOGNE','BREST','CHERBOURG','CONCARNEAU','DUNKERQUE','GRANVILLE','LA ROCHELLE','LE HAVRE','LE VERDON',"""LES SABLES D'OLONNE""","LEZARDRIEUX",'SAINT-MALO','ST NAZAIRE','ST VALERY SUR SOMME','LE VERDON']
    NME = ['AJACCIO','BASTIA','CANNES','MARSEILLE','SETE','TOULON']

    def __init__(self, traits, esm,traits20NM=None,depthFile="../../../data/resultat/depthNM.txt",depthPBMAfile = "../../../data/resultat/profondeurPBMASup1Atl.csv"):
        self.traits = traits
        self.esm = esm
        self.traits20NM=traits20NM
        self.depthPBMAfile = depthPBMAfile
        
        depth=open(depthFile)
        tabDepth=depth.readlines()
        for index in range (len(tabDepth)):
            tabDepth[index]=tabDepth[index].split()
            
            
        self.tabDepth = tabDepth

    def port(self,area='ATL'):
        HEADERPORT=['NOM_SUBDI','LONGITUDE_SUBDI','LATITUDE_SUBDI','DEPARTEMENT_SUBDI']
        port=self.esm[HEADERPORT]
        port['DEPARTEMENT_SUBDI'] = port['DEPARTEMENT_SUBDI'].astype(str)
        port=port.drop_duplicates(subset=HEADERPORT)
        
        if area == 'ATL':
            port = port[port.DEPARTEMENT_SUBDI.isin(self.DAT)]
        elif area == 'MED':
            port = port[port.DEPARTEMENT_SUBDI.isin(self.DME)]
        allport=OrderedDict()
        for index,row in port.iterrows():
            if (row[1],row[2]) not in allport.keys():
                allport[(row[1],row[2])]=row[0]
        return allport

    def select(self, area="ATL", transfeu=True):
        """
        Filtre les données pour récupérer les côtes et ESM
        :param area: aire considérée, parmi {"FR","ATL", "MED"}
        :return: les features des cotes considérées et les esm inclus
        """
        if self.traits20NM == None:
            return self.__select_traits(area), self.__select_esem(area, transfeu)
        return self.__select_traits(area), self.__select_esem(area, transfeu),self.__select_traits20NM(area)

    def __select_traits20NM(self,area):
        # filtre les features par rapport à l'aire considérée
        for feat in self.traits20NM["features"]:
            if area == 'ATL':
                return self.traits20NM["features"][0]
            if area == 'MED':
                return self.traits20NM["features"][1]


    def __select_traits(self, area):
        # filtre les features par rapport à l'aire considérée
        cotes = []
        for feat in self.traits["features"]:
            if area == 'ATL':
                if feat["properties"]["Nom"] not in self.FATL:
                    continue
            if area == 'MED':
                if feat["properties"]["Nom"] not in self.FMED:
                    continue
            cotes.append(feat)
        return cotes


    def __select_esem(self, area, transfeu = True):
        # filtre les ESM en fonction de l'aire considéreé
        dataframe = self.esm[self.HEADER]
        dataframe['NUM'] = dataframe['NUM'].astype(str)
        # Rajouter un 0 pour les ems à Cannes
        
        
        dataframe['NUM'] = dataframe[['NUM','NOM_SUBDI']].apply(lambda x: x.NUM if (x.NOM_SUBDI != "CANNES" or x.NUM[0]=='0') else ('0'+x.NUM)  ,axis = 1 )
        
        
        dataframe['DEPARTEMENT_SUBDI'] = dataframe['DEPARTEMENT_SUBDI'].astype(str)

        
        
        dataframe['LONGITUDE'] = dataframe['LONGITUDE'].apply( lambda x:self.__transform(x) )
        dataframe['LATITUDE'] = dataframe['LATITUDE'].apply( lambda x:self.__transform(x) )
        dataframe['DEPARTEMENT_SUBDI'] = dataframe['DEPARTEMENT_SUBDI'].apply(lambda x:self.DEPARTEMENT[x] if x in self.DEPARTEMENT.keys() else x)
        
        
        dataframe = dataframe[~(dataframe.LATITUDE.isin([0]) & dataframe.LONGITUDE.isin([0]))]
        # filtre les ESM des DOM
        dataframe = dataframe[~dataframe.DEPARTEMENT_SUBDI.isin(self.DOM)]
        # filtre les ESM de l'aire considérée
        dataframe = dataframe[dataframe.ACCES.isin(["Maritime","Selon marées"])]
        
        # On enlève les ESM à Bayonne autre que Bouée
        dfBayonne = dataframe[ dataframe.NOM_SUBDI.isin(["BAYONNE"])]
        dfBayonne = dfBayonne[ ~(dfBayonne.TYPE_SUPPORT.isin(["Bouée"])) ]
        dataframe = dataframe[ ~(dataframe.NUM.isin(dfBayonne.NUM))]   
             
        # On enlève le balisage de polise, les projets et Operation
        dataframe = dataframe[ ~(dataframe.STATUT.isin(["Projet","Opération","Balisage de police"]))]
        
        if area == 'ATL':
            dataframe = dataframe[dataframe.DEPARTEMENT_SUBDI.isin(self.DAT)]
            
            dfPBMA = pd.read_csv(self.depthPBMAfile)
            
            # Tri pour enlever les esms qui sont accessibles à marée basse. 
            dataframe = dataframe[~(dataframe.NUM.isin(dfPBMA["NUM"]))]
            
        elif area == 'MED':
            dataframe = dataframe[dataframe.DEPARTEMENT_SUBDI.isin(self.DME)]
        dataframe = dataframe.drop_duplicates()
        
        #print(len(dataframe.index))
        
        # filtre sur les Feux et les Amers
        if transfeu:
            dataframe['TYPE_SUPPORT'] = dataframe[["NOM",'TYPE_SUPPORT',"STATUT"]].apply( lambda x : self.__transformFeu(x.NOM,x.TYPE_SUPPORT,x.STATUT),axis = 1 )

        return dataframe
        
    def __transformFeu(self,nom,typeEsm):
        
        if typeEsm == "Amer":
            # espar passif
            return "Balise/espar"

        elif typeEsm == "Feu":
            nom = nom.lower()
            if "digue" in nom or "ecluse" in nom or "écluse" in nom or "port" in nom or "jetée" in nom or "jetee" in nom or "quai" in nom  or "môle" in nom or "estacade" in nom or "ponton" in nom or "passerelle" in nom or "appontement" in nom :
                return "Balise/espar"
            else:
                return "Tourelle"
        else:
            return typeEsm
                
    
    def __transform(self,x):
        """
        Transforme les longitudes/latitudes sexagésimal(str) en longitude/latitude en degré,
         si x est deja un entier/float ne fait rien.
        """
        if type(x) is str:
            deg,x = x.split("°")
            
            deg = int(deg)
            
            split = x.split(",")
            if len(split)==2:
                millier,x = split
                millier += "000"
                millier = int(millier)
                centaine,direction = x.split(" ")
            else:
                x = split[0]
                millier,direction = x.split(" ")
                millier += "000"
                millier = int(millier)
                centaine="0"

            if len(centaine)==0:
                centaine += "000"
            elif len(centaine)==1:
                centaine +=   "00"
                
            elif len(centaine)==2:
                centaine +=  "0"
            
            centaine = int(centaine)
            
            sign = 0

            if direction=='n' or direction=='e':
                sign = 1
            else:
                sign = -1
                
            
                
            return sign*(deg + (millier + centaine)/60000)

                
        else:
            return x
        
    #print(__transform("2°17,9 w"))

    def plot(self,area="ATL",saveFig=True):
        cotes, esm = self.select(area)
        # initiate the plot axes
        fig = plt.figure(figsize=[15, 15])  # create a figure to contain the plot elements
        ax = fig.gca(xlabel="Longitude", ylabel="Latitude")
        # loop through the features plotting polygon centroid

        for feat in cotes:
            # convert the geometry to shapely
            geom = asShape(feat["geometry"])
            nom = feat["properties"]
            print("%s - %s" % (nom, geom))
            x, y = geom.xy
            ax.plot(x, y, color=self.v_color(geom), alpha=1, linewidth=1, solid_capstyle='round', zorder=2)

        # loop through the coordinates
        for _, row in esm.iterrows():
            n = row[0]
            lat = row[1]
            lon = row[2]
            # print("%s (%.2f, %.2f)"%(n, lat, lon))
            ax.plot(lon, lat, '+', color='#999999', zorder=1)
        if saveFig:
            plt.savefig(area + '.pdf')
        plt.show()
