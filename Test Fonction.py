import numpy as np
import pandas as pd

data = np.array([[43.6, 2.8,10000000000000000000], [6.0, 2.2,300]])

# Creating pandas dataframe from numpy array
dataset = pd.DataFrame({'Column1': [data[:, 0],data[:, 1]],'taille':data[:, 2]})
print(dataset)
#fig = px.scatter_geo(dataset, locations="Column1",size='taille')
#fig.show()

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# setup Lambert Conformal basemap.
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
# draw coastlines.
m.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
m.drawmapboundary(fill_color='aqua')
# fill continents, set lake color same as ocean color.
m.fillcontinents(lake_color='aqua')
plt.show()

