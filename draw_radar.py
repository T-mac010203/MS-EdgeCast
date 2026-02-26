import numpy as np
#from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def draw(data, path):
    cdict = ['whitesmoke', 'dodgerblue', 'limegreen', 'green', 'darkgreen',
             'yellow', 'goldenrod', 'orange', 'red', 'darkred']
    clevs = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    my_map = colors.ListedColormap(cdict)
    norm = colors.BoundaryNorm(clevs, len(clevs) - 1)
    
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    
    im = ax.imshow(data,cmap=my_map, norm=norm)
    
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(path, dpi=100)
    plt.close()
