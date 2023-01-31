import matplotlib as mpl
import numpy as np

def MakeMaterialsColorbar(input_data_train, input_data_test, fig):
    #cmap = mpl.cm.viridis
    cmap = mpl.cm.cividis
                     #viridis

    bounds1 = np.unique(input_data_train)
    bounds2 = np.unique(input_data_test)
    bounds = set(bounds1).union(bounds2)
    bounds = sorted(bounds)
    #bounds = [min(bounds) - 1] + bounds
    bounds =  bounds + [max(bounds) + 1]

    norm = mpl.colors.BoundaryNorm(bounds,
                                   cmap.N)#,
                                   #extend='both')

    mappable = mpl.cm.ScalarMappable(norm=norm,
                                     cmap=cmap)

    #cb = fig.colorbar(mappable, cax=cb_ax, orientation="horizontal")

    ticks = MakeTicks(bounds)
    #cb.set_ticks(ticks)
    return norm, cmap, mappable, ticks



def MakeTicks(bounds):
    bounds = np.array(bounds)
    ticks = bounds[:-1] + np.diff(bounds)/2
    return ticks

