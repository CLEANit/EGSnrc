import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from scatter_mask_functions import *
import functools

from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import matplotlib.pyplot as plt

cl = ['k','y']#,'r','b']
color_list = ['k','y']#,'r','b']

color_func_list = [functools.partial(functools.partial(slabs,i=0),j=0),
                   functools.partial(functools.partial(slabs,i=1),j=0),
                   functools.partial(functools.partial(slabs,i=2),j=0),
                   functools.partial(functools.partial(columns,i=0),j=1),
                   functools.partial(functools.partial(columns,i=0),j=2),
                   functools.partial(functools.partial(columns,i=1),j=2),
                   functools.partial(functools.partial(checkers,i=1),j=2)]
                   

for cix in range(len(color_func_list)):
    color_func=color_func_list[cix]

    ax = plt.figure().gca(projection='3d')
    xrm = [-4,4]
    yrm = [-4,4]
    zrm = [-4,4]
    mx = 2
    my = 2
    mz = 2
    onez = np.array([[1,1],[1,1]])
    for xi in [xrm[0],xrm[1]-1]:#range(-4,4):
        xr = [xi,xi+1]
        for yi in range(yrm[0],yrm[1]):
            yr = [yi,yi+1]
            for zi in range(zrm[0],zrm[1]):
                zr = [zi,zi+1]
                ax.plot_surface(xr[0]*onez ,np.vstack((yr,yr)) ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(xr[1]*onez ,np.vstack((yr,yr)) ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,yr[0]*onez ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,yr[1]*onez ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,np.dstack((yr,yr))[0] ,zr[0]*onez ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,np.dstack((yr,yr))[0] ,zr[1]*onez ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
    
    for yi in [yrm[0],yrm[1]-1]:#range(-4,4):
        yr = [yi,yi+1]
        for xi in range(xrm[0]+1,xrm[1]-1):
            xr = [xi,xi+1]
            for zi in range(zrm[0],zrm[1]):
                zr = [zi,zi+1]
                ax.plot_surface(xr[0]*onez ,np.vstack((yr,yr)) ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(xr[1]*onez ,np.vstack((yr,yr)) ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,yr[0]*onez ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,yr[1]*onez ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,np.dstack((yr,yr))[0] ,zr[0]*onez ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,np.dstack((yr,yr))[0] ,zr[1]*onez ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
    
    for zi in [zrm[0],zrm[1]-1]:#range(-4,4):
        zr = [zi,zi+1]
        for yi in range(yrm[0]+1,yrm[1]-1):
            yr = [yi,yi+1]
            for xi in range(xrm[0]+1,xrm[1]-1):
                xr = [xi,xi+1]
                ax.plot_surface(xr[0]*onez ,np.vstack((yr,yr)) ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(xr[1]*onez ,np.vstack((yr,yr)) ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,yr[0]*onez ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,yr[1]*onez ,np.dstack((zr,zr))[0] ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,np.dstack((yr,yr))[0] ,zr[0]*onez ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
                ax.plot_surface(np.vstack((xr,xr)) ,np.dstack((yr,yr))[0] ,zr[1]*onez ,color=color_list[int(color_func([xi,yi,zi],[mx,my,mz],2))])
    
    ax.set_xlim(xrm)
    ax.set_ylim(yrm)
    ax.set_zlim(zrm)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
plt.show()
