#!/usr/bin/python3

import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from scatter_mask_functions import *
import functools

mat_func_list = [functools.partial(functools.partial(slabs,i=0),j=0),
                   functools.partial(functools.partial(slabs,i=1),j=0),
                   functools.partial(functools.partial(slabs,i=2),j=0),
                   functools.partial(functools.partial(columns,i=0),j=1),
                   functools.partial(functools.partial(columns,i=0),j=2),
                   functools.partial(functools.partial(columns,i=1),j=2),
                   functools.partial(functools.partial(checkers,i=1),j=2)]
try:
    mat_ind=int(sys.argv[1])
except IndexError:
    print("no pattern index provided, defaulting to 0")
    mat_ind=0


xindf = lambda x: (x%8-4)
yindf = lambda y: (int((y%64)/8)-4)
zindf = lambda z: int(z/64)
indf = lambda x: [(x%8-4),(int((x%64)/8)-4),int(x/64)]


def index_input():
    print("media = vacuum, aluminum, air")  ##header to include-media file
    n=2 ## Number of materials used
    l = range(8*8*8)  ## N voxels in scatter medium
    homog_block_size = [1,1,1] ## blocks of one material will be no smaller in [x,y,z]
    indl = [[indf(li),homog_block_size,n] for li in l] #mat_func inputs for each voxel
    mat_func=mat_func_list[mat_ind] ##Pick mask, stream-edit by sh-script.
    for i,x in enumerate(indl):  ## Line by line, print out material value
        print("set medium = "+str(i)+" "+str(int(mat_func(x[0],x[1],x[2]))))

index_input()
                   
