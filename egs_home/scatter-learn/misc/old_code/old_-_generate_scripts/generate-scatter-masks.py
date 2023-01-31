#!/usr/bin/python3

import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from scatter_mask_functions import *
import functools

mat_func_list = [functools.partial(functools.partial(slabs,i=0),j=0), #0
                   functools.partial(functools.partial(slabs,i=1),j=0), #1
                   functools.partial(functools.partial(slabs,i=2),j=0), #2 
                   functools.partial(functools.partial(columns,i=0),j=1),#3
                   functools.partial(functools.partial(columns,i=0),j=2),#4
                   functools.partial(functools.partial(columns,i=1),j=2),#5
                   functools.partial(functools.partial(checkers,i=1),j=2),#6
                   functools.partial(functools.partial(null_pat,i=1),j=2)]#7

# 0 [1,0,0]
# 1 [0,1,0]
# 2 [0,0,1]
# 3 [1,1,0]
# 4 [1,0,1]
# 5 [0,1,1]
# 6 [1,1,1]
# 7 [0,0,0]
try:
    mat_ind=int(sys.argv[1])
    x_bls = int(sys.argv[2])
    y_bls = int(sys.argv[3])
    z_bls = int(sys.argv[4])
except IndexError:
    print("no pattern index provided, defaulting to 0")
    mat_ind=0


xindf = lambda x: (x%8-4)
yindf = lambda y: (int((y%64)/8)-4)
zindf = lambda z: int(z/64)
indf = lambda x: [(x%8-4),(int((x%64)/8)-4),int(x/64)]


def index_input():
    header = "media = vacuum, aluminum, air"
    print(header)  ##header to include-media file
    n=header.count(",")+1 ## Number of materials used

    l = range(8*8*8)  ## N voxels in scatter medium
    homog_block_size = [x_bls,y_bls,z_bls] ## blocks of one material will be no smaller in [x,y,z]
    indl = [[indf(li),homog_block_size,n] for li in l] #mat_func inputs for each voxel
    mat_func=mat_func_list[mat_ind] ##Pick mask, stream-edit by sh-script.
    for i,x in enumerate(indl):  ## Line by line, print out material value
        print("set medium = "+str(i)+" "+str(int(mat_func(x[0],x[1],x[2]))))

index_input()
                   
