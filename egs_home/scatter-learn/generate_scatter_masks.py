#!/usr/bin/python3

import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from scatter_mask_functions import *
import functools

# 0 [1,0,0]
# 1 [0,1,0]
# 2 [0,0,1]
# 3 [1,1,0]
# 4 [1,0,1]
# 5 [0,1,1]
# 6 [1,1,1]
# 7 [0,0,0]
try:
    rand_num=int(sys.argv[1])

except IndexError:
    print("no pattern index provided, defaulting to 0")
    1/0 ## likely empty Random number..
    mat_ind=0

# Starts at [-4, -4, 0]
indf = lambda rn: [xindf(rn),
                   yindf(rn),
                   zindf(rn)
]

def CountNumberOfMaterials(header):
    N_mats = header.split("media = ")[-1]
    N_mats = N_mats.split(", ")
    N_mats = len(N_mats)
    return N_mats

def index_input(rand_num):
    header = "media = air, graphite, aluminum, lead"  ##header to include-media file
    print(header)  ##header to include-media file
    N_mats = CountNumberOfMaterials(header) ## Number of materials used
    l = range(8*8*8)  ## N voxels in scatter medium
    indices = [indf(li) for li in l] #mat_func inputs for each voxel

    base_indexer = base_r(rand_num, base=N_mats)
    mat_func = base_indexer.index
    #binary#mat_func_list[mat_ind] ##Pick mask, stream-edit by sh-script.
    
    # Line by line, print out material value
    with open(f"input_log_test_-_{rand_num}.csv",'w') as input_log:
        for voxel_number, voxel_index in enumerate(indices):  
            # Get voxel material corresponding to its index and the rand_num
            voxel_material = mat_func(voxel_index[0], 
                                      voxel_index[1],
                                      voxel_index[2])
                                      #rand_num)
            input_log.write(f"{voxel_number},"
                              +f"{voxel_index[0]},"
                              +f"{voxel_index[1]},"
                              +f"{voxel_index[2]},"
                              +f"{voxel_material}\n")


    for voxel_number, voxel_index in enumerate(indices):  
        # Get voxel material corresponding to its index and the rand_num
        voxel_material = mat_func(voxel_index[0], 
                                  voxel_index[1],
                                  voxel_index[2])
                                  #rand_num)
        print(f"set medium = {voxel_number} {voxel_material}")

index_input(rand_num)
                   
