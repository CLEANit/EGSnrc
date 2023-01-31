#!/usr/bin/python

# usage: generate-scatter-media.py > include-scatter-media.dat

# imports
from random import randint, seed

# change seed to change random voxel media arrangement
seed(1)

# list of media
print "media = air water aluminum"

# generate random voxel list
for i in range(8*8*8):
    print "set medium = ", i, randint(0,2)