#!/usr/bin/python

# usage: generate-detector-media.py > include-detector-media.dat

# list of media
print "media = germanium_even germanium_odd"

# generate random voxel list
for i in range(64*64):

    # medium toggles between 0 and 1
    medium = i%2

    # reverse for odd rows
    if (i/64 % 2 == 1):
        medium = 1-medium

    print "set medium = ", i, medium