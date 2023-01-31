#!$(which python)
##!/home/user/miniconda3/envs/skmod/bin/python

from random import randint
import sys
n_mats = sys.argv[1]
n_mats = int(n_mats)
#try:
#    n_mats = sys.argv[1]
#except:
#    n_mats = 3

rn = randint(0, n_mats**64)
print(rn)

