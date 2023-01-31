#!/bin/sh
rand_num=$1
ncase=10000
sigma=0.1
outs=$(./sweep_noclean.sh $rand_num $ncase $sigma)
./sweep_clean.sh "$outs"
