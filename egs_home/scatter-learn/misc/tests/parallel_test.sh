#!/bin/sh

x=$1
echo $(($x*2))

# CALL THIS FROM THE COMMAND LINE	
# seq 1 10 | parallel ./parallel_test.sh 
