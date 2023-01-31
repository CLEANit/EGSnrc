#!/bin/sh

# Produce training data in parallel and train the model. Measures and prints the 
# coefficient of determination, describing the amount of variance in the test-set
# that is described by the model
./ParallelSweep.sh

# Repeatedly simulate a single candidate design, and compare the likelihood of the 
# Surrogate model's prediction to that characteristic of singular simulation results. 
./ConsiderSurrogateLikelihood.sh


echo "Would you like to continue with optimization comparison [y/N]? "
read -r response
#echo $response

