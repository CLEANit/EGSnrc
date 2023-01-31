#!/bin/sh

single_call () {
	ncase=$1
	sigma=$2
	N_sample=$3	
	collection_dir=$4

	if test -z "$ncase"
	then
		echo ncase empty
		exit 1
	fi
	
	if test -z "$sigma"
	then
		echo sigma empty
		exit 1
	fi
	
	./ParallelSweep.sh $ncase $sigma $N_sample
	
	mv evp_-_ncase* $collection_dir
	mv mod_-_ncase* $collection_dir
	mv work_flow_-_ncase* $collection_dir
	mv run_-_ncase* $collection_dir
}


collection_dir="varrep_-_$(date -I)_-_$(ls -1 | grep $(date -I) | wc -l)"
mkdir $collection_dir

echo "ncase,sigma,Nsample,r2" > model_r2.csv
#for ncase in 1000 10000 100000
for ncase in 5000 50000
##for ncase in 100 500 1000 5000 10000 50000 100000 #500000
do
	#for sigma in 0.01 0.1 1
	for sigma in 0.05 0.5
	do
		for N_sample in 10 100 1000
		do	
			single_call $ncase $sigma $N_sample $collection_dir
		done
	done
done
mv model_r2.csv $collection_dir
##
