#!/bin/sh

N_sample=10
N_mat=4
run_dir=$(cat most_recent_run.txt)
for n in $(seq $N_sample)
do
	rn=$(./rand.py $N_mat)
	IsPresent=$(ls -1 $run_dir/og_results/csv_dir | grep "$rn")
	if (test -z $IsPresent)
	then
		echo $rn >> test_random_numbers.txt
	fi
done

for rand_num in $(cat test_random_numbers.txt)
do
	./compare_model_to_distribution.py "$rand_num"
done
