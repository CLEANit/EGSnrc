#!/bin/sh


clean_up () {
	s=$1
	rand_num=$2
	rm "scatter-learn_-_num_$s-$rand_num""_-_include_scatter_media.dat"
	rm scatter-learn_-_num_$s-$rand_num.egsinp
	rm scatter-learn_-_num_$s-$rand_num.mederr
	rm scatter-learn_-_num_$s-$rand_num.egsdat
	rm scatter-learn_-_num_$s-$rand_num.ptracks
	rm scatter-learn_-_num_$s-$rand_num.egslog
	#rm scatter-learn_-_num_$s-$rand_num_-_detector_results.csv
	#rm "scatter-learn_-_num_$s-$rand_num""_-_scatter_input.csv"
}

simulate_main () {
	id_mod=$1
	rand_num=$2
	ncase=$(cat .default_ncase)
	sigma=$(cat .default_sigma)
	./sweep_noclean.sh $rand_num $ncase $sigma $id_mod
}


rand_num=$(cat ./distribution_random_number.txt)
dist_dir="$rand_num""_-_detector_results"
s=$1

# Run simulation
simulate_main $s $rand_num	
clean_up $s $rand_num

# Clean up duplicated input.
rm "scatter-learn_-_num_$s-$rand_num""_-_scatter_input.csv"

results_name="scatter-learn_-_num_$s-$rand_num""_-_detector_results.csv"
processed_name="scatter-learn_-_num_$s-$rand_num""_-_detector_results_3.csv"

# Take only the target column
awk -F "," '{print $3}' $results_name > $processed_name
#awk -F "," '{print $4}' $results_name > $processed_name
cp $results_name $dist_dir/all_results/

