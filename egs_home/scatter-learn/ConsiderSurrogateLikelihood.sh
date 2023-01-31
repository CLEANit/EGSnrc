#!/bin/sh

rand_num=$1
N_mat=$(./misc/Get_N_Mat.sh)
rand_num=${rand_num:=$(python ./rand.py $N_mat)}
echo $rand_num

clean_up () {
	id_mod=$1
	rand_num=$2
	rm "scatter-learn_-_num_$id_mod-$rand_num""_-_include_scatter_media.dat"
	rm scatter-learn_-_num_$id_mod-$rand_num.egsinp
	rm scatter-learn_-_num_$id_mod-$rand_num.mederr
	rm scatter-learn_-_num_$id_mod-$rand_num.egsdat
	rm scatter-learn_-_num_$id_mod-$rand_num.ptracks
	rm scatter-learn_-_num_$id_mod-$rand_num.egslog
	#rm scatter-learn_-_num_$s-$rand_num_-_detector_results.csv
	#rm "scatter-learn_-_num_$s-$rand_num""_-_scatter_input.csv"
}

simulate_main () {
	id_mod=$1
	rand_num=$2
	echo $id_mod
	#./sweep_noclean.sh $rand_num
	ncase=$(cat .default_ncase)
	sigma=$(cat .default_sigma)
	./sweep_noclean.sh $rand_num $ncase $sigma $id_mod
}

init_distribution () {
	# s must be 0
	rand_num=$1
	simulate_main 0 $rand_num 	
	cp scatter-learn_-_num_0-$rand_num.egsinp $dist_dir
	clean_up 0 $rand_num

	results_name="scatter-learn_-_num_0-$rand_num""_-_detector_results.csv"
	processed_name="scatter-learn_-_num_0-$rand_num""_-_detector_results_3.csv"

	# Init collection (not final file). For the first one, take all of the columns
	awk -F "," '{print $1","$2","$3}' $results_name > $processed_name
	#awk -F "," '{print $1","$2","$4}' $results_name > $processed_name

	# Copy results into all_results
	cp $results_name $dist_dir/all_results/
	mv "scatter-learn_-_num_0-$rand_num""_-_scatter_input.csv" $dist_dir
}

#add_to_distribution () {
#	# Run simulation
#	s=$1
#	simulate_main $s $rand_num	
#	clean_up $s $rand_num
#
#	# Clean up duplicated input.
#	rm "scatter-learn_-_num_$s-$rand_num""_-_scatter_input.csv"
#
#	results_name="scatter-learn_-_num_$s-$rand_num""_-_detector_results.csv"
#	processed_name="scatter-learn_-_num_$s-$rand_num""_-_detector_results_3.csv"
#
#	# Take only the target column
#	awk -F "," '{print $3}' $results_name > $processed_name
#	cp $results_name $dist_dir/all_results/
#}



N_sample=100
#N_sample=$(($N_sample-1))

dist_dir="$rand_num""_-_detector_results"
mkdir $dist_dir
mkdir $dist_dir/all_results

init_distribution $rand_num
#for s in $(seq 1 $N_sample)
#do
#	add_to_distribution $s $rand_num
#	## only detector_results_remain
#	#	> col 3 Edep [MeV*cm2] 
#	#	> col 4 D/[Gy*cm2]
#	#awk -i inplace -F "," '{print $3}' scatter-learn_-_num_$s-$rand_num_-_detector_results.csv
#	
#done

echo "$rand_num" > distribution_random_number.txt
seq 1 $N_sample | parallel ./results_distribution_scripts/add_to_distribution.sh


paste -d ","  $(ls -1 *-_detector_results_3.csv | sort -n) > column_3.csv


# Ensure that $rand_num isn't empty -> would delete entire director.
if (test -n "$rand_num")
then
	#rm *-$rand_num*
	mv *-$rand_num* $dist_dir/all_results
fi
mv column_3.csv $dist_dir

#./compare_model_to_distribution.py $rand_num
python ./Bootstrapped_Likelihood_Analysis.py $rand_num

mv ./distribution_random_number.txt $dist_dir
