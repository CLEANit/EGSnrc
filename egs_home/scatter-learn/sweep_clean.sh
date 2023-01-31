#!/bin/sh

GenModId () {
	rand_num=$1
	ith_run=$(ls -1 | grep "$rand_num" | grep "detector_results.csv" | wc -l)
	echo $ith_run
}

# results come in as string... more than one value. Need seperation
results=$1
rand_num=$(echo $results | sed 's/\ .*//')
id=$(echo $results | sed 's/.*\ //')

#echo "rand num:$rand_num"
#echo "id:$id"

if (test -n "$rand_num")
then
	if (test -n "$id")
	then
		## leave as csv?
		## dirname=$(ls -rt1 | grep $(date -I) | tail -n 1)
		dirname=$(cat most_recent_run.txt)
		cd $dirname
		cd og_results
		mv ../../scatter-learn_-_"$id"_-_detector_results.csv csv_dir
		mv ../../scatter-learn_-_"$id"_-_scatter_input.csv csv_dir
		mv ../../scatter-learn_-_"$id".egsinp egsinp_dir
		mv ../../scatter-learn_-_$id.egsdat egsdat_dir
		mv ../../scatter-learn_-_$id.egslog egslog_dir
		mv ../../scatter-learn_-_"$id"_-_include_scatter_media.dat include_dir
		#mv ../../scatter-learn_-_$id.ptracks ptracks_dir
		rm ../../scatter-learn_-_$id.ptracks 
		mv ../../scatter-learn_-_$id.mederr mederr_dir
		mv ../../input_log_test_-_$rand_num.csv input_log_dir
		#mv ../../input_log_test_-_"$rand_num".csv csv_dir
		cd ../..
	fi
fi

	
