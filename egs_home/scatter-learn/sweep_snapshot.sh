#!/bin/sh

GenModId () {
	ith_run=$(ls -1 | grep "$1" | grep "detector_results.csv" | wc -l)
	echo $ith_run
}
	

## Recieve vars
rand_num=$1
id_mod=$(GenModId $rand_num)

it_turn=$2

#clean_up=$2
#clean_up={$clean_up:=""}

## Set variables to defaults if null
ncase=10000
#pattern_ind=${pattern_ind_input:=0}

## Define id and file names
if test -n $id_mod
then
	id="num_$id_mod-$rand_num"
else
	id="num_$rand_num"
fi

input_file=$(echo "scatter-learn_-_"$id".egsinp")

## Make geometric mask in include-scatter-media_-_$$.dat file
./generate_scatter_masks.py $rand_num > scatter-learn_-_"$id"_-_include_scatter_media.dat

# Create random seeds
rnd_seeds=$(./make_rnd_seed.sh 2)

## Make and edit custom input file to include new media definitions
## Fails off
cp scatter-learn-placeholder.egsinp $input_file
sed -i "s/placeholder_id/$id/" $input_file
sed -i "s/placeholder_ncase/$ncase/" $input_file
sed -i "s/placeholder_rndseed/$rnd_seeds/" $input_file
## echo $input_file

## Fails on
#cp scatter-learn.egsinp $input_file
#sed -i "s/ncase\ =\ .*/ncase = $ncase/" $input_file
#sed -i "s/scatter-media\scatter-media_-_$id/" $input_file

## Run scatter-learn,
scatter-learn -i $input_file > scatter-learn_-_$id.egslog 

## parses outputs (of detector) into .csv
./output_parse_-_detector.sh scatter-learn_-_$id.egslog 

## parses ~outputs~ (scatter input) into .csv
./output_parse_-_scatter_media.sh scatter-learn_-_"$id"_-_include_scatter_media.dat 
#
./snapshot_EGSview.sh $rand_num $it_turn
echo $rand_num $id

## leave as csv?
## dirname=$(ls -rt1 | grep $(date -I) | tail -n 1)
#dirname=$(cat most_recent_run.txt)
#cd $dirname
#cd og_results
#mv ../../scatter-learn_-_"$id"_-_detector_results.csv csv_dir
#mv ../../scatter-learn_-_"$id"_-_scatter_input.csv csv_dir
#mv ../../input_log_test_-_"$rand_num".csv csv_dir
#mv ../../$input_file egsinp_dir
#mv ../../scatter-learn_-_$id.egsdat egsdat_dir
#mv ../../scatter-learn_-_$id.egslog egslog_dir
#mv ../../scatter-learn_-_"$id"_-_include_scatter_media.dat include_dir
##mv ../../scatter-learn_-_$id.ptracks ptracks_dir
#rm ../../scatter-learn_-_$id.ptracks 
#mv ../../scatter-learn_-_$id.mederr mederr_dir
#cd ../..
