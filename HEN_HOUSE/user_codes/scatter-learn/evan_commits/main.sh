#!/bin/sh

## Recieve vars
ncase_input=$1
pattern_ind_input=$2

### Set variables to defaults if null
ncase=${ncase_input:=10000}
pattern_ind=${pattern_ind_input:=0}

## Define id and file names
id=$(echo "func_"$pattern_ind"_-_ncase_"$ncase)
input_file=$(echo "scatter-learn_-_"$id".egsinp")

## Make geometric mask in include-scatter-media_-_$$.dat file
./generate-scatter-masks.py $pattern_ind > include-scatter-media_-_$id.dat

## Make and edit custom input file to include new media definitions
## Fails off
cp scatter-learn-placeholder.egsinp $input_file
sed -i "s/placeholder_id/$id/" $input_file
sed -i "s/placeholder_ncase/$ncase/" $input_file

## Fails on
#cp scatter-learn.egsinp $input_file
#sed -i "s/ncase\ =\ .*/ncase = $ncase/" $input_file
#sed -i "s/scatter-media\scatter-media_-_$id/" $input_file

## Run scatter-learn,
scatter-learn -i $input_file > scatter-learn_-_$id.egslog 

## parses outputs (of detector) into .csv
./output_parse_-_detector.sh scatter-learn_-_$id.egslog 

## parses ~outputs~ (scatter input) into .csv
./output_parse_-_scatter_media.sh include-scatter-media_-_$id.dat 

## leave as csv?


