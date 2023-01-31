#!/bin/sh

egslog_file=$1
id=$(echo $egslog_file | sed 's/scatter-learn_-_//' | sed 's/\.egslog//')
#echo "detector: $id"

## Get results for simulation
grep germanium_ $egslog_file | grep "%" | awk '{print $5, $7/100*$5}' > "scatter-learn_-_"$id"_-_detector_results.csv"

## Get indices
grep germanium_ $egslog_file | grep "%" | awk '{print $1-4097}' > detector_results_-_indices_-_$id.csv

cat detector_results_-_indices_-_$id.csv | awk '{print ($1%64-31), int($1/64) - 32}' > detector_results_-_pos_-_$id.csv


## Add indices to simulation-results
paste detector_results_-_pos_-_$id.csv "scatter-learn_-_"$id"_-_detector_results.csv" > detector_results_-_inter_-_$id.csv

## Clean up 1
mv detector_results_-_inter_-_$id.csv "scatter-learn_-_"$id"_-_detector_results.csv"
rm detector_results_-_pos_-_$id.csv
rm detector_results_-_indices_-_$id.csv

## Get media numbers
## Does detector media matter? Also haven't been regenerating it so no need for id.
n=$(($(cat include-detector-media.dat | wc -l)-1))
tail -n $n include-detector-media.dat | awk -F " " '{print $5}' > include-detector-media_-_media-def-only_-_$id.dat

## Concatenate
paste "scatter-learn_-_"$id"_-_detector_results.csv" include-detector-media_-_media-def-only_-_$id.dat > detector_results_-_inter_-_$id.csv

## Clean up 
mv detector_results_-_inter_-_$id.csv   "scatter-learn_-_"$id"_-_detector_results.csv" 
rm include-detector-media_-_media-def-only_-_$id.dat 

## Convert to csv
sed -i 's/\s/,/g' "scatter-learn_-_"$id"_-_detector_results.csv"


## mv: cannot stat 'detector_results_-_inter.csv': No such file or directory
