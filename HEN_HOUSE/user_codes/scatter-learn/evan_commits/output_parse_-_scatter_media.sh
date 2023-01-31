#!/bin/sh

## Receive file
scatter_media_file=$1

## Scrape id
id=$(echo $scatter_media_file | sed 's/include-scatter-media_-_//' | sed 's/\.dat//')

## nrows of data (ignoring first line)
nrow=$(($(cat $scatter_media_file | wc -l)-1))

## Grab material definitions
tail -n $nrow $scatter_media_file | rev | sed 's/\ .*//' > scatter_input_-_inter_mats_-_$id.csv

## Convert index to position 
tail -n $nrow $scatter_media_file | awk -F " " '{print ($4%8-4), (int(($4%64)/8)-4), int($4/64)}' > scatter_input_-_inter_pos_-_$id.csv

## dstack data
paste scatter_input_-_inter_pos_-_$id.csv scatter_input_-_inter_mats_-_$id.csv > "scatter-learn_-_"$id"_-_scatter_input.csv"

## Clean up
rm scatter_input_-_inter_pos_-_$id.csv
rm scatter_input_-_inter_mats_-_$id.csv

## Convert to csv
sed -i 's/\s/,/g' scatter-learn_-_$id"_-_scatter_input.csv"
