#!/bin/sh
dirname=$(echo bfdo_-_$(date -I)_-_$(ls -1 | grep $(date -I) | wc -l))
mkdir $dirname

echo "$dirname" > most_recent_run.txt

mkdir $dirname/og_results/
mkdir $dirname/og_results/csv_dir
mkdir $dirname/og_results/input_test_dir
mkdir $dirname/og_results/egsinp_dir
mkdir $dirname/og_results/egsdat_dir
mkdir $dirname/og_results/egslog_dir
mkdir $dirname/og_results/include_dir
mkdir $dirname/og_results/ptracks_dir
mkdir $dirname/og_results/mederr_dir
mkdir $dirname/og_results/input_log_dir


python -i BruteForce_Design_Optimization.py 
