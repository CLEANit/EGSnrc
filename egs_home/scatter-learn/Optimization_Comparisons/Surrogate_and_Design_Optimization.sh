#!/bin/sh


dirname=$(echo run_-_$(date -I)_-_$(ls -1 | grep $(date -I) | wc -l))

mkdir $dirname

# Write destination dir for sweep_main.sh
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


# Collect random numbers
N_sample=$1
N_sample=${N_sample:=100}

N_Mat=$(./misc/Get_N_Mat)
for n in $(seq $N_sample)
do
	rn=$(python ./rand.py $N_Mat)
	echo $rn >> random_numbers
	#sbatch --partition=lpmoonracer ./sweep_main.sh $rn #$xs $ys $zs $calc_n
done

# Keep only distinct random numbers
sort -un random_numbers -o random_numbers 

# For each random number, run simulation
#for rn in $(cat random_numbers)
#do
#	./sweep_main.sh $rn
#done

# In parallel, pass each number to sweep_main
cat random_numbers | parallel ./sweep_main.sh
rm random_numbers

cp include* $dirname/og_results/
###./put_results_away.sh
#
##run_dir=$(ls -1rtF | grep "/" | tail -n 1)
##cd $run_dir
cd $dirname/og_results/csv_dir
cp ../../../egs_data_collect.py ./

# Collect data
python ./egs_data_collect.py
rm egs_data_collect.py
mv *.hdf5 ../..
cd ../..

## Copy template input-file for record.
cp ../scatter-learn-placeholder.egsinp ./"$(echo scatter-learn-placeholder_-_"$dirname".egsinp | sed 's|\/\.egsinp|.egsinp|')"

cd ..
#
### Move-in & run ML-surrogate reqs.
#cp ../egs_data_rfr.py ./
#cp ../scope_pattern.py ./
python MakeMod.py $dirname
#
### Copy-in other simulation setup. Both for visulization and for record.
#
### Return to ./scatter-learn/
#cd ..

python Surrogate_and_Design_Optimization.py

#sleep 100 && vlc --no-fullscreen dvdsimple://dev/sr0
