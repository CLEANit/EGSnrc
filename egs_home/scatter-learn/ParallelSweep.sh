#!/bin/sh

edit_sweep_main  () {
  # Ncase is dealt with in ./sweep_main.sh
  #       > necessary to have only a single variable input for 
  # 	  parallelization.
  ncase=$1
  sigma=$2
  cp ./misc/sweep_main.sh.backup sweep_main.sh
  sed -i "s/ncase_main_placeholder/$ncase/" sweep_main.sh
  sed -i "s/sigma_main_placeholder/$sigma/" sweep_main.sh

}

date >> surrogate_optimization_prep.txt

ncase=$1
ncase_bk=$(cat .default_ncase)
ncase=${ncase:=$ncase_bk}

sigma=$2
sigma_bk=$(cat .default_sigma)
sigma=${sigma:=$sigma_bk}

N_sample=$3
N_sample_bk=$(cat .default_Nsample)
N_sample=${N_sample:=$N_sample_bk}

edit_sweep_main $ncase $sigma

dirname="run_-_ncase_$ncase""_-_sigma_$sigma""_-_Nsample_$N_sample"
#$(date -I)_-_$(ls -1 | grep $(date -I) | wc -l))
echo $dirname
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
N_Mat=$(./misc/Get_N_Mat.sh)
touch random_numbers
rm random_numbers
for n in $(seq $N_sample)
do
	rn=$(python ./rand.py $N_Mat)
	echo $rn >> random_numbers
done

# Keep only distinct random numbers
sort -un random_numbers -o random_numbers 

# For each random number, run simulation
#for rn in $(cat random_numbers)
#do
#  ./sweep_main.sh $rn
#  sbatch --partition=lpmoonracer \
#		./sweep_main.sh $rn #$xs $ys $zs $calc_n
#done

# In parallel, pass each number to sweep_main
cat random_numbers | parallel ./sweep_main.sh
mv random_numbers $dirname

cp include* $dirname/og_results/

cd $dirname/og_results/csv_dir
cp ../../../egs_data_collect.py ./

# Collect data
python ./egs_data_collect.py
rm egs_data_collect.py
mv *.hdf5 ../..
cd ../..

## Copy template input-file for record.
input_file_name=$(echo scatter-learn-placeholder_-_"$dirname".egsinp \
       	  	   | sed 's|\/\.egsinp|.egsinp|')
cp ../scatter-learn-placeholder.egsinp ./"$input_file_name"

# Return to main, and train new surrogate model
cd ..
python MakeMod.py $dirname $ncase $sigma $N_sample 

## Copy-in other simulation setup. Both for visulization and #
#  for record.
date >> surrogate_optimization_prep.txt
