#!/bin/sh

## Original code name
#code_name_og=$(echo $1 | sed 's/\.py//')
code_name_og="scatter_masks.py"
echo $code_name_og

input_name_og="scatter-learn-placholder.egsinp"


## Sweep through func indices, notice, for-variable is a int.
for func_ind in $(cat -n functions_list | sed 's/ .*//')
do
	## Get func_name for title id_name
	func_name=$(head -n $func_ind functions_list | tail -n 1)

	## Sweep through ncases
	for ncases in $(cat ncases_list)
	do
		## id_name should be attached to all outputs
		id_name=$(echo "func_"$func_name"_-_ncases_"$ncases)
		file_name=$(echo "dir_"$code_name_og"_-_"$id_name)
		mkdir $file_name
		cd $file_name
		
		## Copy original code into id_directory
		code_name=$(echo $(echo $code_name_og"_-_"$id_name)".py")
		input_name=$(echo $(echo $input_name_og"_-_"$id_name)".egsinp")
		cp ../$1 $code_name
		##	> Do all code files need to be copied over?
		##	> could change main to ../output_parse_-_ ...
		##	   ~ make sure exact addressing.
	
		## Stream-edit the code and input files to sweep instance
		sed -i "s/placeholder_func_ind/$func_ind/g" $code_name
		sed -i "s/placeholder_ncases/$ncases/g" $input_name

		## Stream-edit main to call new codes 
		##  > uncessary, main takes inputs now
		##      ~ main also has defaults...	
		##      ~ .. won't this not fail if not given code?
		##	~ .. isn't not failing a bad thing?

		## Submit job to queue
		sbatch ../slurm_script.sh main.sh $code_name $input_name
		cd ..

		## Problems
		##  - when called, scatter-learn -i scatter-learn.egsinp yeilds:
		##      > scatter-learn.mederr **
		##	> scatter-learn.ptracks **
		##	> scatter-learn.egsdat **
		##
	        ##	    ~~ These are actually produce by ./main.sh ~~
		##	> scatter-learn-placeholder.egslog
		##	> detector_results.csv
		##	> scatter_input.csv
		##
		##
		## ** These files are generated in scatter-learn directory.

		## !? How can I get these (** files) to be generated elsewhere ?
		##     > For example, in subdirectory func_$func_name_-_ncases_#ncases
		##



    done
    
done


