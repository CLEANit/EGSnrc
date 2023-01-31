#!/bin/sh

most_recent_run=$(ls -rt1 | grep "run_-_*" | tail -n 1)
rn_bk=$(ls -tr1 $most_recent_run/og_results/mederr_dir | tail -n 1 | sed "s/.*-//" | sed "s/\..*//")
rn=$1
rn=${rn:=$rn_bk}
if (test -n "$rn")
then
	cp $(find ./$most_recent_run/og_results/include_dir/ | grep "$rn") ./
	input_file=$(find $most_recent_run/og_results/egsinp_dir/  | grep $rn)
	ptracks_file=$(find $most_recent_run/og_results/ptracks_dir/  | grep $rn)
	view_file=$(echo $input_file | sed 's/\.egsinp/.egsview/' | awk -F "\/" '{print $4}')
	cp ./scatter-learn_egsview_default.egsview $view_file
	egs_view $input_file $ptracks_file $view_file

	# Clean up
	rm -i $view_file
	rm -i scatter-learn_-_num_0-$rn*
else
	echo "Empty Random Number"
	#./sweep_noclean.sh $rn
fi

