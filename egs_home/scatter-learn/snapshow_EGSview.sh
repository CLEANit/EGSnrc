#!/bin/sh

GetEGS_Xid (){
	xwininfo -root -tree | grep "egs_view (scatter" | sed 's/\([^\^]\)\s\".*/\1/' 
}

rand_num=$1
it_turn=$2

(nohup egs_view "scatter-learn_-_num_0-$rand_num.egsinp" scatter-learn_egsview_default.egsview) & echo $! > prod.pid
sleep 2
Xid=$(GetEGS_Xid)
echo $Xid
import -window $Xid "./optimization_plots/simulation/egs_figs/$it_turn-setup_view_$rand_num".png
kill $(cat prod.pid)

(nohup egs_view "scatter-learn_-_num_0-$rand_num.egsinp" scatter-learn_egsview_default.egsview "scatter-learn_-_num_0-$rand_num.ptracks") & echo $! > prod.pid
sleep 2
Xid=$(GetEGS_Xid)
import -window $Xid "./optimization_plots/simulation/egs_figs/$it_turn-simulate_view_$rand_num".png
kill $(cat prod.pid)
rm prod.pid

