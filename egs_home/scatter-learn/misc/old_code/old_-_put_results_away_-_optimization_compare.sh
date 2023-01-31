#!/bin/sh
dirname=$(echo run_-_$(date -I)_-_$(ls -1 | grep $(date -I) | wc -l))
mkdir $dirname
for x in $(ls -1 | grep "detector_results" | sed 's/scatter-learn_-_//' | sed 's/_-_detector.*//')
do
	mv $(echo *$x*) $dirname
	for y in $(ls -1 | grep $x)
	do
		#echo $y
		mv $y $dirname
	done
	#cd $dirname
	#mkdir $x
	#cd ..
	#mv $x $(echo $dirname/$x)
		
done
cd $dirname
mkdir og_results
mv * og_results
cd ..
