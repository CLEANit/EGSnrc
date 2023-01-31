#!/bin/sh

rnd () {
	echo "$(hexdump -d -n 7 /dev/urandom | head -n 1 | sed 's/.*0 //' | sed 's/ //g')"
	# $(hexdump -d -n 7 /dev/urandom | head -n 1 | sed 's/.*0 //' | sed 's/ //g')
}


get_rnd () {
	rn=$(rnd)
	#while [ "$rn" == '' ] 
	while (test -z "$rn")
	do
		#echo "hold on"
		rn=$(rnd)
	done
	rn=$(echo $rn | sed 's/^0*//')
	rn=$(python -c "print($rn%32000)")
	echo $rn
}

rn=""
for x in $(seq $1)
do
	rn="$rn $(get_rnd)"
done
echo $rn

