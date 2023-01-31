#!/bin/sh

while [ "$(squeue | wc -l)" -gt "1" ]
do
	echo "patiently waiting"
	sleep 5
done
echo "Done EGSnrc"
