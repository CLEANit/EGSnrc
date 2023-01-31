#!/bin/sh

rand_num=10

inner_func () {
	echo "1: $rand_num"
	rand_num="lol"
	echo "2; $rand_num"
}
inner_func
echo "3: $rand_num"
