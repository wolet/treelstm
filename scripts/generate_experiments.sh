#!/usr/bin/env/bash

set -e
threshold=$1
suffix=$2
for dim in 256 128 64 32 16 8 4 2
do
    for root in 8000$suffix 4000$suffix 2000$suffix 1000$suffix 500$suffix 250$suffix 125$suffix
    do
#	threshold=`expr $threshold / 2`
	for seed in 0 1 2 3 4 5 6 7 8 9
	do
	    echo "th sentiment/train_vanilla_model.lua -d $dim -r -s exp/$dim/$root/vanilla/ -t data/sst/$root/random/$seed/ -k -f $threshold"
	done
	echo "th sentiment/train_vanilla_model.lua -d $dim -s exp/$dim/$root/sorted/ -t data/sst/$root/sorted/ -k -f $threshold"
	echo "th sentiment/train_cl_model.lua -d $dim -r onepass  -s exp/$dim/$root/onepass/  -t data/sst/$root/buckets/ -k -f $threshold"
	echo "th sentiment/train_cl_model.lua -d $dim -r babystep -s exp/$dim/$root/babystep/ -t data/sst/$root/buckets/ -k -f $threshold"
    done
done
