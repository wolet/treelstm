#!/usr/bin/env/bash

root=$1
mkdir -p $root/sorted
echo "================================================="
echo "Creating sorted data.."
for f in idx labels parents sents 
do
    cat `find $root/buckets -name \*$f.txt | sort` > $root/sorted/$f.txt
    wc `find $root/buckets -name \*$f.txt | sort` | tail -1
    wc $root/sorted/$f.txt
done
echo "================================================="
