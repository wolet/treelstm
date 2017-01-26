#!/usr/bin/env/bash
echo "================================================="
echo "Distributing buckets.."

root=$1
suffix=$2
for b in 0 1 2 3 4 5 6 7 8 9
do
    mkdir -p $root/buckets/$b
    mv $root/idx.txt.$b $root/buckets/$b/idx.txt
    mv $root/labels.txt.$b $root/buckets/$b/labels.txt
    mv $root/parents.txt.$b $root/buckets/$b/parents.txt
    mv $root/sents.txt.$b $root/buckets/$b/sents.txt
done
echo "================================================="
