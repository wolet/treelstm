#!/usr/bin/env/bash

echo "================================================="
echo "Shuffling sorted data.."

root=$1
for f in 0 1 2 3 4 5 6 7 8 9
do
    mkdir -p $root/random/$f
    paste -d "~" $root/sorted/idx.txt $root/sorted/labels.txt $root/sorted/parents.txt $root/sorted/sents.txt | shuf | awk -v FS="~" -v FILE=$f -v ROOT=$root '{
               print $1 > ""ROOT"/random/"FILE"/idx.txt";
               print $2 > ""ROOT"/random/"FILE"/labels.txt";
               print $3 > ""ROOT"/random/"FILE"/parents.txt";
               print $4 > ""ROOT"/random/"FILE"/sents.txt"; }'
done


for file in idx labels parents sents
do
    wc $root/sorted/$file.txt
    find $root/random -name \*$file.txt | sort | xargs -I '{}' wc '{}'
    echo "_________-"
done
echo "================================================="
