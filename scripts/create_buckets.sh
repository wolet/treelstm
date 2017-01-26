#!/usr/bin/env/bash

set -e
dict_path=$1
label_path=$2
output_path=$3
#python preprocess_cl.py $dict_path $label_path $output_path 10000
python preprocess_cl.py $dict_path $label_path $output_path 700 true

bash distribute.sh $output_path
bash sort.sh $output_path
bash shuffle.sh $output_path

#threshold=10000
#suffix=""
T=(350 175 87 43 21 11)
ii=0
suffix="-root"
for n in 4000 2000 1000 500 250 125
do
    mkdir -p $n$suffix/samples
    #threshold=`expr $threshold / 2`

    threshold=${T[$ii]}
    echo "THRESHOLD: " $threshold, $n
    ii=`expr $ii + 1`
    paste -d "~" $label_path/labels.txt $label_path/parents.txt $label_path/sents.txt | shuf | head -$n | awk -v FS="~" -v FILE=$f -v ROOT=$n$suffix  '{
                print $1 > ""ROOT"/samples/labels.txt";
                print $2 > ""ROOT"/samples/parents.txt";
                print $3 > ""ROOT"/samples/sents.txt"; }'
    python preprocess_cl.py $dict_path $n$suffix/samples $n$suffix $threshold true
    bash distribute.sh $n$suffix
    bash sort.sh $n$suffix
    bash shuffle.sh $n$suffix
done
