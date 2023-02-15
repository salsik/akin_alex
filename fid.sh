#!/bin/bash
for i in {0..10..2}
do
  echo "run test $i times"
done

fol='/data1/data_alex/rico/*';

fol="/data1/data_alex/rico/akin_experiments/samples_generator/*"

for dir in $fol
do 
    echo "$dir" ;
    python -m pytorch_fid "$dir" "data/Akin_SAGAN_500/all_inone" --device cuda
done



#fid calculation


#python -m pytorch_fid '/data1/data_alex/rico/akin experiments/samples_generator/pretrained checkpointscp-007000' "data/Akin_SAGAN_500/all_inone" --device cuda

#for dir in /data1/data_alex/rico/akin experiments/samples_generator/ ; do echo "$dir" done
