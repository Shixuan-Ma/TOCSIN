#!/usr/bin/env bash

# setup the environment
#echo Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_Open_source_model
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="xsum squad writing"
source_models="gpt2-xl gpt-j-6B gpt-neo-2.7B opt-2.7b gpt-neox-20b"
base_models='Fast likelihood lrr logrank standalone'

#evaluate TOCSIN in the white-box setting
echo Evaluate models in the white-box setting:

echo `date`,evaluate TOCSIN on four basemodels and itself
for D in $datasets; do
  for M in $source_models; do
    for BM in $base_models; do
    echo `date`, Evaluating on ${D}_${M} ............
    python ./TOCSIN.py --reference_model_name $M --scoring_model_name $M --dataset $D --basemodel $BM \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
    done
  done
done

#evaluate TOCSIN in the black-box setting
echo `date`, Evaluate models in the black-box setting:
scoring_models="gpt-neo-2.7B"

for D in $datasets; do
  for M in $source_models; do
    M1=gpt-j-6B  # sampling model
   for M2 in $scoring_models; do
     for BM in $base_models; do
      echo `date`, Evaluating on ${D}_${M}.${M1}_${M2} ...
      python ./TOCSIN.py --reference_model_name ${M1} --scoring_model_name ${M2} --dataset $D --basemodel $BM \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
      done
    done
  done
done
