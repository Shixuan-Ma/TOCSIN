#!/usr/bin/env bash

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_API-based_model
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

m1='gpt-j-6B'
models='gpt-neo-2.7B'

datasets="xsum writing pubmed"
source_models="gpt-4 gpt-3.5-turbo gemini"
base_models='likelihood lrr logrank Fast standalone'

#evaluate TOCSIN in the black-box setting
for M in $source_models; do
  for D in $datasets; do
    for M2 in $models; do
      for M1 in $m1; do
        for BM in base_models;do
        echo `date`, Evaluating on ${D}_${M}.${M1}_${M2} ...
        python ./TOCSIN.py --reference_model_name $M1 --scoring_model_name $M2 --basemodel $BM \
                            --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
        done
      done
    done
  done
done
