#!/bin/bash

#this file runs SETS on the solar flare dataset as described in the paper
#feel free to experiment

dataset_name='BasicMotions'
st_contract='30'
max_sh_len='25'

python terce/mine_rules.py $dataset_name $st_contract $max_sh_len
python terce/class_rules.py $dataset_name $st_contract $max_sh_len
python terce/terce.py $dataset_name $st_contract $max_sh_len