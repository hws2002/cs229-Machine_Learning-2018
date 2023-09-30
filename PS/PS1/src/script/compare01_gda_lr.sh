#!/bin/bash
$path=/Users/wooseokhan/Desktop/2023_SUMMER/cs229-Machine_Learning-2018/PS/PS1/src
cd "$path"


for dataset in 1 2
do 
    diff output/p01b_pred_${dataset}.txt output/p01e_pred_${dataset}.txt -c
done
