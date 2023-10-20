#!/bin/bash
path="/Users/wooseokhan/Desktop/2023_SUMMER/cs229-Machine_Learning-2018/PS/PS1/src"
my_predict="output/p03d_pred.txt"
sol="output_sol/p03dsol.txt"
cd "$path"

#
awk '{printf("%.0f\n", $1)}' $my_predict > ${my_predict}_integer.txt
awk '{printf("%.0f\n", $1)}' $sol > ${sol}_integer.txt
#
#check
echo "Checking problem 03d"
if (diff ${my_predict}_integer.txt ${sol}_integer.txt -c) then
    echo "Correct!"
else
    echo "Wrong!"
fi
echo "Done checking problem03d"
