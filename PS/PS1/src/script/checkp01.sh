#!/bin/bash
path="/Users/wooseokhan/Desktop/2023_SUMMER/cs229-Machine_Learning-2018/PS/PS1/src"
cd "$path"

# Loop over the file numbers
for problem_n in 01b 01e
do
    for i in {1..2}
    do
        echo "Checking problem $problem_n with file $i"
        if (diff output/p${problem_n}_pred_${i}.txt output_sol/p${problem_n}_pred_${i}sol.txt -c) then
            echo "Correct!"
        else
            echo "Wrong!"
        fi
        echo "Done checking problem $problem_n with file $i"
    done
done