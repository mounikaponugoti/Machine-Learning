#!/bin/bash

workingPath=$PWD
Target="batchGradientDescent"
exePath=${workingPath/"Test"/"bin"/}

#echo $exePath/$Target

$exePath/$Target -i input_1.txt -v 1 
sleep 2
$exePath/$Target --input input_1.txt -v 1 -l 0.0001
sleep 2
$exePath/$Target --input input_2.txt --variables 2 --iterations 500000 --epsilon 0.00001
sleep 2
$exePath/$Target --input input_2.txt --variables 2 --learningrate 0.0001