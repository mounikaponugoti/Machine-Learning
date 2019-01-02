#!/bin/bash

scriptPath=$(dirname "$(realpath $0)")
Target="batchGradientDescent"
exePath=${scriptPath/"Test"/"bin"}

echo ""
echo "Running batch gradient descent...."
echo "Output files are in $scriptPath"
echo ""

sleep 1
echo "$exePath/$Target -i input_1.txt -v 1 "
$exePath/$Target -i input_1.txt -v 1 

slpeep 1
echo "$exePath/$Target --input input_1.txt -v 1 -l 0.0001"
$exePath/$Target --input input_1.txt -v 1 -l 0.0001

sleep 1
echo "$exePath/$Target --input input_2.txt --variables 2 --iterations 500000 --epsilon 0.00001"
$exePath/$Target --input input_2.txt --variables 2 --iterations 500000 --epsilon 0.00001

sleep 1
echo "$exePath/$Target --input input_2.txt --variables 2 --learningrate 0.0001"
$exePath/$Target --input input_2.txt --variables 2 --learningrate 0.0001

sleep 1
echo "$exePath/$Target --input input_2.txt --variables 2 --learningrate 0.0005 --iterations 5000000 --normalize true --whichnorm mean"
$exePath/$Target --input input_2.txt --variables 2 --learningrate 0.0005 --iterations 5000000 --normalize true --whichnorm mean

sleep 1
echo "$exePath/$Target --input input_2.txt --variables 2 --learningrate 0.0005 --iterations 5000000 --normalize true --whichnorm minmax"
$exePath/$Target --input input_2.txt --variables 2 --learningrate 0.0005 --iterations 5000000 --normalize true --whichnorm minmax
echo ""
echo "Done with running batch gradient descent...."
