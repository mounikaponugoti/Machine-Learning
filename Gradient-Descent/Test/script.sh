#!/bin/bash

scriptPath=$(dirname "$(realpath $0)")
exePath=${scriptPath/"Test"/"bin"}
declare -a Targets=("batchGradientDescent" "stochasticGradientDescent")

echo "Output files are in $scriptPath"
echo ""

for Target in ${Targets[@]}
do 
  echo "Running "$Target"...."
  
  sleep 1
  echo "$exePath/$Target -i input_1.txt -v 1 "
  $exePath/$Target -i input_1.txt -v 1 
  
  sleep 1
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

  echo "Done with "$Target"...."
  echo ""
done