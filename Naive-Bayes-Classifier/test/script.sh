#!/bin/bash

scriptPath=$(dirname "$(realpath $0)")
exePath=${scriptPath/"test"/"bin"}
declare -a Targets=("naiveBayesWithKfoldCrossValidation")

echo "Output files are in $scriptPath"
echo ""

for Target in ${Targets[@]}
do 
  echo "Running "$Target"...."
  
  sleep 1
  echo "$exePath/$Target -i bcwdisc.arff -k 5 "
  $exePath/$Target -i bcwdisc.arff -k 5
  
  sleep 1
  echo "$exePath/$Target -i bcwdisc.arff -k 3 -c class"
  $exePath/$Target -i bcwdisc.arff -k 3

  echo "Done with "$Target"...."
  echo ""
done
