#!/bin/bash

scriptPath=$(dirname "$(realpath $0)")
exePath=${scriptPath/"test"/"bin"}
declare -a Targets=("decisionTreeClassifier")

echo "Output files are in $scriptPath"
echo ""

for Target in ${Targets[@]}
do 
  echo "Running "$Target"...."
  
  sleep 1
  echo "$exePath/$Target -i bcwdisc.arff"
  $exePath/$Target -i bcwdisc.arff
  
  # Train with only 60% of the original data
  sleep 1
  echo "$exePath/$Target -i bcwdisc.arff -c class -T 60"
  $exePath/$Target -i bcwdisc.arff -c class -T 60
  
  echo "Done with "$Target"...."
  echo ""
done
