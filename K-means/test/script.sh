#!/bin/bash

scriptPath=$(dirname "$(realpath $0)")
exePath=${scriptPath/"test"/"bin"}
declare -a Targets=("KmeansClustering")

echo "Output files are in $scriptPath"
echo ""

for Target in ${Targets[@]}
do 
  echo "Running "$Target"...."
  
  sleep 1
  echo "$exePath/$Target -i kmtest.arff -k 5 "
  $exePath/$Target -i kmtest.arff -k 5
  
  sleep 1
  echo "$exePath/$Target -i mesocyclone.arff -k 3 -n true"
  $exePath/$Target -i mesocyclone.arff -k 3 -n true
  
  sleep 1
  echo "$exePath/$Target -i mesocyclone.arff -k 3 -c class"
  $exePath/$Target -i mesocyclone.arff -k 3
  sleep 1
  echo "$exePath/$Target -i mesocyclone.arff -k 5 -c class -n true"
  $exePath/$Target -i mesocyclone.arff -o normalize_Kmeans_meso_5.arff -k 5 -c class -n true
  
  echo "Done with "$Target"...."
  echo ""
done
