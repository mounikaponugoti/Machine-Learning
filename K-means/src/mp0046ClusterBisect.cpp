/************************************************************************************************************
* Name: Mounika Ponugoti
* Course: CS 641 Data Mining Assignment #2
* Date: 2/28/2017
* Instructor: Dr. Ramazan Aygün
* File Name: mp0046ClusterBisect.cpp
* Description: This program does the clustering for the given input data by using Bisecting k-means algorithm.
*				It can cluster original data or normalized data. While clustering intended class attribute is
*				not considered. Two output files are created while executing the program if normalization is 
*				not specified. One output file holds the cluster centeres along with the cluster ids. Another
*				output file holds the input data along with cluster ID where it belongs to. If the clustering
*				is performed on normalized data then 4 output files are created. Two additional files remaps 
*				the cluster centers and input data back.
*
* To Compile: gcc mp0046ClusterBisect.cpp -o mp0046ClusterBisect -std=c++11 
* To run: mp0046ClusterBisect -i inputfile -k N -c classattribute -normalize
**************************************************************************************************************/
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#include <cstring>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <stdlib.h>

struct cluster{
	std::vector<std::vector<double> > ClusterMember;
	std::vector<double> centroid;
	double SSE;
} clusters;

struct eachAttr {
	std::vector<double> attrValues;
	std::vector<double> NormValues;
	double mean;
	double standardDeviation;
	std::string attrName;

	eachAttr(){
		mean = 0;
		standardDeviation = 0;
		attrName = " ";
	}
}temp_eachAttr;

//std::vector<int> assignClusterId;
std::vector<cluster> last_temp_cluster;
std::vector<cluster> originalCluster;
std::vector<eachAttr> inputAttributes;
bool Normalize = false;
int numOfClusters = 1;
int currentClusters = 0;

std::string classAttr = " ";
std::ifstream source;
std::ofstream Cluster_Center_Bisect_Destination;
std::ofstream Clustering_Bisect_Destination;
std::ofstream Cluster_Center_UnnormBisect_Destination;
std::ofstream Clustering_UnnormBisect_Destination;

std::string Cluster_Center_Bisect_OutFileName;
std::string Clustering_Bisect_OutFileName;
std::string Cluster_Center_UnnormBisect_OutFileName;
std::string Clustering_UnnormBisect_OutFileName;
std::string inputFileName;

/* Calculates the mean for all the attributes */
void CalculateMean(int numOfAttributes, int eachAttrLength)
{
   double sum;
   for (int i=0; i<numOfAttributes-1; i++) {
   sum = 0.0;
   for (int j=0; j<eachAttrLength; j++)
      sum += inputAttributes[i].attrValues[j];
      inputAttributes[i].mean = sum/eachAttrLength;
   }
}

/* Calculates the standard deviation for all the attributes */
void CalculateStandardDeviation( int numOfAttributes, int eachAttrLength)
{
   double sumOfSquares;
   double temp;
   for (int i=0; i<numOfAttributes-1; i++) {
	 sumOfSquares = 0.0;
	 temp = 0.0;
	 for (int j=0; j<eachAttrLength; j++) {
		temp = inputAttributes[i].attrValues[j]-inputAttributes[i].mean;
		sumOfSquares += temp*temp;
	 }
	 inputAttributes[i].standardDeviation = sqrt(sumOfSquares/eachAttrLength);
	}
}

/* Calculates the normalized values for all the attributes */
void CalculateNormalization(int numOfAttributes, int eachAttrLength)
{
	double temp;
	for (int i=0; i<numOfAttributes-1; i++) {
		temp = 0.0;
		for (int j=0; j<eachAttrLength; j++) {
			temp = (inputAttributes[i].attrValues[j]-inputAttributes[i].mean)/inputAttributes[i].standardDeviation;
			inputAttributes[i].NormValues.push_back(temp);
		}
	}
}

double calculateDistance(int pos, int clusterNum, int numOfAttributes)
{
	double sum = 0;
	if(Normalize){
		for(int i=0; i<numOfAttributes-1; i++)
		if(inputAttributes[i].attrName != classAttr)
		  sum += (inputAttributes[i].NormValues[pos]-originalCluster[clusterNum].centroid[i])*(inputAttributes[i].NormValues[pos]-originalCluster[clusterNum].centroid[i]);
	}
	else{
		for(int i=0; i<numOfAttributes-1; i++)
			if(inputAttributes[i].attrName != classAttr)
			  sum += (inputAttributes[i].attrValues[pos]-originalCluster[clusterNum].centroid[i])*(inputAttributes[i].attrValues[pos]-originalCluster[clusterNum].centroid[i]);
	}
	return sqrt(sum);
}

/* To debug */
void printClusterMembers()
{
	int	tempCounter=0;
	std::cout << "Members: " << '\n';

	for (auto Num=originalCluster.begin(); Num!=originalCluster.end(); Num++) {
		std::cout << "  Cluster: " << tempCounter;
		for	(auto mem=(*Num).ClusterMember.begin(); mem!=(*Num).ClusterMember.end(); mem++) {
			std::cout << " (";
			for (auto it=mem->begin(); it!=mem->end(); it++) {
				std::cout << " " << std::setw(5) << std::fixed << std::setprecision(2) << std::dec << *it;
			}
			std::cout << " )";
		}
		tempCounter++;
		std::cout << '\n';
	}
}

void printClusterCentroids()
{
	int	tempCounter=0;
	std::cout << "Centroids: " << '\n';
	for (auto Num=originalCluster.begin(); Num!=originalCluster.end(); Num++) {
		std::cout << "  Cluster: " << tempCounter << " (";
		for	(auto mem=(*Num).centroid.begin(); mem!=(*Num).centroid.end(); mem++) {
			std::cout << " " << std::setw(5) << std::fixed << std::setprecision(2) << std::dec << *mem;
		}
		std::cout << " )";
		tempCounter++;
		std::cout << '\n';
	}
}
// Writes centroids to a file
void writeCentroids(int numOfAttributes)
{
	for (int i=0; i<numOfClusters; i++){
		for (int tempCounter=0; tempCounter<numOfAttributes-1; tempCounter++){
		   Cluster_Center_Bisect_Destination << " " << std::setw(7) << std::fixed << std::setprecision(2) << std::dec << originalCluster[i].centroid[tempCounter];
		  if(Normalize){
			  double convert =0;
			  convert = (originalCluster[i].centroid[tempCounter]*inputAttributes[tempCounter].standardDeviation)+inputAttributes[tempCounter].mean;
			  Cluster_Center_UnnormBisect_Destination << " " << std::setw(7) << std::fixed << std::setprecision(2) << std::dec << convert;
		  }
		}
	   Cluster_Center_Bisect_Destination << " " << std::setw(3) << (i+1) << '\n';
	   Cluster_Center_UnnormBisect_Destination << " " << std::setw(3) << (i+1) << '\n';
  }
}

// Writes attributes to a file
void writeAttrToFile(int numOfAttributes, int eachAttrLength)
{
  for (int tempCounter=0; tempCounter<eachAttrLength; tempCounter++) {
	  for (int i=0; i<numOfAttributes; i++) {
		if(Normalize){
			if(inputAttributes[i].attrName == "cluster") { // cluster Id is an integer
				Clustering_Bisect_Destination << " " << std::setw(3) << std::dec << (int) inputAttributes[i].NormValues[tempCounter]+1;
				Clustering_UnnormBisect_Destination << " " << std::setw(3) << std::dec << (int) inputAttributes[i].NormValues[tempCounter]+1;
			}
			else {// Write the normalized values to output file 
				Clustering_Bisect_Destination << " " << std::setw(8) << std::fixed << std::setprecision(4) << std::dec << inputAttributes[i].NormValues[tempCounter];
				Clustering_UnnormBisect_Destination << " " << std::setw(8) << std::fixed << std::setprecision(4) << std::dec << inputAttributes[i].attrValues[tempCounter];
			}
		}
		else{
			if(inputAttributes[i].attrName == "cluster")
				Clustering_Bisect_Destination << " "  << std::setw(3) << std::dec << (int) inputAttributes[i].attrValues[tempCounter]+1;
			else // Write the normalized values to output file 
				Clustering_Bisect_Destination << " " << std::setw(8) << std::fixed << std::setprecision(4) << std::dec << inputAttributes[i].attrValues[tempCounter];
		}
	  }
	  Clustering_Bisect_Destination << '\n';
	  Clustering_UnnormBisect_Destination << '\n';
  }
}

// Calculates the centroid
void calculateCentroid(int numOfAttributes, int numOfClusters, int eachAttrLength, int clustNumToBisect)
{
	for (int Num=0; Num<currentClusters; Num++) {
		std::vector<double> sum(numOfAttributes-1, 0);
		if(Num==clustNumToBisect || Num==(originalCluster.size()-1)){
			for	(int mem=0; mem<originalCluster[Num].ClusterMember.size(); mem++) {
				for (int it=0; it<numOfAttributes-1; it++) {
					sum[it] += originalCluster[Num].ClusterMember.at(mem).at(it);
				}
			}
			for (int it=0; it<numOfAttributes-1; it++) 
				originalCluster[Num].centroid[it] = (double) sum[it]/originalCluster[Num].ClusterMember.size();
		}
	}
}

// Checks the attribute is member of which cluster
bool CheckInCluster(int pos, int clustNum, int numOfAttributes )
{
	bool globalFind = true;
	bool localFind = false;
	
	for (int mem=0; mem<originalCluster[clustNum].ClusterMember.size(); mem++){
		globalFind = true;
		for (int attr=0; attr<numOfAttributes-1; attr++) {
			if(inputAttributes[attr].attrName != classAttr){
				if(Normalize){
					if(originalCluster[clustNum].ClusterMember.at(mem).at(attr) != inputAttributes[attr].NormValues[pos]){
						localFind = false;
						break;
					} else 
						localFind = true;
				} else{
					 if(originalCluster[clustNum].ClusterMember.at(mem).at(attr) != inputAttributes[attr].attrValues[pos]){
						localFind = false;
						break;
					 }else 
						localFind = true;
				}
				globalFind &= localFind;
			}
		}
		globalFind &= localFind;
		if(globalFind)
			return true;
	}
	return false;
}

// Assigns cluster ids to the input data
void assignClusterID(int numOfAttributes, int eachAttrLength, int clustNumToBisect)
{
	// if the input data point belongs to parent cluster which is getting bisect, assign that data to the one which is near.
	for (int pos=0; pos<eachAttrLength; pos++){
		if(((!Normalize)&&((inputAttributes[numOfAttributes-1].attrValues[pos] ==  clustNumToBisect) || (inputAttributes[numOfAttributes-1].attrValues[pos] ==  currentClusters-1)))
			|| ((Normalize) && ((inputAttributes[numOfAttributes-1].NormValues[pos] ==  clustNumToBisect) || (inputAttributes[numOfAttributes-1].NormValues[pos] ==  currentClusters-1)))){
			for (int clustNum=0; clustNum<currentClusters; clustNum++){
				if((clustNum==clustNumToBisect)|| (clustNum== (currentClusters-1))){
					if(CheckInCluster(pos, clustNum, numOfAttributes)){
						if(Normalize)
							inputAttributes[numOfAttributes-1].NormValues[pos] = clustNum;
						else
							inputAttributes[numOfAttributes-1].attrValues[pos] = clustNum;
						break;
					}
				}
			}
		}
	}
}

// Initializes the centroids
void initializeCentroids(int numOfAttributes, int eachAttrLength, std::vector<double>& distance, int clustNumToBisect)
{
	int index;
	bool localFind = false;
	bool globalFind = true;
	bool blongsToThisCluster = false;

	originalCluster.push_back(clusters);
	last_temp_cluster.push_back(clusters);
	currentClusters++;
	/* create new cluster */
	for (int attr=0; attr<numOfAttributes-1; attr++){
		originalCluster[originalCluster.size()-1].centroid.push_back(0);
		last_temp_cluster[originalCluster.size()-1].centroid.push_back(0);
	}
	// generate the index to get the centroid. If this index not belongs to this cluster generate again
	do{
		index = rand()%eachAttrLength;
		if(((!Normalize)&&(inputAttributes[numOfAttributes-1].attrValues[index] ==  clustNumToBisect)) || 
					((Normalize)&&(inputAttributes[numOfAttributes-1].NormValues[index] ==  clustNumToBisect))){
			blongsToThisCluster = true;
			if(Normalize)
				for (int attr=0; attr<numOfAttributes-1; attr++){
					originalCluster[clustNumToBisect].centroid[attr] = inputAttributes[attr].NormValues[index];
				}
			else
				for (int attr=0; attr<numOfAttributes-1; attr++){
					originalCluster[clustNumToBisect].centroid[attr] = inputAttributes[attr].attrValues[index];
				}
		}
	}while(!blongsToThisCluster);

	// choose random centroid from new cluster 
	blongsToThisCluster = false;
	do{
		index = rand()%eachAttrLength;
		if(((!Normalize)&&(inputAttributes[numOfAttributes-1].attrValues[index] ==  clustNumToBisect)) || 
				((Normalize)&&(inputAttributes[numOfAttributes-1].NormValues[index] ==  clustNumToBisect))){
			blongsToThisCluster = true;
			for (int attr=0; attr<numOfAttributes-1; attr++){
				if(inputAttributes[attr].attrName != classAttr){
					if(Normalize){
						if(inputAttributes[attr].NormValues[index] != originalCluster[clustNumToBisect].centroid[attr]) {
							localFind = false;
							break;
						} else 
							localFind = true;
					} else{
						if(inputAttributes[attr].attrValues[index] != originalCluster[clustNumToBisect].centroid[attr]) {
							localFind = false;
							break;
						} else 
							localFind = true;
					}
					globalFind &= localFind;
				}
			}
			globalFind &= localFind;
		}
	}while((!blongsToThisCluster)||globalFind);
	
	if(Normalize){
		for (int attr=0; attr<numOfAttributes-1; attr++){
			originalCluster[originalCluster.size()-1].centroid[attr] = inputAttributes[attr].NormValues[index];
			last_temp_cluster[originalCluster.size()-1].centroid[attr] = 0;
		}
	} 
	else {
		for (int attr=0; attr<numOfAttributes-1; attr++){
			originalCluster[originalCluster.size()-1].centroid[attr] = inputAttributes[attr].attrValues[index];
			last_temp_cluster[originalCluster.size()-1].centroid[attr] = 0;
		}
	}
}

// does the k-means clustering for the selected cluster
void doKmeansClustering(int numOfAttributes, int eachAttrLength, int clustNumToBisect)
{
	std::vector <double> tempClusterMember(numOfAttributes-1);
	cluster tempCluster;
	std::vector <double> distance(2);
	bool goOn = true;

	// save the data of the cluster which we are going to spilt
	tempCluster = originalCluster[clustNumToBisect];
	initializeCentroids(numOfAttributes, eachAttrLength, distance, clustNumToBisect);
		
	do{
		goOn = true;
		/* To debug:
		printClusterCentroids(); */

		last_temp_cluster = originalCluster;
				
		originalCluster[clustNumToBisect].ClusterMember.erase(originalCluster[clustNumToBisect].ClusterMember.begin(),
				originalCluster[clustNumToBisect].ClusterMember.end());
		originalCluster[originalCluster.size()-1].ClusterMember.erase(originalCluster[originalCluster.size()-1].ClusterMember.begin(),
				originalCluster[originalCluster.size()-1].ClusterMember.end());
		
		for(int pos=0; pos < eachAttrLength; pos++){
		
			if(((inputAttributes[numOfAttributes-1].attrValues[pos] == clustNumToBisect)&&(!Normalize))
					|| ((inputAttributes[numOfAttributes-1].NormValues[pos] == clustNumToBisect)&&Normalize))
			{
				distance[0] = calculateDistance(pos, clustNumToBisect, numOfAttributes);
				distance[1] = calculateDistance(pos, originalCluster.size()-1, numOfAttributes);
			
				int lowerIndex = clustNumToBisect;
				
				// check which cluster is near
				if(distance[0] > distance[1])
					lowerIndex = originalCluster.size()-1;
				
				if(Normalize){
					for (int k=0; k<numOfAttributes-1; k++)
						tempClusterMember[k] = inputAttributes[k].NormValues[pos];
				}
				else {
					for (int k=0; k<numOfAttributes-1; k++)
						tempClusterMember[k] = inputAttributes[k].attrValues[pos];
				}
				// copy the input point to the appropriate cluster
				originalCluster[lowerIndex].ClusterMember.push_back(tempClusterMember);
			}
		}
		/* To debug:
		printClusterMembers(); */
		calculateCentroid(numOfAttributes, numOfClusters, eachAttrLength, clustNumToBisect);
		// check are the centroids of last calculation and this matches? i.e. terminating ?
		for(int k=0; k<currentClusters; k++){
			if((k==clustNumToBisect)|| k==(originalCluster.size()-1))
				for (int attr=0; attr<numOfAttributes-1; attr++){
					if(inputAttributes[attr].attrName != classAttr)
						if(fabs(last_temp_cluster[k].centroid[attr] - originalCluster[k].centroid[attr]) > 0.01){
							goOn = false;
							goto end;
						}
				}
		}
		end:;
	} while(!goOn);
}

// calculates the sum of squares of errors
void calculateSSE(int numOfAttributes)
{
	double error = 0;
	double temp = 0;
	for(int Num=0; Num<currentClusters; Num++){
		error = 0;
		for(int mem=0; mem<originalCluster[Num].ClusterMember.size(); mem++){
			for (int attr=0; attr<numOfAttributes-1; attr++){
				if(inputAttributes[attr].attrName != classAttr){
					temp = (originalCluster[Num].centroid[attr]-originalCluster[Num].ClusterMember.at(mem).at(attr));
					error += temp*temp;
				}
			}
		}
		originalCluster[Num].SSE = error;
	}
}

// Bisecting K-means main logic
void doBisectingClustering(int numOfAttributes, int eachAttrLength)
{
	srand (time(NULL));
	std::vector <double> tempClusterMember(numOfAttributes-1);

	originalCluster.push_back(clusters);
	last_temp_cluster.push_back(clusters);
	// initial single cluster
	for (int attr=0; attr<numOfAttributes-1; attr++){
		originalCluster[originalCluster.size()-1].centroid.push_back(0);
		last_temp_cluster[originalCluster.size()-1].centroid.push_back(0);
	}
	currentClusters++;
	for(int pos=0; pos < eachAttrLength; pos++){
		for (int k=0; k<numOfAttributes-1; k++)
			if(Normalize)
				tempClusterMember[k] = inputAttributes[k].attrValues[pos];
			else
				tempClusterMember[k] = inputAttributes[k].attrValues[pos];
		originalCluster[0].ClusterMember.push_back(tempClusterMember);
	}

	for(int pos=0; pos < eachAttrLength; pos++){
		inputAttributes[numOfAttributes-1].attrValues.push_back(0);
		inputAttributes[numOfAttributes-1].NormValues.push_back(0);
	}
	
	//start k-means on big single cluster
	doKmeansClustering(numOfAttributes, eachAttrLength, 0);
	//asiign cluster ids appropriately for the input data
	assignClusterID(numOfAttributes, eachAttrLength, 0);
	
	// repeat untill termination condition reaches
	while(currentClusters < numOfClusters){
		calculateSSE(numOfAttributes);
		int higherIndex = 0;
		for (int k=0; k<currentClusters; k++){
			//std::cout << "SSE" << k <<" : " << originalCluster[k].SSE << '\n';
			if(originalCluster[higherIndex].SSE < originalCluster[k].SSE)
				higherIndex = k;
		}
		doKmeansClustering(numOfAttributes, eachAttrLength, higherIndex);
		assignClusterID(numOfAttributes, eachAttrLength, higherIndex);
		
	}
}


/* Opens the input and output files */
void openFile()
{ 
	source.open(inputFileName.c_str());
	if(!source)
	  std::cerr << "Error: Unable to open the input file" << '\n';
	if(!Normalize){
		Cluster_Center_Bisect_OutFileName = "mp0046ClusterCenterBisect"+std::to_string(numOfClusters)+inputFileName;
		Clustering_Bisect_OutFileName = "mp0046ClusteringBisect"+std::to_string(numOfClusters)+inputFileName;

		Cluster_Center_Bisect_Destination.open(Cluster_Center_Bisect_OutFileName.c_str()) ;
		Clustering_Bisect_Destination.open(Clustering_Bisect_OutFileName.c_str());
		if(!Cluster_Center_Bisect_Destination || !Clustering_Bisect_Destination )
			std::cerr << "Error: Unable to open one of the output file" << '\n';

	} else {
		Cluster_Center_Bisect_OutFileName = "mp0046ClusterCenterNormalizedBisect"+std::to_string(numOfClusters)+inputFileName;
		Clustering_Bisect_OutFileName = "mp0046ClusteringNormalizedBisect"+std::to_string(numOfClusters)+inputFileName;
		Cluster_Center_UnnormBisect_OutFileName = "mp0046ClusterCenterUnnormalizedBisect"+std::to_string(numOfClusters)+inputFileName;
		Clustering_UnnormBisect_OutFileName  = "mp0046ClusteringUnnormalizedBisect"+std::to_string(numOfClusters)+inputFileName;

		Cluster_Center_Bisect_Destination.open(Cluster_Center_Bisect_OutFileName.c_str()) ;
		Clustering_Bisect_Destination.open(Clustering_Bisect_OutFileName.c_str());
		Cluster_Center_UnnormBisect_Destination.open(Cluster_Center_UnnormBisect_OutFileName.c_str());
		Clustering_UnnormBisect_Destination.open(Clustering_UnnormBisect_OutFileName.c_str());

		if(!Cluster_Center_Bisect_Destination || !Clustering_Bisect_Destination 
				|| !Cluster_Center_UnnormBisect_Destination || !Clustering_UnnormBisect_Destination)
			std::cerr << "Error: Unable to open one of the output file" << '\n';
	}	
}	

int readCommandLine(int argc, char **argv)
{
	int argi;
	
	if (argc <= 1 ) {
		std::cout << "mp0046ClusterBasic -i inputfile -K N -c classattribute -normalize" << '\n';
		return 1; 
	}
	for (argi = 1; argi  < argc; argi++)
	{
		if(!strcmp(argv[argi], "-i"))
			inputFileName = argv[argi+1];
		if(!strcmp(argv[argi], "-K"))
			numOfClusters = atoi(argv[argi+1]);
		if(!strcmp(argv[argi], "-c"))
			classAttr = atoi(argv[argi+1]);
		if(!strcmp(argv[argi], "-normalize"))
			Normalize = true;
	}
	return 0;
}

/* Closes the input and output files */
void closeFile()
{
  source.close();
  Cluster_Center_Bisect_Destination.close();
  Clustering_Bisect_Destination.close(); 
  if (Normalize){
	  Cluster_Center_UnnormBisect_Destination.close();
	  Clustering_UnnormBisect_Destination.close();
  }
}

/* Main Program */
int main(int argc, char *argv[])
{
  std::string currentLine;
  std::string isAttribute = "attribute";
  std::string holdAttrName = " ";
 
  int eachAttrLength = 0;
  int numOfAttributes = 0;
  double readValue = 0.0;
  int tempCounter = 0;

  readCommandLine(argc, argv);
  openFile();
 
  /* Handels the lines which are in the beginning of the input file */
  getline(source, currentLine);
  Cluster_Center_Bisect_Destination << currentLine << '\n' << '\n';
  Clustering_Bisect_Destination << currentLine << '\n' << '\n';
  if(Normalize){
	 Cluster_Center_UnnormBisect_Destination << currentLine << '\n' << '\n';
	 Clustering_UnnormBisect_Destination << currentLine << '\n' << '\n'; 
  }
  while(source.good()){
	  if (currentLine.length() != 0){        // discards the empty lines
		  std::istringstream ss(currentLine);
		  if (currentLine.at(0) == '@') {    
			ss.ignore();
			ss >> holdAttrName;
			if(holdAttrName == isAttribute) {  // get the attribute names
				ss >>  holdAttrName;
				inputAttributes.push_back (temp_eachAttr);
				inputAttributes[numOfAttributes].attrName = holdAttrName;
				numOfAttributes ++;
			}
			// if the data is going to start end this loop
			if(holdAttrName == "data") {
				inputAttributes.push_back (temp_eachAttr);
				inputAttributes[numOfAttributes].attrName = "cluster";
				numOfAttributes ++;
				for(int i=0; i<numOfAttributes; i++) {
					Cluster_Center_Bisect_Destination << "@attribute " << inputAttributes[i].attrName << " real" << '\n';
					Clustering_Bisect_Destination << "@attribute " << inputAttributes[i].attrName << " real" << '\n';
					if(Normalize){
					   Cluster_Center_UnnormBisect_Destination << "@attribute " << inputAttributes[i].attrName << " real" << '\n';
					   Clustering_UnnormBisect_Destination << "@attribute " << inputAttributes[i].attrName << " real" << '\n';
					 }
				}

				Cluster_Center_Bisect_Destination  << '\n' << currentLine << '\n';
				Clustering_Bisect_Destination  << '\n' << currentLine << '\n';
				if(Normalize){
					Cluster_Center_UnnormBisect_Destination  << '\n' << currentLine << '\n';
					Clustering_UnnormBisect_Destination  << '\n' << currentLine << '\n'; 
				}
				break;
			}
		  }
	  }
	  getline(source, currentLine);
  }

  /* Reads the data from input file and stores in a vector */
  getline(source, currentLine);
  while(source.good()){
	if (currentLine.length() != 0) {	 // discards the empty lines
		tempCounter = 0;
		std::istringstream ss(currentLine);
		// Extracts the numbers from the current line
		while (ss >> readValue) {
			// save the numbers to a vector
		    inputAttributes[tempCounter].attrValues.push_back(readValue);   
			// if the delimiter is ',' ignore that character 
			if(ss.peek() == ',' )		
				ss.ignore();
			tempCounter++;
		}
	}
	getline(source, currentLine);
  }

  // Get the size of each attribute 
  eachAttrLength = inputAttributes[0].attrValues.size();
  // Calculate the mean, standard deviation, and normalized values 
  CalculateMean (numOfAttributes, eachAttrLength);
  CalculateStandardDeviation (numOfAttributes, eachAttrLength);
  CalculateNormalization (numOfAttributes, eachAttrLength);
  doBisectingClustering(numOfAttributes, eachAttrLength);

  // Writecluster centers to the appropriate output file
  writeCentroids(numOfAttributes);
  writeAttrToFile(numOfAttributes, eachAttrLength);
  /* Close all the input and output files */
  closeFile();
  return 0;
}
