/**********************************************************************************
* This program clusters the input data in arff format by using k-means algorithm.
* It can cluster original data or normalized data. Two output files are created
* while executing the program if normalization is not specified. One output file
* holds the cluster centeres along with the cluster ids. Another output file holds
* the input data along with cluster ID where it belongs to. If the clustering is
* performed on normalized data then 4 output files are created. Two additional
* files remaps the cluster centers and input data back.
***********************************************************************************/
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
} clusters;

struct eachAttribute {
    std::vector<double> attrValues;
    std::vector<double> NormValues;
    double mean;
    double standardDeviation;
    std::string attrName;

    eachAttribute(){
        mean = 0;
        standardDeviation = 0;
        attrName = " ";
    }
}temp_eachAttr;

std::vector<cluster> last_temp_cluster;
std::vector<cluster> originalCluster;
std::vector<eachAttribute> inputAttributes;
bool Normalize = false;
bool userOutputFileName = false;
int numOfClusters = 1;

std::string classAttr = " ";
std::ifstream source;
std::ofstream Cluster_Center_Basic_Destination;
std::ofstream Clustering_Basic_Destination;
std::ofstream Cluster_Center_UnnormBasic_Destination;
std::ofstream Clustering_UnnormBasic_Destination;

std::string Cluster_Center_Basic_OutFileName;
std::string Clustering_Basic_OutFileName;
std::string Cluster_Center_UnnormBasic_OutFileName;
std::string Clustering_UnnormBasic_OutFileName;
std::string inputFileName;
std::string outputFileName;

std::string help_msg = "-h or --help\n"
"  To print this message\n"
"-i [inputFileName] or --input [inputFileName]\n"
"  To specify intput file name\n"
"-o [outputFileName] or --output [outputFileName]\n"
"  To specify output file name, default - batchGradientDecent_Output_[time].txt\n"
"-n [true/false] or --normalize [true/false] \n"
"-k [numberOfClusters] or --clusters [numberOfClusters]\n"
"  To specify number of number of required clusters, default - 1\n"
"-c [classAttribute] or --classattribute [classAttribute]\n"
"  This is optional. If it is provided, classAttribute is not considered while clustering\n";

/* Read the command line */
void readCommandLine(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            inputFileName = argv[i + 1];
            i++;
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            outputFileName = argv[i + 1];
            userOutputFileName = true;
            i++;
        }
        else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--normalize") == 0) {
            Normalize = (strcmp(argv[i + 1], "true") == 0 || strcmp(argv[i + 1], "True") == 0) ? true : false;
            i++;
        }
        else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--classattribute") == 0) {
            classAttr = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--clusters") == 0) {
            numOfClusters = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << help_msg << std::endl;
            exit(1);
        }
        else {
            std::cerr << "Error: Unknown parameter " << argv[i] << std::endl;
            std::cerr << help_msg << std::endl;
            exit(1);
        }
    }
}

/* Calculates the mean for all the attributes */
void CalculateMean(int numOfAttributes, int eachAttrLength) {
    double sum;
    for (int i = 0; i < numOfAttributes - 1; i++) {
        sum = 0.0;
        for (int j = 0; j < eachAttrLength; j++)
            sum += inputAttributes[i].attrValues[j];
        inputAttributes[i].mean = sum / eachAttrLength;
    }
}

/* Calculates the standard deviation for all the attributes */
void CalculateStandardDeviation(int numOfAttributes, int eachAttrLength) {
    double sumOfSquares;
    double temp;
    for (int i = 0; i < numOfAttributes - 1; i++) {
        sumOfSquares = 0.0;
        temp = 0.0;
        for (int j = 0; j < eachAttrLength; j++) {
            temp = inputAttributes[i].attrValues[j] - inputAttributes[i].mean;
            sumOfSquares += temp*temp;
        }
        inputAttributes[i].standardDeviation = sqrt(sumOfSquares / eachAttrLength);
    }
}

/* Calculates the normalized values for all the attributes */
void CalculateNormalization(int numOfAttributes, int eachAttrLength) {
    double temp;
    for (int i = 0; i < numOfAttributes - 1; i++) {
        temp = 0.0;
        for (int j = 0; j < eachAttrLength; j++) {
            temp = (inputAttributes[i].attrValues[j] - inputAttributes[i].mean) / inputAttributes[i].standardDeviation;
            inputAttributes[i].NormValues.push_back(temp);
        }
    }
}

/* Calculates the distance from different cluster centres */
double calculateDistance(int pos, int clusterNum, int numOfAttributes) {
    double sum = 0;
    if (Normalize) {
        for (int i = 0; i < numOfAttributes - 1; i++)
            if (inputAttributes[i].attrName != classAttr)
                sum += (inputAttributes[i].NormValues[pos] - originalCluster[clusterNum].centroid[i])*(inputAttributes[i].NormValues[pos] - originalCluster[clusterNum].centroid[i]);
    }
    else {
        for (int i = 0; i < numOfAttributes - 1; i++)
            if (inputAttributes[i].attrName != classAttr)
                sum += (inputAttributes[i].attrValues[pos] - originalCluster[clusterNum].centroid[i])*(inputAttributes[i].attrValues[pos] - originalCluster[clusterNum].centroid[i]);
    }
    return sqrt(sum);
}

/* To debug */
void printClusterMembers() {
    int	tempCounter = 0;
    std::cout << "Members: " << '\n';

    for (auto Num = originalCluster.begin(); Num != originalCluster.end(); Num++) {
        std::cout << "  Cluster: " << tempCounter;
        for (auto mem = (*Num).ClusterMember.begin(); mem != (*Num).ClusterMember.end(); mem++) {
            std::cout << " (";
            for (auto it = mem->begin(); it != mem->end(); it++) {
                std::cout << " " << std::setw(5) << std::fixed << std::setprecision(2) << std::dec << *it;
            }
            std::cout << " )";
        }
        tempCounter++;
        std::cout << '\n';
    }
}

/* To debug */
void printClusterCentroids() {
    int	tempCounter = 0;
    std::cout << "Centroids: " << '\n';
    for (auto Num = originalCluster.begin(); Num != originalCluster.end(); Num++) {
        std::cout << "  Cluster: " << tempCounter << " (";
        for (auto mem = (*Num).centroid.begin(); mem != (*Num).centroid.end(); mem++) {
            std::cout << " " << std::setw(5) << std::fixed << std::setprecision(2) << std::dec << *mem;
        }
        std::cout << " )";
        tempCounter++;
        std::cout << '\n';
    }
}

/* Writes the centroids to an appropriate file */
void writeCentroids(int numOfAttributes) {
    for (int i = 0; i < numOfClusters; i++){
        for (int tempCounter = 0; tempCounter < numOfAttributes - 1; tempCounter++){
            Cluster_Center_Basic_Destination << " " << std::setw(7) << std::fixed << std::setprecision(2) << std::dec << originalCluster[i].centroid[tempCounter];
            if (Normalize){
                double convert = 0;
                convert = (originalCluster[i].centroid[tempCounter] * inputAttributes[tempCounter].standardDeviation) + inputAttributes[tempCounter].mean;
                Cluster_Center_UnnormBasic_Destination << " " << std::setw(7) << std::fixed << std::setprecision(2) << std::dec << convert;
            }
        }
        Cluster_Center_Basic_Destination << " " << std::setw(3) << (i + 1) << '\n';
        Cluster_Center_UnnormBasic_Destination << " " << std::setw(3) << (i + 1) << '\n';
    }
}

/* Write all the input data along with cluster ids to an appropriate file */
void writeAttrToFile(int numOfAttributes, int eachAttrLength) {
    for (int tempCounter = 0; tempCounter < eachAttrLength; tempCounter++) {
        for (int i = 0; i < numOfAttributes; i++) {
            if (Normalize){
                if (inputAttributes[i].attrName == "cluster") { // cluster Id is an integer
                    Clustering_Basic_Destination << " " << std::setw(3) << std::dec << (int)inputAttributes[i].NormValues[tempCounter];
                    Clustering_UnnormBasic_Destination << " " << std::setw(3) << std::dec << (int)inputAttributes[i].NormValues[tempCounter];
                }
                else {// Write the normalized values to output file 
                    Clustering_Basic_Destination << " " << std::setw(8) << std::fixed << std::setprecision(4) << std::dec << inputAttributes[i].NormValues[tempCounter];
                    Clustering_UnnormBasic_Destination << " " << std::setw(8) << std::fixed << std::setprecision(4) << std::dec << inputAttributes[i].attrValues[tempCounter];
                }
            }
            else {
                if (inputAttributes[i].attrName == "cluster")
                    Clustering_Basic_Destination << " " << std::setw(3) << std::dec << (int)inputAttributes[i].attrValues[tempCounter];
                else // Write the normalized values to output file 
                    Clustering_Basic_Destination << " " << std::setw(8) << std::fixed << std::setprecision(4) << std::dec << inputAttributes[i].attrValues[tempCounter];
            }
        }
        Clustering_Basic_Destination << '\n';
        Clustering_UnnormBasic_Destination << '\n';
    }
}

/* Checks whether the current input data belongs to perticular cluster or not */
bool CheckInCluster(int pos, int clustNum, int numOfAttributes) {
    bool globalFind = true;
    bool localFind = false;

    for (int mem = 0; mem < originalCluster[clustNum].ClusterMember.size(); mem++) {
        globalFind = true;
        for (int attr = 0; attr < numOfAttributes - 1; attr++) {
            if (inputAttributes[attr].attrName != classAttr) {
                if (Normalize){
                    if (originalCluster[clustNum].ClusterMember.at(mem).at(attr) != inputAttributes[attr].NormValues[pos]){
                        localFind = false;
                        break;
                    }
                    else
                        localFind = true;
                }
                else{
                    if (originalCluster[clustNum].ClusterMember.at(mem).at(attr) != inputAttributes[attr].attrValues[pos]){
                        localFind = false;
                        break;
                    }
                    else
                        localFind = true;
                }
                globalFind &= localFind;
            }
        }
        globalFind &= localFind;
        if (globalFind)
            return true;
    }
    return false;
}

/* Assign the cluster ids to the input data */
void assignClusterID(int numOfAttributes, int eachAttrLength) {
    for (int pos = 0; pos < eachAttrLength; pos++){
        std::vector<double> temp;
        for (int k = 0; k < numOfAttributes - 1; k++)
            temp.push_back(inputAttributes[k].attrValues[pos]);
        for (int clustNum = 0; clustNum < numOfClusters; clustNum++){
            if (CheckInCluster(pos, clustNum, numOfAttributes)){
                if (Normalize)
                    inputAttributes[numOfAttributes - 1].NormValues.push_back(clustNum + 1);
                else
                    inputAttributes[numOfAttributes - 1].attrValues.push_back(clustNum + 1);
                break;
            }
        }
    }
}

/* Calculates the centroid */
void calculateCentroid(int numOfAttributes, int numOfClusters, int eachAttrLength) {
    for (int Num = 0; Num < numOfClusters; Num++) {
        std::vector<double> sum(numOfAttributes - 1, 0);
        for (int mem = 0; mem < originalCluster[Num].ClusterMember.size(); mem++) {
            for (int it = 0; it < numOfAttributes - 1; it++) {
                sum[it] += originalCluster[Num].ClusterMember.at(mem).at(it);
            }
        }
        for (int it = 0; it < numOfAttributes - 1; it++)
            originalCluster[Num].centroid[it] = (double)sum[it] / originalCluster[Num].ClusterMember.size();
    }
}

/* Initializes the centroids */
void initializeCentroids(int numOfAttributes, int eachAttrLength, std::vector<double>& distance) {
    int index;
    bool localFind = false;
    bool globalFind = true;
    bool firstCluster = true;
    srand(time(NULL));

    for (int clusterNum = 0; clusterNum < numOfClusters; clusterNum++) {
        distance.push_back(0);
        originalCluster.push_back(clusters);
        last_temp_cluster.push_back(clusters);

        firstCluster = (originalCluster.size() == 1);
        // generate random number untill no two centroids match
        do{
            index = rand() % eachAttrLength;
            for (int i = 0; i < originalCluster.size() - 1; i++) {
                globalFind = true;
                for (int attr = 0; attr < numOfAttributes - 1; attr++) {
                    if (inputAttributes[attr].attrName != classAttr) {
                        if (Normalize){
                            if (inputAttributes[attr].NormValues[index] != originalCluster[i].centroid[attr]) {
                                localFind = false;
                                break;
                            }
                            else
                                localFind = true;
                        }
                        else{
                            if (inputAttributes[attr].attrValues[index] != originalCluster[i].centroid[attr]) {
                                localFind = false;
                                break;
                            }
                            else
                                localFind = true;
                        }
                        globalFind &= localFind;
                    }
                }
                globalFind &= localFind;
                if (globalFind)
                    break;
            }
        } while ((!firstCluster) && globalFind);

        if (Normalize){
            for (int attr = 0; attr < numOfAttributes - 1; attr++){
                originalCluster[clusterNum].centroid.push_back(inputAttributes[attr].NormValues[index]);
                last_temp_cluster[clusterNum].centroid.push_back(0);
            }
        }
        else{
            for (int attr = 0; attr < numOfAttributes - 1; attr++){
                originalCluster[clusterNum].centroid.push_back(inputAttributes[attr].attrValues[index]);
                last_temp_cluster[clusterNum].centroid.push_back(0);
            }
        }
    }
}

/* K means clustering */
void doClustering(int numOfAttributes, int eachAttrLength)  {
    std::vector <double> tempClusterMember(numOfAttributes - 1);
    std::vector <double> distance(numOfClusters);
    bool goOn = true;
    initializeCentroids(numOfAttributes, eachAttrLength, distance);
    while (goOn) {
        goOn = true;
        /* To debug:
        printClusterCentroids(); */

        last_temp_cluster = originalCluster;

        for (int k = 0; k < numOfClusters; k++)
            originalCluster[k].ClusterMember.erase(originalCluster[k].ClusterMember.begin(), originalCluster[k].ClusterMember.end());

        // calculate distance to all the clusters from the given point
        for (int pos = 0; pos < eachAttrLength; pos++){
            for (int clusterNum = 0; clusterNum < numOfClusters; clusterNum++)
                distance[clusterNum] = calculateDistance(pos, clusterNum, numOfAttributes);

            // choose which cluster is closer
            int lowerIndex = 0;
            for (int k = 0; k<numOfClusters; k++)
                if (distance[lowerIndex] > distance[k])
                    lowerIndex = k;
            if (Normalize){
                for (int k = 0; k < numOfAttributes - 1; k++)
                    tempClusterMember[k] = inputAttributes[k].NormValues[pos];
            }
            else {
                for (int k = 0; k < numOfAttributes - 1; k++)
                    tempClusterMember[k] = inputAttributes[k].attrValues[pos];
            }
            // assign the given point to the cluster which is near
            originalCluster[lowerIndex].ClusterMember.push_back(tempClusterMember);
        }

        //printClusterMembers();
        calculateCentroid(numOfAttributes, numOfClusters, eachAttrLength);
        // check does the old centroids and new centroids same? if yes terminate
        for (int k = 0; (k < numOfClusters)&&(goOn); k++){
            for (int attr = 0; attr < numOfAttributes - 1; attr++){
                if (inputAttributes[attr].attrName != classAttr) {
                    if (fabs(last_temp_cluster[k].centroid[attr] - originalCluster[k].centroid[attr]) > 0.01){
                        goOn = false;
                        break;
                    }
                }
            }
        }
    }
}

/* Opens the input and output files */
void openFile()  {
    source.open(inputFileName.c_str());
    if (!source)
        std::cerr << "Error: Unable to open the input file" << '\n';
    if (!Normalize){
        if (!userOutputFileName){
            Cluster_Center_Basic_OutFileName = "Kmeans_ClusterCenters_" + std::to_string(numOfClusters) + "_" + inputFileName;
            Clustering_Basic_OutFileName = "Kmeans_Clustering_" + std::to_string(numOfClusters) + "_" + inputFileName;
        }
        else{
            Cluster_Center_Basic_OutFileName = "ClusterCenters_" + outputFileName;
            Clustering_Basic_OutFileName = outputFileName;
        }
        Cluster_Center_Basic_Destination.open(Cluster_Center_Basic_OutFileName.c_str());
        Clustering_Basic_Destination.open(Clustering_Basic_OutFileName.c_str());
        if (!Cluster_Center_Basic_Destination || !Clustering_Basic_Destination)
            std::cerr << "Error: Unable to open one of the output file" << '\n';
    }
    else {
        if (!userOutputFileName){
            Cluster_Center_Basic_OutFileName = "Kmeans_ClusterCenters_Normalized_" + std::to_string(numOfClusters) + "_" + inputFileName;
            Clustering_Basic_OutFileName = "Kmeans_Clustering_Normalized_" + std::to_string(numOfClusters) + "_" + inputFileName;
            Cluster_Center_UnnormBasic_OutFileName = "Kmeans_ClusterCenters_Unnormalized_" + std::to_string(numOfClusters) + "_" + inputFileName;
            Clustering_UnnormBasic_OutFileName = "Kmeans_Clustering_Unnormalized_" + std::to_string(numOfClusters) + "_" + inputFileName;
        }
        else{
            Cluster_Center_Basic_OutFileName = "ClusterCenters_Normalized_" + outputFileName;
            Clustering_Basic_OutFileName = "Normalized_" + outputFileName;
            Cluster_Center_UnnormBasic_OutFileName = "ClusterCenters_Unnormalized_" + outputFileName;
            Clustering_UnnormBasic_OutFileName = "Unnormalized_" + outputFileName;
        }
        Cluster_Center_Basic_Destination.open(Cluster_Center_Basic_OutFileName.c_str());
        Clustering_Basic_Destination.open(Clustering_Basic_OutFileName.c_str());
        Cluster_Center_UnnormBasic_Destination.open(Cluster_Center_UnnormBasic_OutFileName.c_str());
        Clustering_UnnormBasic_Destination.open(Clustering_UnnormBasic_OutFileName.c_str());

        if (!Cluster_Center_Basic_Destination || !Clustering_Basic_Destination
            || !Cluster_Center_UnnormBasic_Destination || !Clustering_UnnormBasic_Destination)
            std::cerr << "Error: Unable to open one of the output file" << '\n';
    }
}

/* Closes the input and output files */
void closeFile() {
    source.close();
    Cluster_Center_Basic_Destination.close();
    Clustering_Basic_Destination.close();
    if (Normalize){
        Cluster_Center_UnnormBasic_Destination.close();
        Clustering_UnnormBasic_Destination.close();
    }
}

/* Main Program */
int main(int argc, char *argv[]) {
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
    Cluster_Center_Basic_Destination << currentLine << '\n' << '\n';
    Clustering_Basic_Destination << currentLine << '\n' << '\n';
    if (Normalize){
        Cluster_Center_UnnormBasic_Destination << currentLine << '\n' << '\n';
        Clustering_UnnormBasic_Destination << currentLine << '\n' << '\n';
    }
    while (source.good()){
        if (currentLine.length() != 0){        // discards the empty lines
            std::istringstream ss(currentLine);
            if (currentLine.at(0) == '@') {
                ss.ignore();
                ss >> holdAttrName;
                if (holdAttrName == isAttribute) {  // get the attribute names
                    ss >> holdAttrName;
                    inputAttributes.push_back(temp_eachAttr);
                    inputAttributes[numOfAttributes].attrName = holdAttrName;
                    numOfAttributes++;
                }
                // if the data is going to start end this loop
                if (holdAttrName == "data") {
                    inputAttributes.push_back(temp_eachAttr);
                    inputAttributes[numOfAttributes].attrName = "cluster";
                    numOfAttributes++;
                    for (int i = 0; i < numOfAttributes; i++) {
                        Cluster_Center_Basic_Destination << "@attribute " << inputAttributes[i].attrName << " real" << '\n';
                        Clustering_Basic_Destination << "@attribute " << inputAttributes[i].attrName << " real" << '\n';
                        if (Normalize){
                            Cluster_Center_UnnormBasic_Destination << "@attribute " << inputAttributes[i].attrName << " real" << '\n';
                            Clustering_UnnormBasic_Destination << "@attribute " << inputAttributes[i].attrName << " real" << '\n';
                        }
                    }

                    Cluster_Center_Basic_Destination << '\n' << currentLine << '\n';
                    Clustering_Basic_Destination << '\n' << currentLine << '\n';
                    if (Normalize){
                        Cluster_Center_UnnormBasic_Destination << '\n' << currentLine << '\n';
                        Clustering_UnnormBasic_Destination << '\n' << currentLine << '\n';
                    }
                    break;
                }
            }
        }
        getline(source, currentLine);
    }

    /* Reads the data from input file and stores in a vector */
    getline(source, currentLine);
    while (source.good()){
        if (currentLine.length() != 0) {	 // discards the empty lines
            tempCounter = 0;
            std::istringstream ss(currentLine);
            // Extracts the numbers from the current line
            while (ss >> readValue) {
                // save the numbers to a vector
                inputAttributes[tempCounter].attrValues.push_back(readValue);
                // if the delimiter is ',' ignore that character 
                if (ss.peek() == ',')
                    ss.ignore();
                tempCounter++;
            }
        }
        getline(source, currentLine);
    }

    // Get the size of each attribute 
    eachAttrLength = inputAttributes[0].attrValues.size();
    // Calculate the mean, standard deviation, and normalized values 
    CalculateMean(numOfAttributes, eachAttrLength);
    CalculateStandardDeviation(numOfAttributes, eachAttrLength);
    CalculateNormalization(numOfAttributes, eachAttrLength);
    doClustering(numOfAttributes, eachAttrLength);

    // Writecluster centers to the appropriate output file
    writeCentroids(numOfAttributes);
    /* Assign cluster ids */
    assignClusterID(numOfAttributes, eachAttrLength);
    writeAttrToFile(numOfAttributes, eachAttrLength);
    /* Close all the input and output files */
    closeFile();
    return 0;
}
