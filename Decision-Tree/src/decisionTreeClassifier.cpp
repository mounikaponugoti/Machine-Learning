/****************************************************************************************************
* This program prepares the decision tree for the given input data set. To train the tree, only part 
* of the data (%) selected by the user from the given input data is used. After training, entire data 
* is tested and predicted class is stroed as a derived class attribute. Three output files, one file
* holds the  tree, one file holds the input data along with derived class, and last file includes the 
* comfusion matrix and accuracy of the prediction.
******************************************************************************************************/
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

struct Node{
    std::string nodeId;
    double entropy;
    std::string attrUsed;
    std::vector<Node*> childsList;
    double whichClass;
    double majorityClass;
    bool isRoot;
    Node(){
        nodeId = " ";
        attrUsed = " ";
        entropy = 0;
        whichClass = DBL_MAX;
        majorityClass = DBL_MAX;
        isRoot = true;
    }
} Nodes;

Node* rootNode = new Node;

struct eachAttribute {
    std::vector<double> attrValues;
    std::string attrName;
    std::vector<double> possibleAttrValues;
    eachAttribute(){
        attrName = " ";
    }
} eachAttrs;

std::vector<eachAttribute> inputAttributes;

struct countClassesForAttributes{
    int classCount;
    double classValue;
    countClassesForAttributes(){
        classCount = 0;
        classValue = DBL_MAX;
    }
} temp_countClassesForAttributes;

struct classCountInfoOfAttribute{
    std::vector<countClassesForAttributes> Classes;
    double oneOfAttrValue;
    double whichClassHasMaxCount;
    double Information;
    int totalCount;
    int countMaxValue;
    classCountInfoOfAttribute(){
        Information = 0;
        totalCount = 0;
        countMaxValue = 0;
        whichClassHasMaxCount = DBL_MAX;
    }
} temp_classCountInfoOfAttribute;

struct remainingAttributes{
    std::vector<classCountInfoOfAttribute> attribute;
    std::string nameOfAttr;
    int totalCount;
    double entropy;
    remainingAttributes(){
        entropy = 0;
        nameOfAttr = " ";
        totalCount = 0;
    }
} temp_remainingAttributes;

std::string classAttr = " ";
std::ifstream source;
std::ofstream Train_Destination;
std::ofstream Accuracy_Destination;
std::ofstream Apply_Destination;

std::string Train_OutFileName;
std::string Accuracy_OutFileName;
std::string Apply_OutFileName;
std::string inputFileName;

int eachAttrLength = 0;
int numOfAttributes = 0;
int classAttrNum;
int classAttrPossibleValuesCount = 1;

// % of inputdata to train from index 0
int per= 100;

std::vector<classCountInfoOfAttribute> accuracy;

/* Writes the trained tree to an appropriate file */
void writeTheTree(Node *root, std::string spaces) {
    for (int i = 0; i < root->childsList.size(); i++){
        if (root->childsList[i]->whichClass != -DBL_MAX){
            Train_Destination << '\n' << spaces << root->attrUsed << " =" << root->childsList[i]->nodeId;
            if (!(root->childsList[i]->isRoot)){
                Train_Destination << " " << root->childsList[i]->whichClass;
            }
            else{
                std::string space = spaces + "| ";
                writeTheTree(root->childsList[i], space);
            }
        }
    }
}

/* Write all the input data along with derived class to an appropriate file */
void writeAttrToFile() {
    for (int tempCounter = 0; tempCounter < inputAttributes[0].attrValues.size(); tempCounter++){
        for (int i = 0; i < numOfAttributes; i++)
            Apply_Destination << " " << std::setw(8) << std::fixed << std::setprecision(4) << std::dec << inputAttributes[i].attrValues[tempCounter];
        Apply_Destination << '\n';
    }
}

/* To print line of special char with defined length*/
std::string print(int width, char data) {
    std::string str = "";
    for (int i = 0; i < width; i++)
        str = str + data;
    return str;
}

/* Write accuracy to a file*/
void writeAccuracyToFile() {
    double temp = 0;
    double totalAccuracy = 0;
    Accuracy_Destination << print(classAttrPossibleValuesCount * 20, '-') << '\n';
    Accuracy_Destination << std::setw(classAttrPossibleValuesCount * 15) << "Predicted Class" << '\n';
    Accuracy_Destination << print(classAttrPossibleValuesCount * 20, '-') << '\n' << std::setw(10) << "Actual Class|";
    for (int i = 0; i < classAttrPossibleValuesCount; i++)
        Accuracy_Destination << std::setw(10) << std::fixed << std::setprecision(2) << std::dec << inputAttributes[classAttrNum].possibleAttrValues[i];
    Accuracy_Destination << '\n' << print(classAttrPossibleValuesCount * 20, '-') << '\n';
    for (int i = 0; i < classAttrPossibleValuesCount; i++){
        Accuracy_Destination << std::setw(8) << std::fixed << std::setprecision(2) << std::dec << inputAttributes[classAttrNum].possibleAttrValues[i] << "    |";
        for (int j = 0; j < classAttrPossibleValuesCount; j++){
            if (accuracy[i].totalCount != 0)
                temp = ((double)accuracy[i].Classes[j].classCount);// / accuracy[i].totalCount);
            if (i == j){
                totalAccuracy += ((double)accuracy[i].Classes[j].classCount / accuracy[i].totalCount);
            }
            Accuracy_Destination << std::setw(10) << std::fixed << std::setprecision(2) << std::dec << temp;
        }
        Accuracy_Destination << '\n' << std::setw(sizeof("Actual Class"));
    }
    Accuracy_Destination << '\n' << '\n' << "The acuracy is " << std::setw(6) << std::fixed << std::setprecision(5) << std::dec << ((totalAccuracy) / classAttrPossibleValuesCount) * 100 << "  %" << '\n';
}

/* Counts the number of predictions and mispredictions*/
void compareOriginalAndDerivedClasses() {
    // Initialize the vector to holds the correct counts and bad counts
    accuracy.resize(classAttrPossibleValuesCount);
    for (int i = 0; i < classAttrPossibleValuesCount; i++){
        for (int j = 0; j < classAttrPossibleValuesCount; j++){
            accuracy[i].Classes.push_back(temp_countClassesForAttributes);
            accuracy[i].Classes[j].classCount = 0;
            accuracy[i].Classes[j].classValue = inputAttributes[classAttrNum].possibleAttrValues[j];
        }
        accuracy[i].oneOfAttrValue = inputAttributes[classAttrNum].possibleAttrValues[i];
    }
    // Counting is done here
    for (int num = 0; num < inputAttributes[classAttrNum].attrValues.size(); num++){
        for (int i = 0; i < classAttrPossibleValuesCount; i++){
            if (inputAttributes[classAttrNum].attrValues[num] == accuracy[i].oneOfAttrValue){
                for (int j = 0; j < classAttrPossibleValuesCount; j++){
                    if (inputAttributes[numOfAttributes - 1].attrValues[num] == accuracy[i].Classes[j].classValue){
                        accuracy[i].Classes[j].classCount++;
                    }
                }
                accuracy[i].totalCount++;
            }
        }
    }
}

/* Compares the node id's */
bool compareNodeIds(std::string nodeId, double number) {
    std::stringstream ss;
    ss << number;
    return (nodeId == ss.str());
}

/* Returns which class is dominating for a given node*/
double whichHasMaxCount(struct remainingAttributes attributeData) {
    std::vector<countClassesForAttributes> countMaxClass;
    double maxCountClass = -DBL_MAX;
    double maxCount = -DBL_MAX;
    countMaxClass.resize(attributeData.attribute.size());
    for (int i = 0; i < countMaxClass.size(); i++){
        countMaxClass[i].classValue = inputAttributes[classAttrNum].possibleAttrValues[i];
    }
    // Count all the possible classes for the given node
    for (int i = 0; i < countMaxClass.size(); i++){
        for (int j = 0; j < attributeData.attribute[0].Classes.size(); j++){
            if (countMaxClass[i].classValue == attributeData.attribute[i].Classes[j].classValue)
                countMaxClass[i].classCount += attributeData.attribute[i].Classes[j].classCount;
        }
    }
    // Get which class is dominating
    for (int i = 0; i < countMaxClass.size(); i++){
        if (maxCount < countMaxClass[i].classCount){
            maxCount = countMaxClass[i].classCount;
            maxCountClass = countMaxClass[i].classValue;
        }
    }
    return maxCountClass;
}

/* Returns which class the current data belongs to */
double whichClass(Node *root, int num) {
    // Is current node is parent?
    if (root->isRoot){
        for (int attr = 0; attr < numOfAttributes - 1; attr++){
            if (root->attrUsed == inputAttributes[attr].attrName){
                for (int i = 0; i < root->childsList.size(); i++){
                    if (compareNodeIds(root->childsList[i]->nodeId, inputAttributes[attr].attrValues[num])){
                        double dClass = whichClass(root->childsList[i], num);
                        if (root->childsList[i]->whichClass == -DBL_MAX){
                            return root->childsList[i]->majorityClass;
                        }
                        return dClass;
                    }
                }
            }
        }
    }
    // Is current node is child
    else {
        if (root->whichClass == -DBL_MAX){
            return root->majorityClass;
        }
        return root->whichClass;
    }
}

/* Function to test the input data set after training*/
void doTesting(Node *root) {
    for (int i = 0; i < inputAttributes[0].attrValues.size(); i++){
        inputAttributes[numOfAttributes - 1].attrValues[i] = whichClass(root, i);
    }
}

/* Initialize the root node */
void initializeRootNode(Node *root) {
    root->nodeId = "None";
}

void doTraining(std::vector<eachAttribute>reducedInputAttributes, eachAttribute classAttributeValues, Node *parent, double parentEntropy) {
    double entropyMin = DBL_MAX;
    std::string entropyMinAttr;
    int attrPos = 0;
    std::vector<remainingAttributes> remainingAttrToClassify;
    std::vector<unsigned long int> tempCounter;
    // Does all the atrributes are used?
    if ((!reducedInputAttributes.empty())) {
        // Does all the data is classified?
        if ((!reducedInputAttributes[0].attrValues.empty())){
            for (int attr = 0; attr < reducedInputAttributes.size(); attr++){
                remainingAttrToClassify.push_back(temp_remainingAttributes);
                remainingAttrToClassify[attr].totalCount = 0;
                for (int count = 0; count < reducedInputAttributes[attr].possibleAttrValues.size(); count++){
                    remainingAttrToClassify[attr].attribute.push_back(temp_classCountInfoOfAttribute);
                    remainingAttrToClassify[attr].nameOfAttr = reducedInputAttributes[attr].attrName;
                    remainingAttrToClassify[attr].attribute[count].oneOfAttrValue = reducedInputAttributes[attr].possibleAttrValues[count];
                    for (int j = 0; j < classAttrPossibleValuesCount; j++){
                        remainingAttrToClassify[attr].attribute[count].Classes.push_back(temp_countClassesForAttributes);
                        remainingAttrToClassify[attr].attribute[count].Classes[j].classValue = classAttributeValues.possibleAttrValues[j];
                    }
                    for (int num = 0; num < reducedInputAttributes[0].attrValues.size(); num++){
                        if (reducedInputAttributes[attr].attrValues[num] == reducedInputAttributes[attr].possibleAttrValues[count]){
                            remainingAttrToClassify[attr].attribute[count].totalCount++;
                            for (int j = 0; j < classAttrPossibleValuesCount; j++){
                                if (remainingAttrToClassify[attr].attribute[count].Classes[j].classValue == classAttributeValues.attrValues[num])
                                    remainingAttrToClassify[attr].attribute[count].Classes[j].classCount++;
                            }
                        }
                    }
                    for (int j = 0; j < classAttrPossibleValuesCount; j++){
                        if (remainingAttrToClassify[attr].attribute[count].countMaxValue < remainingAttrToClassify[attr].attribute[count].Classes[j].classCount){
                            remainingAttrToClassify[attr].attribute[count].countMaxValue = remainingAttrToClassify[attr].attribute[count].Classes[j].classCount;
                            remainingAttrToClassify[attr].attribute[count].whichClassHasMaxCount = remainingAttrToClassify[attr].attribute[count].Classes[j].classValue;
                        }
                    }
                    remainingAttrToClassify[attr].totalCount += remainingAttrToClassify[attr].attribute[count].totalCount;
                    for (int j = 0; j < classAttrPossibleValuesCount; j++){
                        double temp = ((double)remainingAttrToClassify[attr].attribute[count].Classes[j].classCount / remainingAttrToClassify[attr].attribute[count].totalCount);
                        if ((temp != 0) && (remainingAttrToClassify[attr].attribute[count].totalCount != 0)){
                            remainingAttrToClassify[attr].attribute[count].Information += ((-1)*temp*log2(temp));
                        }
                    }
                }
                // Wait to count total and calculte the entropy.
                for (int count = 0; count < reducedInputAttributes[attr].possibleAttrValues.size(); count++){
                    remainingAttrToClassify[attr].entropy += (((double)remainingAttrToClassify[attr].attribute[count].totalCount / remainingAttrToClassify[attr].totalCount)
                        *remainingAttrToClassify[attr].attribute[count].Information);
                }
                // Which attribute has minimum entropy?
                if (entropyMin > remainingAttrToClassify[attr].entropy){
                    entropyMin = remainingAttrToClassify[attr].entropy;
                    entropyMinAttr = remainingAttrToClassify[attr].nameOfAttr;
                    attrPos = attr;
                }
            }
        }
        parent->majorityClass = whichHasMaxCount(remainingAttrToClassify[attrPos]);

        // Check if the root node is originally a root node? If the entropy is not different from the parent entropy mark it as leaf.
        // If it is orinally a parent node then check, does it has any child as leaf node? if yes mark it otherwise mark that child as parent.
        if (parentEntropy != entropyMin){
            parent->attrUsed = entropyMinAttr;
            parent->isRoot = true;
            for (int count = 0; count < remainingAttrToClassify[attrPos].attribute.size(); count++){
                std::stringstream ss;
                parent->childsList.push_back(new Node);
                ss << remainingAttrToClassify[attrPos].attribute[count].oneOfAttrValue;
                parent->childsList[count]->nodeId = ss.str();
                // Is it a leaf node??
                if ((remainingAttrToClassify[attrPos].attribute[count].countMaxValue >= (0.999*remainingAttrToClassify[attrPos].attribute[count].totalCount)) ||
                    ((reducedInputAttributes.size() - 1) == 0) || (entropyMin == 0) || (entropyMin == DBL_MAX)){
                    parent->childsList[count]->isRoot = false;
                    parent->childsList[count]->whichClass = remainingAttrToClassify[attrPos].attribute[count].whichClassHasMaxCount;
                    if (remainingAttrToClassify[attrPos].attribute[count].whichClassHasMaxCount == DBL_MAX){
                        parent->childsList[count]->whichClass = -DBL_MAX;
                        parent->childsList[count]->majorityClass = parent->majorityClass;
                    }
                }
                else{ // Is it a parent node?
                    int getWhichAttr = 0;
                    std::vector<eachAttribute>reducedInputFunctionParameter = reducedInputAttributes;
                    eachAttribute reducedClassAttributeValues = classAttributeValues;
                    // Get the name of the attribute which has low entropy
                    for (int attr = 0; attr < reducedInputAttributes.size(); attr++){
                        if (reducedInputAttributes[attr].attrName == entropyMinAttr){
                            getWhichAttr = attr;
                            break;
                        }
                    }
                    for (int attr = 0; attr < reducedInputAttributes.size(); attr++){
                        reducedInputFunctionParameter[attr].attrValues.erase(reducedInputFunctionParameter[attr].attrValues.begin(), reducedInputFunctionParameter[attr].attrValues.end());
                    }
                    reducedClassAttributeValues.attrValues.erase(reducedClassAttributeValues.attrValues.begin(), reducedClassAttributeValues.attrValues.end());
                    for (int num = 0; num < reducedInputAttributes[0].attrValues.size(); num++){
                        if (remainingAttrToClassify[attrPos].attribute[count].oneOfAttrValue == reducedInputAttributes[getWhichAttr].attrValues[num]){
                            for (int attr = 0; attr < reducedInputAttributes.size(); attr++){
                                reducedInputFunctionParameter[attr].attrValues.push_back(reducedInputAttributes[attr].attrValues[num]);
                            }
                            reducedClassAttributeValues.attrValues.push_back(classAttributeValues.attrValues[num]);
                        }
                    }
                    parent->childsList[count]->whichClass = remainingAttrToClassify[attrPos].attribute[count].whichClassHasMaxCount;
                    if (remainingAttrToClassify[attrPos].attribute[count].whichClassHasMaxCount == DBL_MAX)
                        parent->childsList[count]->whichClass = -DBL_MAX;
                    reducedInputFunctionParameter.erase(reducedInputFunctionParameter.begin() + getWhichAttr);
                    // Recurssive call
                    doTraining(reducedInputFunctionParameter, reducedClassAttributeValues, parent->childsList[count], remainingAttrToClassify[attrPos].attribute[count].Information);
                    // In last call if that node was orinally not a parent node
                    if (!(parent->childsList[count]->isRoot))
                    {
                        parent->childsList[count]->whichClass = remainingAttrToClassify[attrPos].attribute[count].whichClassHasMaxCount;
                        if (remainingAttrToClassify[attrPos].attribute[count].whichClassHasMaxCount == DBL_MAX){
                            parent->childsList[count]->whichClass = -DBL_MAX;
                            parent->childsList[count]->majorityClass = parent->majorityClass;
                        }
                    }

                }
            }
        }
        else parent->isRoot = false;
    }
}
/* Read the command line */
int readCommandLine(int argc, char **argv) {
    int argi;
    if (argc <= 1) {
        std::cout << "./decisionTreeClassifier -i inputfile -c classattribute -T %data" << '\n';
        exit(1);
    }
    for (argi = 1; argi < argc; argi++){
        if (!strcmp(argv[argi], "-i"))
            inputFileName = argv[argi + 1];
        if (!strcmp(argv[argi], "-T"))
            per= atoi(argv[argi + 1]);
        if (!strcmp(argv[argi], "-c"))
            classAttr = argv[argi + 1];
    }

    return 0;
}

/* Opens the input and output files */
void openFile() {
    source.open(inputFileName.c_str());
    if (!source)
        std::cerr << "Error: Unable to open the input file" << '\n';
    Train_OutFileName = "decisionTreeClassifierTrain" + std::to_string(per) + inputFileName;
    Train_Destination.open(Train_OutFileName.c_str());
    Accuracy_OutFileName = "decisionTreeClassifierAccuracy" + std::to_string(per) + inputFileName;
    Apply_OutFileName = "decisionTreeClassifierApply" + std::to_string(per) + inputFileName;
    Accuracy_Destination.open(Accuracy_OutFileName.c_str());
    Apply_Destination.open(Apply_OutFileName.c_str());
    if (!Accuracy_Destination || !Apply_Destination || !Train_Destination)
        std::cerr << "Error: Unable to open one of the output file" << '\n';
}

/* Closes the input and output files */
void closeFile() {
    source.close();
    Train_Destination.close();
    Accuracy_Destination.close();
    Apply_Destination.close();
}

/* Main Program */
int main(int argc, char *argv[]) {
    std::string currentLine;
    std::string isAttribute = "attribute";
    std::string holdAttrName = " ";
    std::string spaces = "";
    double readValue = 0.0;
    int tempCounter = 0;

    readCommandLine(argc, argv);
    openFile();

    // Handels the lines which are in the beginning of the input file
    getline(source, currentLine);
    Apply_Destination << currentLine << '\n' << '\n';

    while (source.good()){
        if (currentLine.length() != 0){        // discards the empty lines
            std::istringstream ss(currentLine);
            if (currentLine.at(0) == '@') {
                ss.ignore();
                ss >> holdAttrName;
                if (holdAttrName == isAttribute) {  // get the attribute names
                    ss >> holdAttrName;
                    inputAttributes.push_back(eachAttrs);
                    inputAttributes[numOfAttributes].attrName = holdAttrName;
                    ss.ignore(10, '{');
                    while (ss >> readValue){
                        inputAttributes[numOfAttributes].possibleAttrValues.push_back(readValue);
                        if (ss.peek() == ',')
                            ss.ignore();
                    }
                    if (inputAttributes[numOfAttributes].attrName == classAttr)
                        classAttrNum = numOfAttributes;
                    numOfAttributes++;
                }
                // If the data is going to start end this loop
                if (holdAttrName == "data") {
                    inputAttributes.push_back(eachAttrs);
                    inputAttributes[numOfAttributes].attrName = "dt_class";
                    numOfAttributes++;
                    for (int i = 0; i < numOfAttributes; i++) {
                        Apply_Destination << "@attribute " << inputAttributes[i].attrName << " real" << '\n';
                    }

                    Apply_Destination << '\n' << currentLine << '\n';
                    break;
                }
            }
        }
        getline(source, currentLine);
    }

    // Reads the data from input file and stores in a vector
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

    classAttrPossibleValuesCount = inputAttributes[classAttrNum].possibleAttrValues.size();
    // Get the size of each attribute 
    eachAttrLength = inputAttributes[0].attrValues.size();
    std::vector<eachAttribute>reducedInputFunctionParameter;
    // Get the last index of the training data set
    int index = ((double)per/ 100)*eachAttrLength;
    for (int i = 0; i < numOfAttributes - 1; i++){
        reducedInputFunctionParameter.push_back(eachAttrs);
        reducedInputFunctionParameter[i].attrName = inputAttributes[i].attrName;
        reducedInputFunctionParameter[i].possibleAttrValues = inputAttributes[i].possibleAttrValues;
        for (int j = 0; j < index; j++){
            reducedInputFunctionParameter[i].attrValues.push_back(inputAttributes[i].attrValues[j]);
        }
    }
    // Delete the class attribute from the reduced data
    reducedInputFunctionParameter.erase(reducedInputFunctionParameter.begin() + classAttrNum);
    initializeRootNode(rootNode);
    doTraining(reducedInputFunctionParameter, inputAttributes[classAttrNum], rootNode, 1);

    // Initialize the new attribute to hold derived class
    for (int i = 0; i < inputAttributes[0].attrValues.size(); i++){
        inputAttributes[numOfAttributes - 1].attrValues.push_back(0);
    }

    doTesting(rootNode);
    compareOriginalAndDerivedClasses();

    // Write the tree to the file
    writeTheTree(rootNode, spaces);
    // Write the entire input data along with new derived class
    writeAttrToFile();
    // Write the confusion matrix
    writeAccuracyToFile();
    // Close all the input and output files
    closeFile();

    return 0;
}
