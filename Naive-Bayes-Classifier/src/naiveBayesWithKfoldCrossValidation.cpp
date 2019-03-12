/************************************************************************************************
* This program implements the naive Bayesian classifier. The classifier is tested by using
* k-fold cross validation. The input data is partioned into K-folds using statified sampling.
* The Nth classifier is trained using all the folds exept the Nth fold in the data test and
* tested using Nth fold. The predicted class for the tested data is stored as a derived class
* attribute. The perfomance characteristics gathered over the whole program run are written to
* two output files, one holds the input data along with derived class and other holds
* the comfusion matrix and accuracy of the prediction.
***************************************************************************************************/
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

struct eachAttribute {
    std::vector<double> attrValues;
    std::string attrName;
    std::vector<double> possibleAttrValues;
    eachAttribute() {
        attrName = " ";
    }
}temp_eachAttr;

struct countClassesForAttributes {
    int classCount;
    double classValue;
    double probability;
    countClassesForAttributes() {
        classCount = 0;
        classValue = DBL_MAX;
        probability = 0;
    }
} countClasseValuesForAttr;

struct classCountInfoOfAttributes {
    std::vector<countClassesForAttributes> Classes;
    double oneOfAttrValue;
    int totalCount;
    classCountInfoOfAttributes() {
        totalCount = 0;
        oneOfAttrValue = DBL_MAX;
    }
} temp_countInfoOfAttr;

struct remainingAttributes {
    std::vector<classCountInfoOfAttributes> attribute;
    std::string nameOfAttr;
    remainingAttributes() {
        nameOfAttr = " ";
    }
} temp_remainingAttr;

struct prob {
    double probability;
    double classValue;
    prob() {
        probability = 1;
        classValue = DBL_MAX;
    }
};

struct eachFold {
    double total;
    double correctPredicted;
    eachFold() {
        total = 0;
        correctPredicted = 0;
    }
};

std::vector<eachFold> eachFoldCount;
std::vector<eachAttribute> inputAttributes;
std::vector<countClassesForAttributes> totalClassesCount;
std::vector<classCountInfoOfAttributes> accuracy;
std::vector<remainingAttributes> countClaasesEachAttr;
std::vector<countClassesForAttributes> folds;
std::vector<prob> probabilityVector;

std::string classAttr = " ";
std::ifstream source;
std::ofstream Accuracy_Destination;
std::ofstream Apply_Destination;

std::string outputFileName;
std::string Accuracy_OutFileName;
std::string Apply_OutFileName;
std::string inputFileName;

int eachAttributeLength = 0;
int numOfAttributes = 0;
int classAttrNum;
int classAttrPossibleValuesCount = 1;
int K = 2;
bool userOutputFileName = false;

std::string help_msg = "-h or --help\n"
"  To print this message\n"
"-i [inputFileName] or --input [inputFileName]\n"
"  To specify input file name\n"
"-o [outputFileName] or --output [outputFileName]\n"
"  To specify output file name, default - naiveBayesian_Output_[FoldClassification/FoldConfusion]_inputfilename.txt\n"
"-k [numberOfFolds] or --foldss [numberOfFolds]\n"
"  To specify number of number of required clusters, default - 2\n"
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
        else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--classattribute") == 0) {
            classAttr = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--folds") == 0) {
            K = atoi(argv[i + 1]);
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

/* Write all the input data along with derived class to an appropriate file */
void writeAttrToFile() {
    for (int tempCounter = 0; tempCounter < inputAttributes[0].attrValues.size(); tempCounter++) {
        for (int i = 0; i < numOfAttributes - 1; i++) {
            Apply_Destination << " " << std::setw(8) << std::fixed
                << std::setprecision(4) << std::dec << inputAttributes[i].attrValues[tempCounter];
        }
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
    for (int i = 0; i < classAttrPossibleValuesCount; i++) {
        Accuracy_Destination << std::setw(8) << std::fixed << std::setprecision(2) << std::dec << inputAttributes[classAttrNum].possibleAttrValues[i] << "    |";
        for (int j = 0; j < classAttrPossibleValuesCount; j++) {
            if (accuracy[i].totalCount != 0)
                temp = ((double)accuracy[i].Classes[j].classCount);// / accuracy[i].totalCount);
            if (i == j) {
                totalAccuracy += ((double)accuracy[i].Classes[j].classCount / accuracy[i].totalCount);
            }
            Accuracy_Destination << std::setw(10) << std::fixed << std::setprecision(2) << std::dec << temp;
        }
        Accuracy_Destination << '\n' << std::setw(sizeof("Actual Class"));
    }
    Accuracy_Destination << '\n';
    for (int i = 0; i < K; i++) {
        Accuracy_Destination << "The acuracy of fold " << i << " is " << std::setw(6) << std::fixed << std::setprecision(5)
            << std::dec << (eachFoldCount[i].correctPredicted / eachFoldCount[i].total) << '\n';
    }
    Accuracy_Destination << '\n' << "The total acuracy is " << std::setw(6) << std::fixed << std::setprecision(5)
        << std::dec << (totalAccuracy) / classAttrPossibleValuesCount << '\n';
}

/* Counts the number of predictions and mispredictions*/
void compareOriginalAndDerivedClasses() {
    // Initialize the vector to holds the correct counts and bad counts
    accuracy.resize(classAttrPossibleValuesCount);
    for (int i = 0; i < classAttrPossibleValuesCount; i++) {
        for (int j = 0; j < classAttrPossibleValuesCount; j++) {
            accuracy[i].Classes.push_back(countClasseValuesForAttr);
            accuracy[i].Classes[j].classCount = 0;
            accuracy[i].Classes[j].classValue = inputAttributes[classAttrNum].possibleAttrValues[j];
        }
        accuracy[i].oneOfAttrValue = inputAttributes[classAttrNum].possibleAttrValues[i];
    }
    // Counting is done here
    for (int num = 0; num < inputAttributes[classAttrNum].attrValues.size(); num++) {
        for (int i = 0; i < classAttrPossibleValuesCount; i++) {
            if (inputAttributes[classAttrNum].attrValues[num] == accuracy[i].oneOfAttrValue) {
                for (int j = 0; j < classAttrPossibleValuesCount; j++) {
                    if (inputAttributes[numOfAttributes - 2].attrValues[num] == accuracy[i].Classes[j].classValue) {
                        accuracy[i].Classes[j].classCount++;
                    }
                }
                accuracy[i].totalCount++;
            }
        }
    }
}

void calculateProbablitity() {
    for (int attr = 0; attr < numOfAttributes - 2; attr++) {
        if (classAttrNum != attr) {
            for (int i = 0; i < inputAttributes[attr].possibleAttrValues.size(); i++) {
                for (int cls = 0; cls < classAttrPossibleValuesCount; cls++) {
                    if (totalClassesCount[cls].classCount != 0) {
                        countClaasesEachAttr[attr].attribute[i].Classes[cls].probability = ((double)countClaasesEachAttr[attr].attribute[i].Classes[cls].classCount / totalClassesCount[cls].classCount);
                    }
                    else {
                        countClaasesEachAttr[attr].attribute[i].Classes[cls].probability = 0;
                    }
                }
            }
        }
    }
}
/* To debug:  */
void printProbability() {
    for (int attr = 0; attr < numOfAttributes - 2; attr++) {
        if (classAttrNum != attr) {
            std::cout << inputAttributes[attr].attrName << '\n';
            for (int i = 0; i < inputAttributes[attr].possibleAttrValues.size(); i++) {
                std::cout << " " << inputAttributes[attr].possibleAttrValues[i] << '\n';
                for (int cls = 0; cls < classAttrPossibleValuesCount; cls++) {
                    std::cout << "  " << countClaasesEachAttr[attr].attribute[i].Classes[cls].classCount << " " << totalClassesCount[cls].classCount
                        << " " << countClaasesEachAttr[attr].attribute[i].Classes[cls].probability << '\n';
                }
            }
        }
    }
}

/* Returns which class the current data belongs to */
double whichClass(int num) {
    double belongsTo;
    double maxProbability = 0;
    // calculate conditional probability
    for (int attr = 0; attr < numOfAttributes - 2; attr++) {
        if (classAttrNum != attr) {
            for (int i = 0; i < inputAttributes[attr].possibleAttrValues.size(); i++) {
                if (countClaasesEachAttr[attr].attribute[i].oneOfAttrValue == inputAttributes[attr].attrValues[num]) {
                    for (int cls = 0; cls < classAttrPossibleValuesCount; cls++) {
                        for (int j = 0; j < classAttrPossibleValuesCount; j++) {
                            if (probabilityVector[j].classValue == countClaasesEachAttr[attr].attribute[i].Classes[cls].classValue) {
                                probabilityVector[j].probability *= (countClaasesEachAttr[attr].attribute[i].Classes[cls].probability);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int cls = 0; cls < classAttrPossibleValuesCount; cls++) {
        if (maxProbability <= probabilityVector[cls].probability) {
            belongsTo = probabilityVector[cls].classValue;
            maxProbability = probabilityVector[cls].probability;
        }
        probabilityVector[cls].probability = 1;
    }
    return belongsTo;
}

/* function to test the input data set after training*/
void doTesting(std::vector<int>&testData, int foldNum) {
    calculateProbablitity();
#ifdef DEBUG
    printProbability();
#endif
    for (int i = 0; i < testData.size(); i++) {
        eachFoldCount[foldNum].total++;
        inputAttributes[numOfAttributes - 2].attrValues[testData[i]] = whichClass(testData[i]);
        if (inputAttributes[numOfAttributes - 2].attrValues[testData[i]] == inputAttributes[classAttrNum].attrValues[testData[i]])
            eachFoldCount[foldNum].correctPredicted++;
    }
#ifdef DEBUG
    std::cout << foldNum << " " << eachFoldCount[foldNum].total << " " << eachFoldCount[foldNum].correctPredicted << '\n';
#endif
}

void countTotalClassesInTrainData(std::vector<int>&trainData, int size) {
    probabilityVector.resize(classAttrPossibleValuesCount);

    for (int j = 0; j < classAttrPossibleValuesCount; j++) {
        totalClassesCount.push_back(countClasseValuesForAttr);
        totalClassesCount[j].classValue = inputAttributes[classAttrNum].possibleAttrValues[j];
        probabilityVector[j].classValue = inputAttributes[classAttrNum].possibleAttrValues[j];
    }
    for (int num = 0; num < size; num++) {
        for (int j = 0; j < classAttrPossibleValuesCount; j++) {
            if (totalClassesCount[j].classValue == inputAttributes[classAttrNum].attrValues[trainData[num]])
                totalClassesCount[j].classCount++;
        }
    }
}

void doTraining(std::vector<int>&trainData) {
    std::vector<unsigned long int> tempCounter;

    countTotalClassesInTrainData(trainData, trainData.size());

    // Does all the atrributes are used?
    for (int attr = 0; attr < numOfAttributes - 2; attr++) {
        countClaasesEachAttr.push_back(temp_remainingAttr);
        countClaasesEachAttr[attr].nameOfAttr = inputAttributes[attr].attrName;
        if (classAttrNum != attr) {
            for (int count = 0; count < inputAttributes[attr].possibleAttrValues.size(); count++) {
                countClaasesEachAttr[attr].attribute.push_back(temp_countInfoOfAttr);
                countClaasesEachAttr[attr].attribute[count].oneOfAttrValue = inputAttributes[attr].possibleAttrValues[count];
                for (int j = 0; j < classAttrPossibleValuesCount; j++) {
                    countClaasesEachAttr[attr].attribute[count].Classes.push_back(countClasseValuesForAttr);
                    countClaasesEachAttr[attr].attribute[count].Classes[j].classValue = inputAttributes[classAttrNum].possibleAttrValues[j];
                }
                for (int num = 0; num < trainData.size(); num++) {
                    if (inputAttributes[attr].attrValues[trainData[num]] == inputAttributes[attr].possibleAttrValues[count]) {
                        countClaasesEachAttr[attr].attribute[count].totalCount++;
                        for (int j = 0; j < classAttrPossibleValuesCount; j++) {
                            if (countClaasesEachAttr[attr].attribute[count].Classes[j].classValue == inputAttributes[classAttrNum].attrValues[trainData[num]]) {
                                countClaasesEachAttr[attr].attribute[count].Classes[j].classCount++;
                            }
                        }
                    }
                }
            }
        }
    }
}

/* Opens the input and output files */
void openFile() {
    source.open(inputFileName.c_str());
    if (!source)
        std::cerr << "Error: Unable to open the input file" << '\n';
    if (!userOutputFileName) {
        Apply_OutFileName = "naiveBayesian_Output_" + std::to_string(K) + "FoldClassification_" + inputFileName;
        Accuracy_OutFileName = "naiveBayesian_Output_" + std::to_string(K) + "FoldConfusion_" + inputFileName;
    }
    else {
        Apply_OutFileName = outputFileName + "_FoldClassification_" + inputFileName;
        Accuracy_OutFileName = outputFileName + std::to_string(K) + "_FoldConfusion_" + inputFileName;
    }

    Apply_Destination.open(Apply_OutFileName.c_str());
    if (!Accuracy_Destination)
        std::cerr << "Error: Unable to open one of the output file " << Apply_OutFileName.c_str() << '\n';
    Accuracy_Destination.open(Accuracy_OutFileName.c_str());
    if (!Apply_Destination)
        std::cerr << "Error: Unable to open one of the output file" << Accuracy_OutFileName.c_str() << '\n';
}

/* Closes the input and output files */
void closeFile() {
    source.close();
    Apply_Destination.close();
    Accuracy_Destination.close();
}

/* Main Program */
int main(int argc, char *argv[]) {
    std::string currentLine;
    std::string isAttribute = "attribute";
    std::string holdAttrName = " ";
    std::string spaces = "";
    double readValue = 0.0;
    int tempCounter = 0;
    std::vector<int> trainData;
    std::vector<int> testData;

    readCommandLine(argc, argv);
    openFile();
    eachFoldCount.resize(K);
    /* Handels the lines which are in the beginning of the input file */
    getline(source, currentLine);
    Apply_Destination << currentLine << '\n' << '\n';

    while (source.good()) {
        if (currentLine.length() != 0) {        // discards the empty lines
            std::istringstream ss(currentLine);
            if (currentLine.at(0) == '@') {
                ss.ignore();
                ss >> holdAttrName;
                if (holdAttrName == isAttribute) {  // get the attribute names
                    ss >> holdAttrName;
                    inputAttributes.push_back(temp_eachAttr);
                    inputAttributes[numOfAttributes].attrName = holdAttrName;
                    ss.ignore(10, '{');
                    while (ss >> readValue) {
                        inputAttributes[numOfAttributes].possibleAttrValues.push_back(readValue);
                        if (ss.peek() == ',')
                            ss.ignore();
                    }
                    if (inputAttributes[numOfAttributes].attrName == classAttr)
                        classAttrNum = numOfAttributes;
                    numOfAttributes++;
                }
                // if the data is going to start end this loop
                if (holdAttrName == "data") {
                    inputAttributes.push_back(temp_eachAttr);
                    inputAttributes[numOfAttributes].attrName = "bayesClass";
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
    inputAttributes.push_back(temp_eachAttr);
    inputAttributes[numOfAttributes].attrName = "fold";
    numOfAttributes++;
    /* Reads the data from input file and stores in a vector */
    getline(source, currentLine);
    while (source.good()) {
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
    eachAttributeLength = inputAttributes[0].attrValues.size();

    // initialize the new attribute to hold derived class
    for (int i = 0; i < inputAttributes[0].attrValues.size(); i++) {
        inputAttributes[numOfAttributes - 2].attrValues.push_back(0);
    }
    folds.resize(classAttrPossibleValuesCount);
    for (int i = 0; i < classAttrPossibleValuesCount; i++) {
        folds[i].classValue = inputAttributes[classAttrNum].possibleAttrValues[i];
        folds[i].classCount = 0;
    }
    // get the last index of the training data set
    for (int i = 0; i < eachAttributeLength; i++) {
        for (int j = 0; j < classAttrPossibleValuesCount; j++) {
            if (folds[j].classValue == inputAttributes[numOfAttributes - 3].attrValues[i]) {
                inputAttributes[numOfAttributes - 1].attrValues.push_back(folds[j].classCount%K);
                folds[j].classCount++;
            }
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < eachAttributeLength; j++) {
            if (inputAttributes[numOfAttributes - 1].attrValues[j] != i)
                trainData.push_back(j);
            else
                testData.push_back(j);
        }
        // Train
        doTraining(trainData);
        // test
        doTesting(testData, i);
        // Clear the buffers for next fold
        trainData.erase(trainData.begin(), trainData.end());
        testData.erase(testData.begin(), testData.end());
        totalClassesCount.erase(totalClassesCount.begin(), totalClassesCount.end());
        probabilityVector.erase(probabilityVector.begin(), probabilityVector.end());
        countClaasesEachAttr.erase(countClaasesEachAttr.begin(), countClaasesEachAttr.end());
    }

    compareOriginalAndDerivedClasses();
    // write the entire input data along with new derived class
    writeAttrToFile();
    // write the confusion matrix
    writeAccuracyToFile();
    // Close all the input and output files
    closeFile();

    return 0;
}
