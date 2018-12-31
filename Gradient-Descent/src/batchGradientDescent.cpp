/*************************************************************************************
* Calculates the parameters of the linear equation (y=theta0+theta1*x0+...+thetan*xn-1)
*	by using batch grandient descent algorithm.
**************************************************************************************/
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <time.h>
#include <algorithm>

// Uncomment below line to print debug data while executing
// #define DEBUG

std::string inputFileName;
std::string outputFileName;
int numberOfVariables = 1;
double learningRate = 0.0005;
double epsilon = 0.0005;
long int iterations = 100000;
bool randomInitialParameterValues = true;
bool userOutputFileName = false;

std::string help_msg = "-h or --help\n"
"  To print this message\n"
"-i [inputFileName] or --input [inputFileName]\n"
"  To specify intput file name\n"
"-o [outputFileName] or --output [outputFileName]\n"
"  To specify output file name, default - gradientDecent_Output_[time].txt\n"
"-r [true/false] or --random [true/false] \n"
"  To specify whether to initialize the variables to random values or 0, default - true\n"
"-v [numberOfVariables] or --variables [numberOfVariables]\n"
"  To specify number of variables in a liner equation, default - 1\n"
"-l [learningRate] or --learningrate [learningRate]\n"
"  To specify learning rate in the range of [0, 1), default - 0.0005\n"
"-e [epsilon] or --epsilon [epsilon]\n"
"  Algorithm terminates when the improvement of two consecutive iterations is\n"
"  not more than epsilon, default - 0.0005; see also option -k\n"
"-k [iterations] or --iterations [iterations]\n"
"  Maximum number of iterartions to run the algorithm, default - 100,000\n"
"NOTE: Algorithm terminates if any of the condition specified by the option -k or -e is met";

template <class T>
class sample {
public:
	std::vector<T> input;
	T actualOutput;
};

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
		else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--variables") == 0) {
			std::istringstream ss(argv[i + 1]);
			ss >> numberOfVariables;
			i++;
		}
		else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--learningRate") == 0) {
			std::istringstream ss(argv[i + 1]);
			ss >> learningRate;
			i++;
		}
		else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--epsilon") == 0) {
			std::istringstream ss(argv[i + 1]);
			ss >> epsilon;
			i++;
		}
		else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--random") == 0) {
			randomInitialParameterValues = (strcmp(argv[i + 1], "true") == 0 || strcmp(argv[i + 1], "True") == 0) ? true : false;
			i++;
		}
		else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--iterations") == 0) {
			std::istringstream ss(argv[i + 1]);
			ss >> iterations;
			i++;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			std::cout << help_msg << std::endl;
			exit(1);
		}
		else {
			std::cout << "Error: Unknown parameter" << std::endl;
			std::cout << help_msg << std::endl;
			exit(1);
		}
	}
}

template <class T = float>
void readInputFile(std::vector<sample<T> >& inputSampleData, std::ifstream& source) {
	std::string currentLine;
	T readValue;
	getline(source, currentLine);
	while (source.good()) {
#ifdef DEBUG
		std::cout << currentLine << '\n';
#endif
		int i = 0;
		sample<T> tempSample;
		// Process nonempty lines
		if (currentLine.length() != 0) {
			std::istringstream ss(currentLine);
			tempSample.input.reserve(numberOfVariables);
			// Extract data from the current line
			while (i < numberOfVariables) {
				ss >> readValue;
				tempSample.input.push_back(readValue);
				// If the delimiter is ',' or '|' ignore it
				if (ss.peek() == ',' || ss.peek() == '|') {
					ss.ignore();
				}
				i++;
			}
			// Read the actual output
			ss >> readValue;
			tempSample.actualOutput = readValue;
		}
		inputSampleData.push_back(tempSample);
		getline(source, currentLine);
	}
}

void initializeParameters(double* theta) {
	if (randomInitialParameterValues) {
		// Get the random numbers in the range of 0 to 1
		for (int i = 0; i < numberOfVariables + 1; i++) {
			theta[i] = ((double)rand() / (RAND_MAX)) + 1;
		}
	}
	else {
		for (int i = 0; i < numberOfVariables + 1; i++) {
			theta[i] = 0;
		}
	}
}

// Calculate output = theta[0]+x0*theta[1]+x1*theta[2]+.....+(xn-1)*theta[n]
template <class T = float>
double calculateOutput(std::vector<sample<T> >& inputSampleData, double* theta, int sampleNum) {
	double cal_output = theta[0];
	for (int j = 0; j < numberOfVariables; j++) {
		cal_output += theta[j + 1] * inputSampleData[sampleNum].input[j];
	}
	return cal_output;
}

// Print actual output and calculated output
template <class T = float>
void verify(std::vector<sample<T> >& inputSampleData, std::ofstream& destination, double* theta) {
	destination << "\nActual Output, Calculated Output" << '\n';
	for (int i = 0; i < inputSampleData.size(); i++) {
		double cal_output = calculateOutput<T>(inputSampleData, theta, i);
		destination << std::setprecision(4) << inputSampleData[i].actualOutput << ", " << cal_output << '\n';
	}
	destination << '\n';
}

// Calculate cost = ((actual output - calculated output)^2) / 2*m
template <class T = float>
double costFunction(std::vector<sample<T> >& inputSampleData, double* theta) {
	long int m = inputSampleData.size();
	double cost = 0;
	for (long int i = 0; i < m; i++) {
		double cal_output = calculateOutput<T>(inputSampleData, theta, i);
		cost += (cal_output - inputSampleData[i].actualOutput)*(cal_output - inputSampleData[i].actualOutput);
	}
	return cost / (2 * m);
}

// Calculate gradient of theta[0]...theta[n]
template <class T = float>
void gradiantCalculation(std::vector<sample<T> >& inputSampleData, double *gradiantWithRespectToTheta, double* theta) {
	long int m = inputSampleData.size();
	for (int i = 0; i < m; i++) {
		double cal_output = calculateOutput<T>(inputSampleData, theta, i);
		double temp = cal_output - inputSampleData[i].actualOutput;
		gradiantWithRespectToTheta[0] += (1.0 / m) * (temp);
		for (int j = 0; j < numberOfVariables; j++) {
			gradiantWithRespectToTheta[j + 1] += (1.0 / m) * inputSampleData[i].input[j] * (temp);
		}
	}
}

// Actual Gradient Descent algorithm
template <class T = float>
void gradientDescent(std::vector<sample<T> >& inputSampleData, std::ofstream& destination, double learningRate, double epsilon, double* theta, long int iterartions) {
	double new_cost = std::numeric_limits<double>::max();
	double previous_cost = 0;
	double min_cost = 0;
	double* bestTheta = new double[numberOfVariables + 1];
	double* gradiantWithRespectToTheta = new double[numberOfVariables + 1];
	long int i = 0;
	// Calculate cost 
	new_cost = costFunction<T>(inputSampleData, theta);
	min_cost = new_cost;
	// Keep a copy of best theta
	memcpy(bestTheta, theta, sizeof(double)*(numberOfVariables + 1));

#ifdef DEBUG
	for (int j = 0; j < numberOfVariables + 1; j++) {
		destination << std::setprecision(6) << theta[j] << ", ";
	}
	destination << std::setprecision(6) << new_cost << '\n';
#endif // DEBUG

	while ((i < iterartions) && (fabs(new_cost - previous_cost) > epsilon)) {
		// Reset values to zero
		memset(gradiantWithRespectToTheta, 0, sizeof(double)*(numberOfVariables + 1));
		previous_cost = new_cost;
		// Caculate the gradient of all the learning coefficients
		gradiantCalculation<T>(inputSampleData, gradiantWithRespectToTheta, theta);
		// Update the coefficients
		for (int j = 0; j < numberOfVariables + 1; j++) {
			theta[j] -= learningRate * gradiantWithRespectToTheta[j];
		}
		// Calculate the new cost with respect to updated coefficients
		new_cost = costFunction<T>(inputSampleData, theta);
#ifdef DEBUG
		for (int j = 0; j < numberOfVariables + 1; j++) {
			destination << std::setprecision(6) << theta[j] << ", ";
		}
		destination << std::setprecision(6) << new_cost << '\n';
#endif // DEBUG

		if (min_cost > new_cost) {
			min_cost = new_cost;
			memcpy(bestTheta, theta, sizeof(double)*(numberOfVariables + 1));
		}
		i++;
	}
	// Log the final output
	destination << "\n\nMinimum cost: " << min_cost << '\n';
	destination << "\nLinear equation: ";
	destination << "Y =  " << std::fixed << std::setprecision(6) << bestTheta[0];
	for (int j = 1; j < numberOfVariables + 1; j++) {
		destination << " + " << std::setprecision(6) << bestTheta[j] << " x" << (j - 1);
	}
	destination << std::endl;

	verify<T>(inputSampleData, destination, bestTheta);

	// Freeup allocated memory
	delete[] bestTheta;
	delete[] gradiantWithRespectToTheta;
}

int main(int argc, char *argv[]) {
	// Initialize varibales
	std::ifstream source;
	std::ofstream destination;
	std::vector<sample<double> > inputSampleData;
	srand(time(NULL));
	if (argc == 1) {
		std::cerr << "Use ./gradientDescent --help for more options" << '\n';
		return -1;
	}
	// Read command line arguments
	readCommandLine(argc, argv);

	double* theta = new double[numberOfVariables + 1];
	if (!userOutputFileName) {
		time_t t = std::time(0);
		struct tm* stamp = localtime(&t);
		std::ostringstream ss;
		ss << static_cast<long long>(stamp->tm_year + 1900) << "_" << static_cast<long long>(stamp->tm_mon + 1) << "_"
			<< static_cast<long long>(stamp->tm_mday) << "_" << static_cast<long long>(stamp->tm_hour) << "."
			<< static_cast<long long>(stamp->tm_min) << "." << static_cast<long long>(stamp->tm_sec);
		outputFileName = "gradientDecent_Output_" + ss.str() + ".txt";
	}
	// Open input file
	source.open(inputFileName.c_str());
	if (!source) {
		std::cerr << "Error: Unable to open the input file" << '\n';
		return -1;
	}
	// Open output file
	destination.open(outputFileName.c_str());
	if (!destination) {
		std::cerr << "Error: Unable to open the output file" << '\n';
		return -1;
	}
	// Read input data
	readInputFile<double>(inputSampleData, source);
	source.close();
	initializeParameters(theta);

	// Log the parameters to outut file for reference
	destination << "Learning Rate: " << learningRate << std::endl;
	destination << "Number Of Variables: " << numberOfVariables << std::endl;
	destination << "Number Of Max Iterations: " << iterations << std::endl;
	destination << "Epsilon: " << epsilon << std::endl;
	destination << "Initial Theta Values: ";
	for (int i = 0; i < numberOfVariables + 1; i++) {
		destination << std::setprecision(6) << theta[i] << ", ";
	}
	destination << std::endl << std::endl;

#ifdef DEBUG
	for (int j = 0; j < numberOfVariables + 1; j++) {
		destination << "Theta" << (j) << ", ";
	}
	destination << "Cost " << std::endl;
#endif // DEBUG

	gradientDescent<double>(inputSampleData, destination, learningRate, epsilon, theta, iterations);
	destination.close();

	delete[] theta;
	return 0;
}
