/*************************************************************************************
* Calculates the parameters of the linear equation (y=theta0+theta1*x0+...+thetan*xn-1)
*	by using stochastic grandient descent algorithm. In stochastic gradient descent
*   algorithm parameters are updated for every one input sample.
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
bool normalize = false;
std::string whichNorm = "mean";

std::string help_msg = "-h or --help\n"
"  To print this message\n"
"-i [inputFileName] or --input [inputFileName]\n"
"  To specify intput file name\n"
"-o [outputFileName] or --output [outputFileName]\n"
"  To specify output file name, default - stochasticGradientDecent_Output_[time].txt\n"
"-r [true/false] or --random [true/false] \n"
"  To specify whether to initialize the variables to random values or 0, default - true\n"
"-n [true/false] or --normalize [true/false] \n"
"  To specify whether to normalize the inputs or not, default - false\n"
"-w [minmax/mean/standard] or --whichnorm [minmax/mean/standard]\n"
"  Specify the type of normalization for the input samples, default - mean\n"
"  minmax: min-max normalization x'= (x-min(x))/(max(x)-min(x))\n"
"  mean: mean normalization x'= (x-mean(x))/(max(x)-min(x))\n"
"  standard: standardization x'= (x-mean(x))/standardDeviation\n"
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

template <class T>
class compare {
public:
	compare(int col) : m_column(col) {}
	bool operator()(const sample<T>  i, const sample<T>  j) {
		return i.input.at(m_column) < j.input.at(m_column);
	}
private:
	int m_column;
};


void readCommandLine(int argc, char *argv[]) {
	bool isUserSetWhichNorm = false;
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
			inputFileName = argv[i + 1];
			i++;
		} else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
			outputFileName = argv[i + 1];
			userOutputFileName = true;
			i++;
		} else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--variables") == 0) {
			std::istringstream ss(argv[i + 1]);
			ss >> numberOfVariables;
			i++;
		} else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--learningrate") == 0) {
			std::istringstream ss(argv[i + 1]);
			ss >> learningRate;
			i++;
		} else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--epsilon") == 0) {
			std::istringstream ss(argv[i + 1]);
			ss >> epsilon;
			i++;
		} else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--random") == 0) {
			randomInitialParameterValues = (strcmp(argv[i + 1], "true") == 0 || strcmp(argv[i + 1], "True") == 0) ? true : false;
			i++;
		} else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--normalize") == 0) {
			normalize = (strcmp(argv[i + 1], "true") == 0 || strcmp(argv[i + 1], "True") == 0) ? true : false;
			i++;
		} else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--whichnorm") == 0) {
			whichNorm = argv[i + 1];
			i++;
			isUserSetWhichNorm = true;
		} else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--iterations") == 0) {
			std::istringstream ss(argv[i + 1]);
			ss >> iterations;
			i++;
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			std::cout << help_msg << std::endl;
			exit(1);
		} else {
			std::cerr << "Error: Unknown parameter " << argv[i] << std::endl;
			std::cerr << help_msg << std::endl;
			exit(1);
		}
	}
	if (isUserSetWhichNorm &&(!normalize)) {
		std::cerr << "Error: Turn on normalization to specify type of normalization " << std::endl;
		std::cerr << help_msg << std::endl;
		exit(1);
	}
}

template <class T = float>
void readInputFile(std::vector<sample<T> >& inputSampleData, std::ifstream& source, std::ofstream& destination) {
	sample<T> tempSample;
	std::string currentLine;
	T readValue;
	// Used to normalize the input samples
	sample<T> s_min;
	sample<T> s_max;
	sample<double> s_mean;
	sample<double> s_standDev;
	if (normalize) {
		// Resize the variables depending on user input
		s_max.input.resize(numberOfVariables);
		s_min.input.resize(numberOfVariables);
		s_mean.input.resize(numberOfVariables);
		s_standDev.input.resize(numberOfVariables);
		// Initialize variables
		for (int i = 0; i < numberOfVariables; i++) {
			s_max.input[i] = std::numeric_limits<T>::min();
			s_min.input[i] = std::numeric_limits<T>::max();
			s_mean.input[i] = 0;
			s_standDev.input[i] = 0;
		}
	}

	tempSample.input.resize(numberOfVariables);
	getline(source, currentLine);
	while (source.good()) {
		int i = 0;
		// Process nonempty lines
		if (currentLine.length() != 0) {
			std::istringstream ss(currentLine);
			// Extract data from the current line
			while (i < numberOfVariables) {
				ss >> readValue;
				tempSample.input[i] = readValue;
				// Get the required data to normalize
				if (normalize) {
					if (readValue > s_max.input[i]) {
						s_max.input[i] = readValue;
					} 
					if (readValue < s_min.input[i]) {
						s_min.input[i] = readValue;
					}
					s_mean.input[i] += readValue;
				}
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

	if (normalize) {
		if (whichNorm == "minmax") {  // Min - max normalization
			for (int i = 0; i < inputSampleData.size(); i++) {
				for (int j = 0; j < numberOfVariables; j++) {
					inputSampleData[i].input[j] = (inputSampleData[i].input[j] - s_min.input[j]) / (s_max.input[j] - s_min.input[j]);
				}
			}
		} else if (whichNorm == "mean") {  // Mean normalization
			// Calculate the mean
			for (int i = 0; i < numberOfVariables; i++) {
				s_mean.input[i] = s_mean.input[i] / inputSampleData.size();
			}
			for (int i = 0; i < inputSampleData.size(); i++) {
				for (int j = 0; j < numberOfVariables; j++) {
					inputSampleData[i].input[j] = (inputSampleData[i].input[j] - s_mean.input[j]) / (s_max.input[j] - s_min.input[j]);
				}
			}
		} else if( whichNorm == "standard") { // Standardization
			// Calculate the mean 
			for (int i = 0; i < numberOfVariables; i++) {
				s_mean.input[i] = s_mean.input[i] / inputSampleData.size();
			}
			// Calculate sum of square of errors
			for (int i = 0; i < inputSampleData.size(); i++) {
				for (int j = 0; j < numberOfVariables; j++) {
					s_standDev.input[j] += ((inputSampleData[i].input[j] - s_mean.input[j])*(inputSampleData[i].input[j] - s_mean.input[j]));
				}
			}
			// Calculate standard deviation for each variable
			for (int j = 0; j < numberOfVariables; j++) {
				s_standDev.input[j] += sqrt(s_standDev.input[j] / (inputSampleData.size() - 1));
			}
			// Standardize the input samples
			for (int i = 0; i < inputSampleData.size(); i++) {
				for (int j = 0; j < numberOfVariables; j++) {
					inputSampleData[i].input[j] = (inputSampleData[i].input[j] - s_mean.input[j]) / s_standDev.input[j];
				}
			}
		}

		destination << "Normalized inputs " << std::endl;
		for (int i = 0; i < inputSampleData.size(); i++) {
			for (int j = 0; j < numberOfVariables; j++) {
				destination << std::setprecision(4) << inputSampleData[i].input[j] << ", ";
			}
			destination << inputSampleData[i].actualOutput << std::endl;
		}
		destination << std::endl;
	}
}

void initializeParameters(double* theta) {
	if (randomInitialParameterValues) {
		// Get the random numbers in the range of 0 to 1
		for (int i = 0; i < numberOfVariables + 1; i++) {
			theta[i] = ((double)rand() / (RAND_MAX)) + 1;
		}
	} else {
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
	destination << "\nActual Output, Calculated Output" << std::endl;
	for (int i = 0; i < inputSampleData.size(); i++) {
		double cal_output = calculateOutput<T>(inputSampleData, theta, i);
		destination << std::setprecision(4) << inputSampleData[i].actualOutput << ", " << cal_output << std::endl;
	}
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
void gradiantCalculation(std::vector<sample<T> >& inputSampleData, double *gradiantWithRespectToTheta, double* theta, long int sampleNum) {
	double cal_output = calculateOutput<T>(inputSampleData, theta, sampleNum);
	double temp = cal_output - inputSampleData[sampleNum].actualOutput;
	gradiantWithRespectToTheta[0] += temp;
	for (int j = 0; j < numberOfVariables; j++) {
		gradiantWithRespectToTheta[j + 1] += inputSampleData[sampleNum].input[j] * (temp);
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
	destination << std::setprecision(6) << new_cost << std::endl;
#endif // DEBUG
    // Using simple termination conditions
	while ((i < iterartions) && (fabs(new_cost - previous_cost) > epsilon)) {
        // Randomly shuffle input data
        std::random_shuffle(inputSampleData.begin(), inputSampleData.end());
        previous_cost = new_cost;
        for (int i = 0; i < inputSampleData.size(); i++) {
            // Reset values to zero
            memset(gradiantWithRespectToTheta, 0, sizeof(double)*(numberOfVariables + 1));
            // Caculate the gradient of all the learning coefficients
            gradiantCalculation<T>(inputSampleData, gradiantWithRespectToTheta, theta, i);
            // Update the coefficients
            for (int j = 0; j < numberOfVariables + 1; j++) {
                theta[j] -= learningRate * gradiantWithRespectToTheta[j];
            }
        }
		// Calculate the new cost with respect to updated coefficients; use validation data set if available
		new_cost = costFunction<T>(inputSampleData, theta);
#ifdef DEBUG
		for (int j = 0; j < numberOfVariables + 1; j++) {
			destination << std::setprecision(6) << theta[j] << ", ";
		}
		destination << std::setprecision(6) << new_cost << std::endl;
#endif // DEBUG

		if (min_cost > new_cost) {
			min_cost = new_cost;
			memcpy(bestTheta, theta, sizeof(double)*(numberOfVariables + 1));
		}
		i++;
	}
	// Log the final output
	destination << "\nMinimum cost: " << min_cost << std::endl;
	destination << "\nLinear equation: ";
	destination << "Y =  " << std::fixed << std::setprecision(6) << bestTheta[0] << " ";
	for (int j = 1; j < numberOfVariables + 1; j++) {
		//destination << (bestTheta[j] > 0 ? " + " : " - ");
		destination << std::setprecision(6) << std::showpos << bestTheta[j] << " x" << std::noshowpos << (j - 1) << " ";
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
		std::cerr << "Use ./stochasticGradientDescent --help for more options" << std::endl;
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
		outputFileName = "stochasticGradientDecent_Output_" + ss.str() + ".txt";
	}
	// Open input file
	source.open(inputFileName.c_str());
	if (!source) {
		std::cerr << "Error: Unable to open the input file" << std::endl;
		return -1;
	}
	// Open output file
	destination.open(outputFileName.c_str());
	if (!destination) {
		std::cerr << "Error: Unable to open the output file" << std::endl;
		return -1;
	}
	// Read input data
	readInputFile<double>(inputSampleData, source, destination);
	source.close();
	initializeParameters(theta);

	// Log the parameters to outut file for reference
	destination << "Learning Rate: " << learningRate << std::endl;
	destination << "Number Of Variables: " << numberOfVariables << std::endl;
	destination << "Number Of Max Iterations: " << iterations << std::endl;
	destination << "Epsilon: " << epsilon << std::endl;
	destination << "Normalize: " << std::boolalpha << normalize << std::noboolalpha << std::endl;
	if (normalize) {
		destination << "Which Normalization: " << whichNorm << std::endl;
	}
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
