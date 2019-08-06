/*********************************************************************************************
 This program trains and tests the back propagaation neural network by reading input, expected 
 output, and test cases from the input files. The number of hidden layers and other 
 parameters can be modified according to the need.
 To compile: g++ Multilayer_BPNN.cpp -o Multilayer_BPNN -std=c++11
 To run: ./Multilayer_BPNN input_file expected_output_file test_cases_file
**********************************************************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <math.h>
#include <time.h>

//#define DEBUG

#define sigmoid(x) (1.0 / (1.0 + (double)exp(-(double)(x))))
#define sigmoidDerivative(x) ((double)(x)*(1.0-(x))) 
#define rand_05_to_05() (((double)rand()/RAND_MAX)-0.5)
#define random() ((double)rand()/RAND_MAX)
#define alpha 0.01

#define ITERATIONS 1000000
#define INPUT_UNITS 3
#define HIDDEN_LAYERS 2
#define OUTPUT_UNITS 1
#define BIAS_VALUE 1

// Includes input and output layers
#define TOTAL_LAYERS HIDDEN_LAYERS+2

// Initialize with number of units in each hidden layer
int ALL_LAYER_UNITS[TOTAL_LAYERS] = { INPUT_UNITS, 3, 2, OUTPUT_UNITS };

struct eachNeuron {
	// Connected to other units in the next layer
	std::vector <double> weights_to_nextlayer_neurons;	
	std::vector <double> delta_weights_to_nextlayer_neurons;
	double output;
	double delta_output;
	double error;
} eachUnit;

// Just store only final weights of the trained network
struct eachLayer {
	std::vector <eachNeuron> neurons;
}layer;

//To store final trained network parameters
struct eachNeuron_wts {
    // Connected to other units in the next layer
    std::vector <double> weights_to_nextlayer_neurons;
};
struct eachLayer_wts {
    std::vector <eachNeuron_wts> neurons;
};

std::ofstream weights_output_file;
bool momentumOn = false;
double momentum = 0.5;
double sigma = 1;
double current_error = std::numeric_limits<double>::max();

// Open the input file
void openInputFile(std::ifstream& in_file, std::string filename) {
    in_file.open(filename.c_str());
    if (!in_file) {
        std::cerr << "Error: Cannot open the file " << filename << std::endl;
        exit(1);
    }
}

// Open the input file
void openOutputFile(std::ofstream& in_file, std::string filename) {
    in_file.open(filename.c_str());
    if (!in_file) {
        std::cerr << "Error: Cannot open the file " << filename << std::endl;
        exit(1);
    }
}

// Open and read the input files
void readInputFile(std::ifstream& in_file, std::string filename, std::vector<std::vector<double>>& v) {
    openInputFile(in_file, filename);
    double value = 0;
    std::string current_line = "";
    getline(in_file, current_line);
    while (in_file.good()) {
        if (current_line.length() != 0) {
            std::istringstream ss(current_line);
            std::vector<double> temp;
            // Extracts the numbers from the current line
            while (ss >> value) {
                // Save the numbers to a vector
                temp.push_back(value);
                // If the delimiter is ',' ignore that character 
                if (ss.peek() == ',') {
                    ss.ignore();
                }
            }
            v.push_back(temp);
        }
        getline(in_file, current_line);
    }
}

// Write trained network parameters to output file
void writeWeightsToFile(std::vector<eachLayer_wts>& final_weights) {
    // Write parameters to weights file
    weights_output_file << "Number of layers: " << TOTAL_LAYERS << std::endl;
    weights_output_file << "Number of neurons in each layer (excluding bias): ";
    for (int i = 0; i < TOTAL_LAYERS; i++) {
        weights_output_file << ALL_LAYER_UNITS[i] << " ";
    }
    weights_output_file << std::endl;
    weights_output_file << "Current Error: " << current_error << std::endl;

    // Final weights of the neural network
    for (int m = 0; m < TOTAL_LAYERS - 1; m++) {
        for (int i = 0; i < ALL_LAYER_UNITS[m]; i++) {
            for (int j = 0; j < ALL_LAYER_UNITS[m + 1]; j++) {
                weights_output_file << "Layer " << m << " Unit " << i << " to unit " << j << " " << final_weights[m].neurons[i].weights_to_nextlayer_neurons[j] << std::endl;
            }
        }
    }
}

// Gives the unique random number
double uniqueRandomNumber(std::unordered_set <double>& used_random_numbers) {
    double random_number = rand_05_to_05();
    // Get the unique random numbers between -0.5 to 0.5 to initialize weights
    while (used_random_numbers.count(random_number) > 0) {
        random_number = rand_05_to_05();
    }
    used_random_numbers.insert(random_number);
    return random_number;
}

// Initialize the weights of the nueral network with unique random numbers
void InitializeWeights(std::vector<eachLayer>& layers) {
    std::unordered_set <double> used_random_numbers;
#ifdef DEBUG
    std::cout << "Initializing weights for neurons" << std::endl;
#endif	
    // Initialize weights for all the layers neurons
    for (int m = 0; m < TOTAL_LAYERS - 1; m++) {
        // Last neuron in a given layer is bias
        layers[m].neurons.resize(ALL_LAYER_UNITS[m]+1);
        // Go through all the neurons in a given layer
        for (int i = 0; i < ALL_LAYER_UNITS[m]+1; i++) {
            // Each neuron is connected to all the neuron in the next layer
            layers[m].neurons[i].weights_to_nextlayer_neurons.resize(ALL_LAYER_UNITS[m + 1]);
            layers[m].neurons[i].delta_weights_to_nextlayer_neurons.resize(ALL_LAYER_UNITS[m + 1]);
            for (int j = 0; j < ALL_LAYER_UNITS[m + 1]; j++) {
                layers[m].neurons[i].weights_to_nextlayer_neurons[j] = uniqueRandomNumber(used_random_numbers);
            }
        }
    }
}

// Keep a copy of final weights of the network for testing purpose
void storeFinalWeights(std::vector<eachLayer>& layers, std::vector<eachLayer_wts>& final_weights) {
    // Final weights of the neural network
    for (int m = 0; m < TOTAL_LAYERS - 1; m++) {
        final_weights[m].neurons.resize(ALL_LAYER_UNITS[m] + 1);
        for (int i = 0; i < ALL_LAYER_UNITS[m]; i++) {
            final_weights[m].neurons[i].weights_to_nextlayer_neurons.resize(ALL_LAYER_UNITS[m + 1]);
            for (int j = 0; j < ALL_LAYER_UNITS[m + 1]; j++) {
                final_weights[m].neurons[i].weights_to_nextlayer_neurons[j] = layers[m].neurons[i].weights_to_nextlayer_neurons[j];
#ifdef DEBUG
                std::cout << "Layer " << m << " Unit " << i << " to unit " << j << " " << final_weights[m].neurons[i].weights_to_nextlayer_neurons[j] << std::endl;
#endif
            }
        }
    }
}

// Updates the weights depending on the error
void updateTheWeights(std::vector<eachLayer>& layers, std::vector<eachLayer>& last_iteration) {
#ifdef DEBUG
    std::cout << "Update the weights" << std::endl;
#endif	
	double temp = 0;
	for (int m = 0; m < TOTAL_LAYERS-1; m++) {
        // Update the weights
		for (int i = 0; i < layers[m].neurons.size(); i++) {
			for (int j = 0; j < layers[m].neurons[i].weights_to_nextlayer_neurons.size(); j++) {
				if (momentumOn){
					temp = momentum*(last_iteration[m].neurons[i].delta_weights_to_nextlayer_neurons[j]);
				}
				layers[m].neurons[i].weights_to_nextlayer_neurons[j] += layers[m].neurons[i].delta_weights_to_nextlayer_neurons[j] + temp;
			}
		}
	}
}

// Calculate the output depending on the weights
void forwardCalculation(std::vector<eachLayer>& layers) {
#ifdef DEBUG
    std::cout << "Forward calculation" << std::endl;
#endif	
    for (int m = 1; m < TOTAL_LAYERS; m++) {
        // Last neuron in a given layer is always bias
        layers[m].neurons.resize(ALL_LAYER_UNITS[m] + 1);
        for (int i = 0; i < ALL_LAYER_UNITS[m] + 1; i++) {
            // Each neuron is connected to all the neurons in the next layer
            for (int j = 0; j < ALL_LAYER_UNITS[m - 1]; j++) {
                // Get the weighted sum
                layers[m].neurons[i].output += layers[m - 1].neurons[j].weights_to_nextlayer_neurons[i] * layers[m - 1].neurons[j].output;
            }
            layers[m].neurons[i].output = sigmoid(layers[m].neurons[i].output);
        }
    }
}

// Propogate the error backward
void backwardCalculations(std::vector<eachLayer>& layers, std::vector<std::vector<double>>& expected_outputs, int item) {
#ifdef DEBUG
    std::cout << "Backward calculation" << std::endl;
    std::cout << "Calculating delta of output neurons..." << std::endl;
#endif
	// Calculate the output deltas
	int m = layers.size()-1;
	for (int i = 0; i < OUTPUT_UNITS; i++) {
		// Calculate the error
		layers[m].neurons[i].error = expected_outputs[item][i]-layers[m].neurons[i].output;
		// delta = (expected-actual)*out*(1-out)
		layers[m].neurons[i].delta_output = layers[m].neurons[i].error*sigmoidDerivative(layers[m].neurons[i].output);
		// Total error
		current_error += layers[m].neurons[i].error*layers[m].neurons[i].error;
	}
#ifdef DEBUG
    std::cout << "Calculating hidden layer(s) neurons delta..." << std::endl;
#endif
	// Calculate other layer neurons delta
	for(int m = layers.size()-2; m >= 0; m--){
		for (int i = 0; i < ALL_LAYER_UNITS[m]; i++) {
			// delta = sum of (delta of each neuron in next layer*weight to that neuron)*out*(1-out)
			layers[m].neurons[i].error = 0;
			for (int k = 0; k < ALL_LAYER_UNITS[m+1]; k++) {
				layers[m].neurons[i].error += layers[m].neurons[i].weights_to_nextlayer_neurons[k] * layers[m+1].neurons[k].delta_output;
			}
			layers[m].neurons[i].delta_output = layers[m].neurons[i].error*sigmoidDerivative(layers[m].neurons[i].output);
		}
	}
#ifdef DEBUG
    std::cout << "Calculating hidden layer(s) neurons delta weights..." << std::endl;
#endif
    // Propogate the delta backward
    for(int m = layers.size()-2; m >= 0; m--){
		for (int i = 0; i <  ALL_LAYER_UNITS[m]+1 ; i++) {
			// Propagate delta to weights from the previous layer
			// delta weight = deltaOfOutput*hidden_output
			for (int j = 0; j < ALL_LAYER_UNITS[m+1]; j++) {
				layers[m].neurons[i].delta_weights_to_nextlayer_neurons[j] = alpha*layers[m+1].neurons[j].delta_output*layers[m].neurons[i].output;
			}
    	}
	}
}

// Start learning depending on the inputs and expected outputs
void startLearning(std::vector<eachLayer>& layers, std::vector<eachLayer>& last_iteration, std::vector<std::vector<double>>& expected_outputs, std::vector<std::vector<double>> input_data){
    std::cout << "Learning...." << std::endl;
    // Output of bias depends on the BIAS_VALUE
    for (int m = 0; m < TOTAL_LAYERS - 1; m++) {
        // Bias
        layers[m].neurons[ALL_LAYER_UNITS[m]].output = BIAS_VALUE;
    }

	for (int epoch = 0; epoch < ITERATIONS; epoch++) {
		current_error = 0;
		for (int item = 0; item < input_data.size(); item++) {
#ifdef DEBUG
		std::cout << std::setfill('.') << std::setw(30) << "Item: " << item << std::endl;
#endif		
			// Output of input newron is same as external input; so just copy it
			for(int j = 0; j < input_data[item].size(); j++){ 
				layers[0].neurons[j].output = input_data[item][j];
			}
            
			forwardCalculation(layers);
			backwardCalculations(layers, expected_outputs, item);
			updateTheWeights(layers, last_iteration);
		}
		current_error = current_error / input_data.size();
		//std::cout << epoch << " " << current_error << std::endl;
	}
}

// Test the trained neural network
void testNeuralNetwork(std::vector<std::vector<double>>& test_data, std::vector<eachLayer_wts>& final_weights) {
    std::vector <eachLayer> test_layers(TOTAL_LAYERS);               // last layer is output
    std::cout << "Testing...." << std::endl;
    for (int m = 0; m < TOTAL_LAYERS - 1; m++) {
        // Resize the vector for test inputs
        test_layers[m].neurons.resize(ALL_LAYER_UNITS[m] + 1);
        // Bias output is BIAS_VALUE
        test_layers[m].neurons[ALL_LAYER_UNITS[m]].output = BIAS_VALUE;
    }
    // Output layer
    test_layers[TOTAL_LAYERS - 1].neurons.resize(ALL_LAYER_UNITS[TOTAL_LAYERS - 1] + 1); 
    for (int item = 0; item < test_data.size(); item++) {
        // Output of input newron is same as external input; so just copy it
        test_layers[0].neurons.resize(ALL_LAYER_UNITS[0]); // input layer
        for (int j = 0; j < test_data[item].size(); j++) {
            test_layers[0].neurons[j].output = test_data[item][j];
        } 
        
        // Calculate the output
        for (int m = 1; m < TOTAL_LAYERS; m++) {
           // test_layers[m].neurons.resize(ALL_LAYER_UNITS[m]);
            for (int i = 0; i < ALL_LAYER_UNITS[m]; i++) {
                // Each neuron is connected to all the neurons in the next layer
                for (int j = 0; j < ALL_LAYER_UNITS[m - 1]; j++) {
                    // Get the weighted sum
                    test_layers[m].neurons[i].output += final_weights[m - 1].neurons[j].weights_to_nextlayer_neurons[i] * test_layers[m - 1].neurons[j].output;
                }
                test_layers[m].neurons[i].output = sigmoid(test_layers[m].neurons[i].output);
            }
        }
        std::cout << "Inputs, " << "Output " << std::endl;
        // Print inputs and final output
        for (int j = 0; j < test_layers[0].neurons.size()-1; j++) {
            std::cout << test_layers[0].neurons[j].output << " ";
        }
        for (int m = 0; m < ALL_LAYER_UNITS[TOTAL_LAYERS - 1]; m++) {
            std::cout << test_layers[TOTAL_LAYERS - 1].neurons[m].output << std::endl;
        }
    }
}

void trainNeuralNetwork(std::vector<std::vector<double>>& input_data, std::vector<std::vector<double>>& expected_outputs, std::vector<eachLayer_wts>& final_weights) {
    std::vector<eachLayer> layers(TOTAL_LAYERS);               // last layer is output
    std::vector<eachLayer> last_iteration(TOTAL_LAYERS);       // last layer is output

    InitializeWeights(layers);

    if (momentumOn) {
#ifdef DEBUG
        std::cout << "With momentum" << std::endl;
#endif
        last_iteration = layers;
        momentum = true;
    }
    else {
#ifdef DEBUG
        std::cout << "Without momentum" << std::endl;
#endif
    }

    startLearning(layers, last_iteration, expected_outputs, input_data);
    storeFinalWeights(layers, final_weights);
    writeWeightsToFile(final_weights);
}
int main(int argc, char *argv[]) {
    std::vector<std::vector<double>> expected_outputs;         // store expected ouput
    std::vector<std::vector<double>> input_data;               // store input data
    std::vector<std::vector<double>> test_data;                // store test inputs
    std::vector<eachLayer_wts> final_weights(TOTAL_LAYERS);    // store final trained network weights

    std::ifstream input_file;
    std::ifstream test_input_file;
    std::ifstream expected_output_file;

    std::string input_filename = argv[1];
    std::string expected_output_filename = argv[2];
    std::string test_input_filename = argv[3];
    std::string output_filename = "weights_" + input_filename;

	srand(time(NULL));

	if (argc != 4) std::cerr << "Use ./MultilayerBPNN inputFile expectedOutputFile testInputsFile\n" << std::endl;

    readInputFile(input_file, input_filename, input_data);
    readInputFile(expected_output_file, expected_output_filename, expected_outputs);
    readInputFile(test_input_file, test_input_filename, test_data);
    openOutputFile(weights_output_file, output_filename);

    trainNeuralNetwork(input_data, expected_outputs, final_weights);
    testNeuralNetwork(test_data, final_weights);
	return 0;
}