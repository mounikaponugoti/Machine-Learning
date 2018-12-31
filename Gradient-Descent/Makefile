CXX       := g++
CXXFLAGS  := -g -std=c++11
TARGET    := batchGradientDescent
BUILD_DIR := ./bin
SRC_DIR   := ./src

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(SRC_DIR)/$(TARGET).cpp
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/$(TARGET).cpp -o $(BUILD_DIR)/$(TARGET) 

build:
	@mkdir $(BUILD_DIR)
  
clean:
	-@rm $(BUILD_DIR)/*
