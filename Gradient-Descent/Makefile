CXX       = g++
CFLAGS    = -c
LDFLAGS   =
CXXFLAGS  = -g -std=c++11
BUILD_DIR = bin
SRC_DIR   = src

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp) 
OBJECT_FILES    := $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
EXECUTABLE_FILES := $(OBJECT_FILES:%.o=%)

all: $(EXECUTABLE_FILES)

# Simplest way to produce multiple executables
$(EXECUTABLE_FILES): $(BUILD_DIR)/%:$(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS)  $(LDFLAGS) -o $@ $<

clean_obj: 
	@mkdir $(BUILD_DIR)/*.o
  
clean:
	-@rm $(BUILD_DIR)/*

.PHONY: all clean_obj
