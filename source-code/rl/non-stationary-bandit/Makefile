CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall
LDFLAGS = -larmadillo

TARGET = simulation
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.csv

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
