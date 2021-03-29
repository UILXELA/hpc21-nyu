CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

CC = gcc
CFLAGS = -O3 -march=native -fopenmp

RM = rm -f
MKDIRS = mkdir -p

CXX_TARGETS = $(basename $(wildcard *.cpp))
CC_TARGETS = $(basename $(wildcard *.c))


all : $(CXX_TARGETS) $(CC_TARGETS)

%:%.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

%:%.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	$(RM) $(CXX_TARGETS) $(CC_TARGETS)

