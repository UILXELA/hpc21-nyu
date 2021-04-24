CXX = mpic++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

CC = mpicc
CFLAGS = -O3 -march=native -fopenmp

RM = rm -f
MKDIRS = mkdir -p

CXX_TARGETS = $(basename $(wildcard *.cpp))
CC_TARGETS = $(basename $(wildcard *.c))


all : $(CXX_TARGETS) $(CC_TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

%:%.c *.h
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	$(RM) $(CXX_TARGETS) $(CC_TARGETS)
