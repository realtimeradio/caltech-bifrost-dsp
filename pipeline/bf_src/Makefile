CC=/usr/local/cuda/bin/nvcc
CXXFLAGS=-std=c++11
NVFLAGS=-lcublas -Wno-deprecated-gpu-targets

BINDIR = .
SRCDIR = .

.PHONY: all fast_debug debug verbose junk dada_buffers clean

all: CXXFLAGS += -O3 -use_fast_math -arch=sm_75 -restrict
all: beam

beam: $(SRCDIR)/cublas_beamform.cu
	$(CC) -o $(BINDIR)/$@ $^ -L$(CXXFLAGS) $(NVFLAGS)

clean:
	-rm ./beam
