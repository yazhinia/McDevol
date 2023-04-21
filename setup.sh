#!/bin/bash

# compile cpp codes for kmer frequency calculation and printing fasta sequence files for binned contigs
cd util
g++ -o kmerfreq kmerfreq.cpp
g++ -o get_sequence_bybin get_sequence_bybin.cpp
# end

# compile and build cpp pybind11 module for getting fractional read counts from bamfiles
so_file=(`find bam2counts/ -maxdepth 1 -name "bam2counts*.so"`)
if [ ! ${#so_file[@]} -gt 0 ]; then
bash setup_bam2counts.sh
fi
# end

# compile and build cpp pybind11 module for distance calculation
cd metadevol_distance
so_file=(`find -name "metadevol_distance*.so"`)
if [ ! ${#so_file[@]} -gt 0 ]; then
if [ -d "build" ]; then
rm -rf build
fi
fi
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make && cd../ && cp metadevol_distance.cpython* ../ && cd ../
# end

cd ../


