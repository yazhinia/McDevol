#!/bin/bash

# compile cpp codes for kmer frequency calculation and printing fasta sequence files for binned contigs
cd util
g++ -o kmerfreq kmerfreq.cpp -O3
g++ -o get_sequence_bybin get_sequence_bybin.cpp -O3
# end

# compile and build cpp pybind11 module for getting fractional read counts from bamfiles
bam2countso_file=(`find bam2counts/ -maxdepth 1 -name "bam2counts*.so"`)
if [ ! ${#bam2countso_file[@]} -gt 0 ]; then
bash setup_bam2counts.sh
cd bam2counts
if [ -d "build" ]; then
rm -rf build
fi
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPYBIND11_FINDPYTHON=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make && cd ../ && cp bam2counts.cpython* ../ && cd ../
fi
# end


# compile and build cpp pybind11 module for distance calculation
bayesiandistso_file=(`find bayesian_distance/ -maxdepth 1 -name "bayesian_distance*.so"`)

if [ ! ${#bayesiandistso_file[@]} -gt 0  ]; then
cd bayesian_distance
if [ -d "build" ]; then
rm -rf build
fi
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPYBIND11_FINDPYTHON=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make && cd ../ && cp bayesian_distance.cpython* ../ && cd ../
fi
echo "completed building bayesian distance module"
# end

so_file1=(`find -maxdepth 1 -name "bam2counts*.so"`)
so_file2=(`find -maxdepth 1 -name "bayesian_distance*.so"`)

if [ -z $so_file1 ]; then
cp bam2counts/bam2counts.cpython*.so .
fi

if [ -z $so_file2 ]; then
cp bayesian_distance/bayesian_distance.cpython*.so .
fi

cd ../

export LD_LIBRARY_PATH=$(pwd)/bam2counts/bamtools/build/src/
