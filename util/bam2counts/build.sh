#!/bin/bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make && cd ../ && cp bam2counts.cpython* ../ && cd ../
