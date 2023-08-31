#!/bin/bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPYBIND11_FINDPYTHON=ON ..
make && cd ../ && cp bam2counts.cpython* ../ && cd ../
