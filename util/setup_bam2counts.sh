#!/bin/bash

# compile and build cpp pybind11 module for getting fractional read counts from bamfiles
cd bam2counts

if [ ! -d "bamtools" ]; then
## build and install bamtools
git clone https://github.com/pezmaster31/bamtools.git
cd bamtools
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ .. -DBUILD_SHARED_LIBS=ON
make && make install

export LD_LIBRARY_PATH=$(pwd)/src/

#cd ../ && rm -rf build && mkdir build && cd build

#cmake -DCMAKE_INSTALL_PREFIX=../ ..
#make && make install
cd ../../../

else

echo "bamtools is already installed"

fi

echo "bamtools is installed"

# end

