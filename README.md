## McDevol

McDevol - <ins>*M*</ins>etagenome binning of <ins>*c*</ins>ontigs based on <ins>*De*</ins>con<ins>*vol*</ins>ution of abundance and k-mer profiles. 

## Introduction
Metagenome binning relies on the following underlying basis (i) Contigs originated from same genome will have correlated abundance profiles across samples and (ii) k-mer (tetramer) frequency is a characteristics of microbial genomes and distinguishes genomes from different genus. Thus, contigs from same genome show correlation in k-mer frequency

Using this basis, contigs from the same genomes could be identified in the metagenome assembly and grouped into *M*etagenome-*a*ssembled *g*enomes (MAGs).

## Algorithm
McDevol uses novel Bayesian statistics-based distance measure between contigs to cluster them by their read counts and k-mer profiles. The initial clusters are merged further by distances with relaxed threshold into connected components. The outline of algorithm is shown below.

![McDevol_algorithm](https://user-images.githubusercontent.com/29796007/235193887-ba72c9b6-dffa-4440-a88c-9fbd5e603378.png)

## Installation
      git clone https://github.com/yazhinia/McDevol.git --recurse-submodules
      cd McDevol
      git submodule init
      git submodule update --init --recursive
      
      conda create -n mcdevol_env python=3.8 numpy scipy pandas memory_profiler alive_progress psutil
      conda activate mcdevol_env
      bash ./set_up.sh
      export PATH=$PATH:<path to McDevol>      
Now ready to use.

## Command line
`python3 mcdevol.py -i bamfiles -c contig.fasta`

`-i | --input` directory in which all bamfiles present

`-c | --contigs` fasta file of contig sequenes assembled from all samples (sample-wise assembly)

note: bamfiles should be unsorted (i.e., alignments are arranged by read names as provided by aligners by default)

## Additional options

`-l | --minlength` minimum length of contigs to be considered for binning [default 1000kb]

`-o | --output` name of output file [default, date]

`-d | --outdir` output directory [default, working directory]

`--fasta` output fasta file for each bin


## Help
`python3 mcdevol.py -h or python3 mcdevol.py`


## Custome installation with bamtools pre-installed
MetaDevol uses bamtools API for processing alignment bam files. When you run `bash setup.sh`, bamtools will be automatically installed and no modification is required. If the user has bamtools already installed in their system, then please go to bam2counts folder of McDevol and edit CMakeLists.txt file at target_link_libraries and target_include_directories lines as follows.

      target_link_libraries(bam2counts PRIVATE "${PATH}/bamtools/lib64/libbamtools.so")
      target_link_libraries(bam2counts PRIVATE -lz)
      target_include_directories(bam2counts PRIVATE "${PATH}/bamtools/include/bamtools/")
      target_include_directories(bam2counts PRIVATE "${PATH}/bamtools/src/")
      
update `${PATH}` to the absolute parent path of bamtools where it is installed. Then in bam2counts folder, run `bash build.sh && cd ../ && bash set_up.sh` to install McDevol. 



