## McDevol

McDevol - <ins>*M*</ins>etagenome binning of <ins>*c*</ins>ontigs based on <ins>*De*</ins>con<ins>*vol*</ins>ution of abundance and k-mer profiles. 

## Introduction
Metagenome binning relies on the following underlying basis,
(1) Contigs originated from same genome will have correlated abundance profiles across samples
(2) k-mer (tetramer) frequency is a characteristics of microbial genomes and distinguishes genomes from different genus. Thus, contigs from same genome show correlation in k-mer frequency

![binning_twosource_of_information](https://user-images.githubusercontent.com/29796007/227135720-bee8b197-3b8a-4020-9582-4c917a2b9b0a.png)
Using this basis, contigs from the same genomes could be identified in the metagenome assembly and grouped into *M*etagenome-*a*ssembled *g*enomes (MAGs).

## Algorithm
McDevol uses novel Bayesian statistics-based distance measure between contigs to cluster them by their read counts and k-mer profiles. The initial clusters are merged further by distances with relaxed threshold into connected components. Non-Megative Matrix Factorization is then performed on components to bin contigs through learning linear mixture models. The outline of algorithm is shown below.

![MetaDevol_algorithm_workflow](https://user-images.githubusercontent.com/29796007/230059880-d9d4f062-5793-4ff2-963d-7e9193314266.png)

## Installation
      git clone https://github.com/yazhinia/McDevol.git
      cd McDevol
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



