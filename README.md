# McDevol

McDevol - <ins>*M*</ins>etagenome binning of <ins>*c*</ins>ontigs based on <ins>*De*</ins>con<ins>*vol*</ins>ution of abundance and k-mer profiles. 

# Introduction
Metagenome binning relies on the following underlying basis,
(1) Contigs originated from same genome will have correlated abundance profiles across samples
(2) k-mer (tetramer) frequency is a characteristics of microbial genomes and distinguishes genomes from different genus. Thus, contigs from same genome show correlation in k-mer frequency

![binning_twosource_of_information](https://user-images.githubusercontent.com/29796007/227135720-bee8b197-3b8a-4020-9582-4c917a2b9b0a.png)
Using this basis, contigs from the same genomes could be identified in the metagenome assembly and grouped into <ins>*M*<\ins>etagenome - <ins>*a*<\ins>ssembled <ins>*g*<\ins>enomes (MAGs).

# Algorithm
McDevol clusters metagenome contigs using novel Bayesian statistics-based distance measure between contigs by their read counts and k-mer profiles. The initial clusters are merged further by distances with relaxed threshold to combine clusters into connected components. Non-Megative Matrix Factorization is performed on components to bin contigs by learning linear mixture models. The outline of algorithm is show below.

![MetaDevol_algorithm_workflow](https://user-images.githubusercontent.com/29796007/230059880-d9d4f062-5793-4ff2-963d-7e9193314266.png)

# Command line
`python3 main.py -i bamfiles -c contig.fasta`

`-i | --input` directory in which all bamfiles present

`-c | --contigs` fasta file of contig sequenes assembled from all samples (sample-wise assembly)

## additional options

`-l | --minlength` minimum length of contigs to be considered for binning [default 1000kb]

`-o | --output` name of output file [default, date]

`-d | --outdir` output directory [default, working directory]


## help
`python3 main.py -h or python3 main.py`


## requirement
MetaDevol uses bamtools API for alignment file processing. Users are requested to install it separately (refer https://github.com/pezmaster31/bamtools/wiki).
