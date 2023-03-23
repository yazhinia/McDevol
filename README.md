# MetaDevol

MetaDevol - Metagenome binning based on Deconvolution of abundance and k-mer profiles. 

Shown below an illustration for the underlying basis for binning.
(1) Contigs originated from same genome will have correlated abundance profiles across samples;
(2) k-mer (tetramer) frequency is a characteristics of microbial genomes and distinguishes genomes from different genus. Thus, contigs from same genome show correlation in k-mer frequency.

![binning_twosource_of_information](https://user-images.githubusercontent.com/29796007/227135720-bee8b197-3b8a-4020-9582-4c917a2b9b0a.png)

# Command line
`python3 main.py -i bamfiles -c contig.fasta`

`-i | --input` directory in which all bamfiles present
`-c | --contigs fasta file of contig sequenes assembled from all samples (sample-wise assembly)

## additional options

`-l | --minlength` minimum length of contigs to be considered for binning [default 1000kb]
`-o | --output` name of output file [default, date]
`-d | --outdir` output directory [default, working directory]

## help
`python3 main.py -h or python3 main.py`
