## McDevol
A fast and memory-efficient metagenome binning tool

## Introduction
Metagenome binning relies on the following underlying basis (i) contigs originated from the same genome will have correlated abundance profiles across samples and (ii) k-mer (tetramer) frequency is a characteristics of microbial genomes and distinguishes genomes from different genus. Thus, contigs from the same genome show correlation in k-mer frequency. Using correlation in profiles of read and k-mer counts, contigs from the same genomes could be identified in the metagenome assembly and binned into *M*etagenome-*a*ssembled *g*enomes (MAGs).

## Algorithm
McDevol uses a novel Bayesian statistics-based distance measure on read counts and k-mer profiles to bin metagenomic contigs. The method has two steps, (i) initial agglomerative clustering using bayesian distance and (ii) density-based clustering using summed read and k-mer count profiles to merge clusters of possibly the same genome into components to provide final genomic bins. An outline of algorithm is depicted below.

![McDevol_algorithm](https://user-images.githubusercontent.com/29796007/235193887-ba72c9b6-dffa-4440-a88c-9fbd5e603378.png)

## Installation
      git clone https://github.com/yazhinia/McDevol.git --recurse-submodules
      cd McDevol
      pip install -r requirements.txt
      bash setup.sh
      export PATH=$PATH:<path to McDevol>
      mcdevol.py -i test -c test/contigs.fasta -o out # testing
Now ready to use.
<!--- conda create -n mcdevol_env python numpy scipy pandas alive_progress
      conda activate mcdevol_env --->
## Advantages

(i) McDevol finds contigs belonging to the same genome using a novel distance measure, defined as the posterior probability that the count profiles of contigs are drawn from the same distribution.

(ii) It applies a simple agglomerative algorithm to get high-purity clusters followed by merging clusters of the same genomic origin through density-based clustering to increase completeness. This approach is much simpler and faster than an iterative medoid clustering and expectation-maximization algorithm used by MetaBat2 and MaxBin2, respectively. 

(iii) It does not rely on a set of single-copy marker genes to refine clusters as done by other existing binners which results in over-estimation completeness and purity measures during CheckM evaluation.

Together, this tool is very fast, memory-efficient and less dependent on external tools.

<!--- McDevol takes roughly 2min to complete metagenome binning of CAMI2 marine dataset while MetaBAT2, the fastest and memory-efficient binner that exists, takes ~1hr. Memory usage of McDevol is ~400Mb while MetaBAT2 requires 1.5Gb. Together, McDevol is the fastest and memory-efficient binning tool and would be suitable choice for large-scale metagenome binning. More details on McDevol performance will be given in the near future... --->


## Command line
`mcdevol.py -i bamfiles -c contig.fasta`

`-i | --input` directory in which all bamfiles are present

`-c | --contigs` a fasta file for contig sequences (single-sample or co-assembly)

note: input bamfiles should be unsorted (i.e., a default output of aligners and alignments are arranged by read names). As of now, McDevol supports bamfiles from `bwa-mem` and `bowtie2` tools.

## Additional options

`-l | --minlength` minimum length of contigs to be considered for binning [default 1000kb]

`-o | --output` the name of output file [default, 'mcdevol']

`-d | --outdir` output directory [default, working directory]

`--fasta` output fasta file for each bin


## Help
`mcdevol.py -h or mcdevol.py`

## Recommended workflow
We recommend single-sample assembly to obtain contigs as it minimizes constructing ambiguous assemblies for strain genomes. Perform mapping on a concatenated list of contigs for each sample and run McDevol. Bins from single-sample assembly input are redundant because the same genomic region can be represented by multiple contigs assembled independently from different samples. To remove redundancy, we recommend the following post-binning redundancy reduction steps.

## Metagenome binning of contigs from sample-wise assembly
When the contigs are assembled from each sample, perform post-binning assembly and clustering on _every bin_ produced by Mcdevol. For this, users are requested to have plass (https://github.com/soedinglab/plass) and MMseqs2 (https://github.com/soedinglab/MMseqs2) separately installed.

### 1) post-binning assembly
      plass nuclassemble bin<0..N>.fasta bin<0..N>_assembled.fasta tmp --max-seq-len 10000000 --keep-target false --contig-output-mode 0 --min-seq-id 0.990 --chop-cycle false
      
### 2) sequence clustering
      mmseqs easy-linclust bin<0..N>.fasta output<0..N> tmp --min-seq-id 0.970 --min-aln-len 200 --cluster-mode 2 --shuffle 0 -c 0.99 --cov-mode 1 --max-seq-len 10000000

<!---## Custome installation with bamtools pre-installed
MetaDevol uses bamtools API for processing alignment bam files. When you run `bash setup.sh`, bamtools will be automatically installed and no modification is required. If the user has bamtools already installed in their system, then please go to bam2counts folder of McDevol and edit CMakeLists.txt file at target_link_libraries and target_include_directories lines as follows.

      target_link_libraries(bam2counts PRIVATE "${PATH}/bamtools/lib64/libbamtools.so")
      target_link_libraries(bam2counts PRIVATE -lz)
      target_include_directories(bam2counts PRIVATE "${PATH}/bamtools/include/bamtools/")
      target_include_directories(bam2counts PRIVATE "${PATH}/bamtools/src/")
      
update `${PATH}` to the absolute parent path of bamtools where it is installed. Then in bam2counts folder, run `bash build.sh && cd ../ && bash set_up.sh` to install McDevol. --->
