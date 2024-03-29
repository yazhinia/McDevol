#!/usr/bin/env python

import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import gzip
import version
import argparse
from datetime import datetime
# from src.contigs_binning_current import binning
# from src.contigs_binning_onlyreads import binning
# from src.contigs_binning_onlykmers import binning
from src.contigs_binning import binning

def main():
	""" McDevol accurately reconstructs genomic bins from metagenomic samples using contig abundance and k-mer profiles"""
	parser = argparse.ArgumentParser(description="McDevol: An accurate metagenome binning of contigs based on decovolution of abundance and k-mer profiles")
	
	# input files
	# input_group = parser.add_mutually_exclusive_group(required=True)
	parser.add_argument("-i", "--input", type=str, help= "directory that contains all alignment files in bam format")
	# input_group.add_argument("--count_matrix", type=str, help= "read count matrix in Contigs times samples dimension (CxN) in tab separated format without headers")
	# contig file
	parser.add_argument("-c", "--contigs", type=str, help="contig sequence file in fasta format (or zip)")
	
	# options processing
	parser.add_argument("-l", "--minlength", type=int, help="minimum length of contigs to be considered for initial binning", default=2000)
	parser.add_argument("-s", "--seq_identity", type=float, help="minimum sequence identity for selecting contigs during read mapping", default=97.0)
	parser.add_argument("-o", "--output", help="Name for the output file")
	parser.add_argument("--fasta", help="output bin sequences in fasta", action="store_true")
	parser.add_argument("-n", "--ncores", help="Number of cores to use", default=os.cpu_count(), type=int)
	parser.add_argument("-d", "--outdir", help="output directory, when undefined it is set to the input directory", default=None)
	parser.add_argument("-v", "--version", help="print version and exit", action="store_true")

	args = parser.parse_args()

	if not len(sys.argv) > 1:
		parser.print_help()
		sys.exit(0)

	if args.version:
		print("McDevol",version.__version__)
		exit(0)

	if args.input is None :
		print("Input missing! Please specifiy input directory to find bam files with -i/--input")
		exit(1)

	if not os.path.isdir(os.path.abspath(args.input)):
		print(f"Incorrect input for -i. Please specify the directory in which bam files are present (e.g, parent_path/bamfiles)")
		exit(1)
	else:
		if not any(fname.endswith('.bam') for fname in os.listdir(args.input)):
			print(f"input directory doesn't contain bam files")
			exit(1)
		else:
			args.input = os.path.abspath(args.input)
	
	if args.contigs is None:
		print("Input missing! Please specifiy contig assembly file with -c or --contigs")
		exit(1)
	else:
		args.contigs = os.path.abspath(args.contigs)
		if args.contigs is gzip.GzipFile(args.contigs, 'r'):
			args.contigs = gzip.open(args.contigs,'rb')
	
	if args.outdir is None:
		args.outdir = os.path.dirname(args.input) +  '/'
	else:
		args.outdir = args.outdir + '/'
			
	if args.output is None:
		args.output = "mcdevol" # datetime.today().strftime('%Y-%m-%d')


	try:
		os.makedirs(args.outdir)
	except:
		pass
	
	try:
		os.makedirs(args.outdir + '/tmp')
	except:
		pass
	
	
	args.tmp = args.outdir + 'tmp/'

	print("McDevol: metagenome binning started...\n")
	print("input directory:\t\t", args.input)
	print("contig minimum length cutoff:\t", args.minlength, 'bp')
	print("sequence identity cutoff:\t", args.seq_identity, 'bp')
	print("output name:\t\t\t", args.output)
	print("output directory:\t\t", args.outdir)
	print("temporary folder:\t\t", args.tmp,'\n')

	args.logfile = open(args.outdir + '/logfile.txt', 'w')

	binning(args)

	args.logfile.close()

if __name__ == "__main__":
	main()
