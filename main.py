#!/usr/bin/python3
import os
import sys
import gzip
import shutil
import version
import argparse
from datetime import datetime
from metadevol import binning
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

def main():
	""" MetaDevol accurately reconstructs genomic bins from metagenomic samples using contig abundance and k-mer profiles"""
	parser = argparse.ArgumentParser(description="MetaDevol: An accurate metagenome binning of contigs based on decovolution of abundance and k-mer profiles")
	
	# usage = "metadevol.py -i <input directory of bam files> -c <assembly file>"
	
	# input files
	# group = parser.add_mutually_exclusive_group(required=True)
	# group.add_argument("-i", "--input", type=str, help="alignment files in bam or sam format")
	# group.add_argument("-f", "--inputfile", type=str, help="file having list of bam or sam format")
	parser.add_argument("-i", "--input", type=str, help= "directory that contains alignment files in bam format")
	# contig file
	parser.add_argument("-c", "--contigs", type=str, help="sequence file of contigs assembled from each samples in fasta format")
	
	# options processing
	parser.add_argument("-l", "--minlength", type=int, help="minimum length of contigs to be considered for initial binning", default=1500)
	parser.add_argument("-o", "--output", help="Name for the output file")
#    parser.add_argument("--cuda", help="Use gpu", action="store_true")
#    parser.add_argument("-N", "--ncores", help="Number of cores to use", default=1, type=int)
	parser.add_argument("-d", "--outdir", help="output directory, when undefined it is set to the input directory", default=None)
	parser.add_argument("-v", "--version", help="print version and exit", action="store_true")

	args = parser.parse_args()

	if not len(sys.argv) > 1:
		parser.print_help()
		sys.exit(0)

	if args.version:
		print("MetaDevol",version.__version__)
		exit(0)

	if args.input is None:
		print("Input missing! Please specifiy input directory to find bam files with -i/--input")
		exit(1)

	if not os.path.isdir(os.path.abspath(args.input)):
		print(f"Incorrect input for -i. Please specify the directory in which bam files are present (e.g, bamfiles/)")
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
		args.outdir = os.path.dirname(args.input)
		
	if args.output is None:
		args.output = datetime.today().strftime('%Y-%m-%d')

		if not os.path.exists(args.outdir + '/' + args.output):
			os.makedirs(args.outdir + '/' + args.output)
		else:
			pass

	args.outdir = args.outdir + '/' + args.output

	if not os.path.exists(args.input + '/tmp'):
		os.makedirs(args.input + '/tmp')
	else:
		# shutil.rmtree(args.input + '/tmp/')
		# os.makedirs(args.input + '/tmp', exist_ok=True)
		pass

	print("metagenome binning started...\n")
	print("input directory:\t\t", args.input)
	print("contig minimum length cutoff:\t", args.minlength, 'bp')
	print("output name:\t\t\t", args.output)
	print("output directory:\t\t", args.outdir)
	print("temporary folder:\t\t", args.outdir + '/tmp','\n')

	binning(args)

if __name__ == "__main__":
	main()