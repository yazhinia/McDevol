#!/bin/bash -e

fail() {
    echo "Error: $1"
    exit 1
}

notExists() {
        [ ! -f "$1" ]
}

#get absolute pathway function
abspath(){
    if [ -d "$1" ]; then
        (cd "$1"; pwd)
    elif [ -f "$1" ]; then #if file exists
        if [ -z "${1##*/*}" ]; then
            echo "$(cd "${1%/*}"; pwd)/${13##*/}"
        else
            echo "$(pwd)/$1"
        fi
    elif [ -d "$(dirname "$1")" ]; then #if directory to $1 exists
            echo "$(cd "$(dirname "$1")"; pwd)/$(basename "$1")"
    fi
}

SHORT=i:,l:,tmp:,h
LONG=alignment:,length_cutoff:,tmp_dir:,help
OPTS=$(getopt -a -n metadevol --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

usage()
{
    echo "Usage: $(basename $0) -i <alignment file > -l <length cutoff>"
    exit 2
}

while :
do
  case "$1" in
    -i | --alignment )
      ALIGNMENT_PATH="$2" #$(dirname ${2})
      shift 2
      ;;
    -l | --length_cutoff )
      CONTIG_LEN="$2"
      shift 2
      ;;
    -tmp | --tmp_dir )
      TMP_PATH="$2"
      shift 2
      ;;  
    -h | --help )
      usage
      ;;
    --)
      shift;
      break
      ;;
    * | ?)
      echo "Unexpected option: $1"
      usage
      ;;
  esac
done

CONTIG_FILENAME="CONTIGS_LIST"

# check contig length cutoff has been given or assign the default cutoff otherwise
if [ -z ${CONTIG_LEN} ];
then
CONTIG_LEN=1500
fi

# display error if input is missing
if [ -z ${ALIGNMENT_PATH} ];
then
    echo "Input is missing"
    usage
    exit 1
fi

# create temporary folder if not given
if [ -z ${TMP_PATH} ];
then
    TMP_PATH="tmp_folder"
    if [ ! -d ${TMP_PATH} ];
    then
      echo "creating tmp_folder"
      mkdir ${TMP_PATH}
    fi
else
    if [ ! -d ${TMP_PATH} ];
    then
        echo "creating temporary folder named \"${TMP_PATH}\"" 
        mkdir ${TMP_PATH}
    fi
fi

#  # remove if the list of parsed contigs based on length cutoff exist
# if [ -f "${TMP_PATH}/${CONTIG_FILENAME}" ];
# then
#     rm -f "${TMP_PATH}/${CONTIG_FILENAME}"
# fi

echo "Based on your inputs, \"${ALIGNMENT_PATH}\" is used to find sam files and \"${TMP_PATH}\" is used as a temporary directory"

# step 1
# parsing contigs by length and read alignments to contigs

# echo "Parsing contigs using ${CONTIG_LEN} length cutoff"; date +"%T"
# python3 parse_contigsbylen.py ${ALIGNMENT_PATH} ${CONTIG_LEN} ${CONTIG_FILENAME} ${TMP_PATH}
# date +"%T"

# echo "parsing alignment file"; date +"%T"
# ./parse_alignment ${ALIGNMENT_PATH} ${TMP_PATH}
# date +"%T"

# step 2
# obtain read counts

# echo "obtaining read counts"; date +"%T"
# python3 sam2count.py ${ALIGNMENT_PATH} ${CONTIG_FILENAME} ${TMP_PATH}
# ./bamtools ${ALIGNMENT_PATH} ${TMP_PATH}
# date +"%T"

# # step 3

echo "Processing read counts to get clusters and optimized W & Z matrices"; date +"%T"
python3 clustering_initialization_optimization.py ${TMP_PATH}
date +"%T"
echo "Metagenomic binning of contigs has been completed"


# if [ -d ${TMP_PATH} ];
# then
#     echo "runs were completed. Removing temporary folder"
#     rm -rf ${TMP_PATH}
# fi



# //
# step 2
# parse alignment
# echo "Parsing alignment file"; date +"%T"
# python3 parse_alignment.py ${ALIGNMENT_PATH} ${TMP_PATH}
# date +"%T"
