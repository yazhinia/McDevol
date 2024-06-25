import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def get_random_fragments(fasta_file, output_file, fragment_length=2000):
	fragments = []
	
	for record in SeqIO.parse(fasta_file, "fasta"):
		print(record.id)
		seq_length = len(record.seq)
		start_pos = random.randint(0, seq_length - fragment_length)
		length = fragment_length + random.randint(0, seq_length - start_pos - fragment_length)
		end_pos = start_pos + length
		fragment = record.seq[start_pos:end_pos]
		fragment_id = f"{record.id}_{start_pos}:{end_pos}"
		fragments.append(SeqRecord(Seq(fragment), id=fragment_id, description=""))
	
	SeqIO.write(fragments, output_file, "fasta")
	print(f"Fragments written to {output_file}")
	
fasta_file = '/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/contigs_2k.fasta'
get_random_fragments(fasta_file, '/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/augment_seq1.fasta')
get_random_fragments(fasta_file, '/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/augment_seq2.fasta')

