#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <vector>
#include <array>
#include <set>
#include <unordered_map>
#include <algorithm>

void construct_set(std::string &tmp_dir, std::set<std::string> &selected_contigs) {
    std::ifstream contigs_list;
    contigs_list.open(tmp_dir + "selected_contigs");
    std::string line, id;
    while (std::getline(contigs_list, line)) {
        std::stringstream st(line);
        st >> id;
        st >> id;
        selected_contigs.insert(id);
        st >> id;
    }
    contigs_list.close();
}

bool condition_header(std::string &line) {
    if (line[0] == '>') {
        return true;
    }
    else {
        return false;
    }
}

void get_complete_sequence(std::ifstream &fasta, std::set<std::string> &selected_contigs, std::string &line, std::string &sequence) {
    
    if (selected_contigs.find(line.substr(1,-1)) != selected_contigs.end()) {
        fasta >> line;
        while (!condition_header(line) && !fasta.eof()) {
            sequence.append(line);
            fasta >> line;
        }
    }
    else {
        fasta >> line;
        while (!condition_header(line) && !fasta.eof()) {
            fasta >> line;
        }
    }
}

int main(int argc, char *argv[]) {
    
    if(argc != 3 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        std::cout << "provide working directory and input fasta file as below \n";
        std::cout << "kmerfreq working_dir contig.fasta" << "\n";
        return EXIT_SUCCESS;
    }
    else {
        auto start = std::chrono::high_resolution_clock::now();
        std::string tmp_dir = argv[1]; 
        std::string contigs_sequences = argv[2]; 
        // std::string tmp_dir = "/big/work/metadevol/cami2_datasets/marine/";
        // std::string contigs_sequences = "/big/work/metadevol/cami2_datasets/marine/all_contigs.fasta"; 
        std::set<std::string> selected_contigs;
        std::cout << "calculating k-mer frequencies for contigs ..." << "\n";
        construct_set(tmp_dir, selected_contigs);
        
        unsigned int total_contigs = selected_contigs.size();
        
        std::cout << total_contigs << " sequences \n";

        // get a set of contigs that have sequence length above cut-off   
        std::vector<std::array<int, 256>> kmer_counts(total_contigs);
        std::vector<float> GC_counts(total_contigs);

        // assign each of four nucleotide base to an int value
        std::array<int,256> nt{};
        nt['A'] = 0;
        nt['T'] = 1;
        nt['G'] = 2;
        nt['C'] = 3;

        std::array<int,256> nt_r{}; // reverse complement
        nt_r['A'] = nt['T'];
        nt_r['T'] = nt['A'];
        nt_r['G'] = nt['C'];
        nt_r['C'] = nt['G'];
    
        std::ifstream fasta(contigs_sequences);

        if (!fasta.is_open()) {
            std::cerr << "Error: Unable to open Fasta file!" << "\n";
            return 1;
        }

        std::string line, sequence;
        std::pair<int,int> kmer_type;
        fasta >> line;

        unsigned int sequence_counter = 0;
        while (condition_header(line)) {

            sequence = "";
            get_complete_sequence(fasta, selected_contigs, line, sequence);
            int hash_value = 0;

            if (!sequence.empty()) {

                // forward direction
                hash_value += nt[sequence[0]] * 4 * 4 * 4;
                hash_value += nt[sequence[1]] * 4 * 4;
                hash_value += nt[sequence[2]] * 4;
                hash_value += nt[sequence[3]];

                if (((hash_value-0) | (255-hash_value)) >= 0) {
                    kmer_counts[sequence_counter][hash_value]++;
                }

                for (size_t i = 1; i < sequence.length()-3; i++) {
                    hash_value -= nt[sequence[i-1]] * 4 * 4 * 4;
                    hash_value *= 4;
                    hash_value += nt[sequence[i+3]];

                    if (((hash_value-0) | (255-hash_value)) >= 0) {
                        kmer_counts[sequence_counter][hash_value]++;
                    }
                }

                // reverse directions
                hash_value = 0;
                hash_value += nt_r[sequence[sequence.length()-1]] * 4 * 4 * 4;
                hash_value += nt_r[sequence[sequence.length()-2]] * 4 * 4;
                hash_value += nt_r[sequence[sequence.length()-3]] * 4;
                hash_value += nt_r[sequence[sequence.length()-4]];

                if (((hash_value-0) | (255-hash_value)) >= 0) {
                    kmer_counts[sequence_counter][hash_value]++;
                }

                for (size_t i = sequence.length()-1; i > 3; i--) {
                    hash_value -= nt_r[sequence[i]] * 4 * 4 * 4;
                    // std::cout << sequence[i] << " " << nt_r[sequence[i]] << "\n";
                    hash_value *= 4;
                    hash_value += nt_r[sequence[i-4]];

                    if (((hash_value-0) | (255-hash_value)) >= 0) {
                        kmer_counts[sequence_counter][hash_value]++;
                    }
                }

                for (size_t i = 0; i < sequence.length(); i++) {
                    if (nt[sequence[i]] == 2 || nt[sequence[i]] == 3) {
                        GC_counts[sequence_counter] = GC_counts[sequence_counter] + 1.0f;
                    }
                }
                GC_counts[sequence_counter] = GC_counts[sequence_counter] / sequence.length();
                sequence_counter++;
            }
        }
        
        // write counts to a file
        std::ofstream outfile(tmp_dir + "kmer_counts");

        for (auto i = kmer_counts.begin(); i !=kmer_counts.end(); i++) {
            for (auto j = i->begin(); j != i->end() ; j++) {
                outfile << *j <<"\n";
            }
        }


        // write fraction of GC counts to a file
        std::ofstream outfile1(tmp_dir + "GC_fractionof_contigs");
        for (size_t i = 0; i < total_contigs ; i++) {
            outfile1 << GC_counts[i] << "\n";
        }

        outfile.close();
        outfile1.close();
        std::cout << sequence_counter << " sequences processed \n";
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        std::cout << duration.count() << " seconds\n";
        return  EXIT_SUCCESS;
    }
}

