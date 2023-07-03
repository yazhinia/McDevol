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


void construct_unorderedmap(std::string &tmp_dir, std::unordered_map<std::string, size_t> &selected_contigs) {
    std::ifstream contigs_list;
    contigs_list.open(tmp_dir + "selected_contigs");
    std::string line, id;
    unsigned int sequence_counter = 0;
    while (std::getline(contigs_list, line)) {
        std::stringstream st(line);
        st >> id;
        st >> id;
        selected_contigs[id] = sequence_counter;
        st >> id;
        sequence_counter++;
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

std::string get_header(std::string line) {
    if(line.length() > 0) {
        if (line.find('_') != std::string::npos) {
            return line.substr(1,int(line.find('_'))-1);
        } else {
            return line.substr(1,int(line.find(' '))-1);
        }
    }
    return "";
}

// bool stringContainsCharacters(const std::string& str)
// {
//     const std::string characters = "#;/_-=";

//     for (char c : str) {
//         if (std::find(characters.begin(), characters.end(), c) != characters.end())
//         {
//             return true;
//         }
//         if(std::isdigit(c)) {
//             return true;
//         }
//         else {
//             return false;
//         }
//     }
//     return false;
// }

void get_complete_sequence(std::ifstream &fasta, std::unordered_map<std::string, size_t> &selected_contigs, std::string &line, std::string &sequence, std::string &current_contig, unsigned int &sequence_counter) {
        
    line = get_header(line);
    auto it = selected_contigs.find(line);
    
    if (it != selected_contigs.end()) {
        
        if (current_contig != line) { // update sequence id and sequence counter when it matches to selected contigs
            sequence_counter=selected_contigs.at(line);
        }
        
        getline(fasta, line);
        
        while (!condition_header(line) && !fasta.eof()) {
            sequence.append(line);
            getline(fasta, line);
        }    
    }
    else {
        getline(fasta, line);
        while (!condition_header(line) && !fasta.eof()) {
            getline(fasta, line);
        }
    }
}

int main(int argc, char *argv[]) {
// int main() {
    
    if(argc != 3 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        std::cout << "provide working directory and input cds fasta file as below \n";
        std::cout << "count_codons working_dir cds.fasta" << "\n";
        return EXIT_SUCCESS;
    }
    else {
        auto start = std::chrono::high_resolution_clock::now();
        std::string tmp_dir = argv[1]; 
        std::string contigs_sequences = argv[2]; 
        std::unordered_map<std::string, size_t> selected_contigs;
        std::cout << "counting codon frequencies from contig sequences ..." << "\n";
        construct_unorderedmap(tmp_dir, selected_contigs);
        
        unsigned int total_contigs = selected_contigs.size();

        // get a set of contigs that have sequence length above cut-off   
        std::vector<std::array<int, 64>> codon_counts(total_contigs);

        std::unordered_map<char, int> nt = {
        {'A', 0},
        {'T', 1},
        {'G', 2},
        {'C', 3},
        {'N', 65},
        };

        std::ifstream fasta(contigs_sequences);

        if (!fasta.is_open()) {
            std::cerr << "Error: Unable to open Fasta file!" << "\n";
            return 1;
        }

        std::string line, sequence;
        std::string current_contig;
        
        if (getline(fasta, line)) {

            current_contig = get_header(line);

            unsigned int sequence_counter = 0;
            
            while ((condition_header(line)) || (!fasta.eof())) {
                
                sequence = "";

                get_complete_sequence(fasta, selected_contigs, line, sequence, current_contig, sequence_counter);
                
                int hash_value=0;

                if (!sequence.empty()) {

                    for (size_t i = 0; i < sequence.length()-5; i += 3) {

                        hash_value = nt.at(sequence[i]) * 16 + nt.at(sequence[i+1]) * 4 + nt.at(sequence[i+2]);

                        if (hash_value < 65) {

                            if ((hash_value == 16) | (hash_value == 18)) {
                                std::cout << i << "th codon position " << current_contig << " " << sequence[i] << sequence[i+1] << sequence[i+2] << " Warning! stop codon in the middle of coding sequence. Please check your input\n";
                            }
                            if (((hash_value-0) | (63-hash_value)) >= 0) {
                                codon_counts[sequence_counter][hash_value]++;
                            }
                        }
                        hash_value = 0;
                    }
                }

                // if (current_contig != get_header(line)) {
                //     sequence_counter++;
                //     current_contig = get_header(line);
                // }
            }
        
        // write counts to a file
        std::ofstream outfile(tmp_dir + "codon_counts");

        for (auto i = codon_counts.begin(); i !=codon_counts.end(); i++) {
            for (auto j = i->begin(); j != i->end() ; j++) {
                outfile << *j <<"\n";
            }
        }
        outfile.close();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        std::cout << sequence_counter++ << " sequences processed " << duration.count() << " seconds\n";
        return  EXIT_SUCCESS;
        }
    }
}

