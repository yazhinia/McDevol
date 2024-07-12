#include <iostream>
#include <stdio.h>
#include <string>
#include <cmath>
#include <fstream>
#include <cstring>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <functional>
#include <filesystem>
#include <unordered_map>


const unsigned int FLAG_0 = 0;      // 0 primary alignment
const unsigned int FLAG_1 = 1;      // 0x1 template having multiple segments in sequencing
const unsigned int PAIRED_FLAG = 2;      // 0x2 each segment properly aligned according to the aligner (paired)
const unsigned int FLAG_3 = 4;      // 0x4 segment unmapped
const unsigned int FLAG_4 = 8;      // 0x8 next segment in the template unmapped
const unsigned int FLAG_5 = 16;     // 0x10 SEQ being reverse complemented
const unsigned int FLAG_6 = 32;     // 0x20 SEQ of the next segment in the template being reverse complemented
const unsigned int FORWARD_FLAG = 64;     // 0x40 the first segment in the template (forward read R1)
const unsigned int REVERSE_FLAG = 128;    // 0x80 the last segment in the template (reverse read R2)
const unsigned int FLAG_9 = 256;    // 0x100 secondary alignment
const unsigned int FLAG_10 = 512;   // 0x200 not passing filters, such as platform/vendor quality controls
const unsigned int FLAG_11 = 1024;  // 0x400 PCR or optical duplicate
const unsigned int FLAG_12 = 2048;  // 0x800 supplementary alignment

struct al_data{
    std::string read;
    std::string contig;
    unsigned int pair_dir; 
    bool paired;
    bool averaged;
    float sequence_identity;
};

struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // A simple combination function
        return h1 ^ h2;
    }
};

using AlnParser = std::vector<al_data>;
using AlnStat = std::pair<float, float>;
using ContigsMap = std::unordered_map<std::string, float>;
using ContigLinks = std::unordered_map<std::pair<std::string, std::string>, float, PairHash>;

void addpairlinks(ContigLinks& contiglinks, const std::string& c1, const std::string& c2, float frac_contigs_mapped) {
    
    std::string first = (c1 < c2) ? c1 : c2;
    std::string second = (c1 < c2) ? c2 : c1;
    std::pair<std::string, std::string> linkPair = std::make_pair(first, second);
    contiglinks[linkPair] += frac_contigs_mapped;
}

void construct_map(std::unordered_map<std::string, float>& contigs_map, std::string line, unsigned int minlength) {
    float init_count = 0.0f;
    std::string header, id, length;
    unsigned int contig_length;
    if (line.rfind("@SQ",0) == 0) {
        std::istringstream iss(line);
        iss >> header >> id >> length;
        std::string contig_id=id.substr(3, id.length());
        contig_length = std::stoi(length.substr(3, length.length()));
        if (contig_length >= minlength) {
            contigs_map.insert({contig_id, init_count});
        }
    }
}

std::tuple<int, int, bool> get_alnpos(std::string &cigar_str) {
    int start, end, pos1, pos2 = 0;
    char op1, op2;
    bool flag;

    if (sscanf(cigar_str.c_str(), "%d%c", &pos1, &op1) == 2) {
        start = 0;
        end = pos1;
        flag = true;
    } else if (sscanf(cigar_str.c_str(), "%d%c%d%c", &pos1, &op1, &pos2, &op2) == 4) {
        if(op1 == 'M') {
            start = 0;
            end = pos1;
            if(op2 == 'S') {
                flag = true;
            } else {
                flag = false;
            }
        } else {
            start = pos1;
            end = start + pos2;
            if(op1 == 'S') {
                flag = true;
            } else {
                flag = false;
            }
        }
    } else {
        start = 0;
        end = 0;
        flag = true;
    }

    return std::make_tuple(start, end, flag);
}


std::pair<float, float> get_seqid_alncov(std::tuple<int, int, bool> &alnpos, std::string &qual_str, std::string &md_str) {
    
    unsigned int matches = 0;
    unsigned int mismatches = 0;
    std::string mismatch_string;
    std::istringstream ss(md_str);
    unsigned int future_matches = 0;
    ss >> future_matches;

    int start = std::get<0>(alnpos);
    int end = std::get<1>(alnpos);
    
    if(!std::get<2>(alnpos)) {
        start = 0;
        end = end - start;
    }

    unsigned int alignment_length = end - start;

    for (unsigned int i = start; i < end; i++) {
        bool is_match;
        if (future_matches > 0) {
            future_matches--;
            is_match = true;
        } else {
            is_match = false;
            char tt;
            ss >> tt;
            int t_match;
            if(ss >> t_match) {
                future_matches = t_match;
            } else {
                //do nothing
            }
        }
        if (qual_str[i] >= 20 + 33) {
            if(is_match){
                matches++;
            } else {
                mismatches++;
            }
        }
    }
    float seq_id, alignment_coverage;

    seq_id = (float)(matches * 100) / (float)(matches+mismatches);
    alignment_coverage = (float)(alignment_length * 100) / (float)(qual_str.length() + start); // account for variable read length

    // std::cout << matches << " matches " << mismatches << " mismatches " <<  seq_id << " " << alignment_coverage << " " << matches+mismatches << " in alnstats \n";
    
    return std::make_pair(seq_id, alignment_coverage);
}

void storealnstats(AlnParser &parsedaln, AlnStat &alnstats, std::string &currentread_id, std::string &contig_id, unsigned int &pair_flag, bool & proper_pair) {
    auto ic = std::find_if(parsedaln.rbegin(), parsedaln.rend(),[&](const al_data& a) {return a.contig == contig_id;});
    if (ic != parsedaln.rend()){ 
        if (ic->pair_dir != pair_flag){ // if mate pair alignment is found
            if (!ic->averaged) {
                ic->sequence_identity = (ic->sequence_identity + alnstats.first) / 2.0f;
                ic->averaged = true;
            } else {
                if(ic->sequence_identity < ((ic->sequence_identity + alnstats.first) / 2.0f)) {
                    ic->sequence_identity = (ic->sequence_identity + alnstats.first) / 2.0f;
                    ic->averaged = true;
                }
            }
            ic->paired = true;
        } else { // update sequence identity if the same read (of the same direction) mapped to the same contig with higher sequence identity
            if (ic->sequence_identity < alnstats.first) {
                ic->sequence_identity = alnstats.first;
            }
        }
    } else { // read mapped to different contig 
        parsedaln.push_back({currentread_id, contig_id, pair_flag, proper_pair, false, alnstats.first});
    }
}

void counting(AlnParser &parsedaln, ContigsMap &contigs_map, ContigLinks &contig_links, float &sequenceidentity, std::ofstream &eachcount) {
    auto it = std::max_element(parsedaln.begin(), parsedaln.end(),[](const al_data& a,const al_data& b) { return a.sequence_identity < b.sequence_identity;});

    // for (size_t r = 0; r < parsedaln.size(); r++) {
    //     std::cout << parsedaln[r].read << " " << parsedaln[r].contig << " " << parsedaln[r].pair_dir << " " << parsedaln[r].paired << " " << parsedaln[r].sequence_identity << " complete set \n";
    //     // eachcount << parsedaln[r].read << " " << parsedaln[r].contig << " " << parsedaln[r].pair_dir << " " << parsedaln[r].paired << " " << parsedaln.size() << " " << parsedaln[r].sequence_identity << "\n";
    // }

    if (it->sequence_identity >= sequenceidentity) {
        parsedaln.erase(std::remove_if(parsedaln.begin(),parsedaln.end(), [&](const al_data& a) { return it->sequence_identity > a.sequence_identity;}), parsedaln.end());

        if (parsedaln.size() > 0) {
            unsigned int paired_count = std::count_if(parsedaln.begin(), parsedaln.end(),[](const al_data& a) { return a.paired == true;});
            unsigned int non_paired_count = parsedaln.size() - paired_count;
            float frac_contigs_mapped = 1.0f / parsedaln.size();
            float value_unpaired = 0.0f, value_added_to_paired = 0.0f, value_added_per_paired = 0.0f;

            if (paired_count > 0) {
                // this operation could lead to higher frac_contigs_mapped than
                // total counts of shorter contig obtained from summed value of 
                // value added to paired and unpaired
                value_unpaired = frac_contigs_mapped / 2.0f;
                value_added_to_paired = value_unpaired * non_paired_count;
                value_added_per_paired = value_added_to_paired / paired_count;
            } else {
                // Currently, total assigned value 1 for only single read in a pair
                // 0.5 and 0.5 for each separate alignment of a paired read
                // This is same as perfect paired read mapping to two contigs
                value_unpaired = frac_contigs_mapped; // discordant alignment
            }

            std::vector<std::string> contiglist;
            for (size_t r = 0; r < parsedaln.size(); r++) {
                contiglist.push_back(parsedaln[r].contig);
                auto it = contigs_map.find(parsedaln[r].contig);
                if (parsedaln[r].paired) {
                    it->second = it->second + frac_contigs_mapped + value_added_per_paired;
                    // std::cout << parsedaln[r].read << " " << parsedaln[r].contig << " " << parsedaln[r].pair_dir << " " << parsedaln[r].paired << " " << parsedaln[r].sequence_identity << " " << frac_contigs_mapped + value_added_per_paired << " after filter set \n";
                    eachcount << parsedaln[r].read << " " << parsedaln[r].contig << " " << parsedaln[r].pair_dir << " " << parsedaln[r].paired << " " << parsedaln.size() << " " << frac_contigs_mapped + value_added_per_paired << "\n";
                } else {
                    it->second = it->second + value_unpaired;
                    // std::cout << parsedaln[r].read << " " << parsedaln[r].contig << " " << parsedaln[r].pair_dir << " " << parsedaln[r].paired << " " << parsedaln[r].sequence_identity << " " << value_unpaired << " after filter set \n";
                    eachcount << parsedaln[r].read << " " << parsedaln[r].contig << " " << parsedaln[r].pair_dir << " " << parsedaln[r].paired << " " << parsedaln.size() << " " << value_unpaired << "\n";
                }
            }

            // add shared reads links to contigs
            for (std::size_t i = 0; i < contiglist.size(); ++i) {
                for (std::size_t j = i + 1; j < contiglist.size(); ++j) {
                    auto it1 = contigs_map.find(contiglist[i]);
                    auto it2 = contigs_map.find(contiglist[j]);
                    addpairlinks(contig_links, contiglist[i], contiglist[j], frac_contigs_mapped);
                }
            }
            contiglist.clear(); // clear contiglist data of current read alignments
        }
    } else {
        parsedaln.clear();
    }
}

void fractionate_countlinks(ContigLinks &contiglinks, ContigsMap &contigs_map, std::string &tmp_dir, std::string &outname) {
    std::ofstream linkfile;
    linkfile.open(tmp_dir + '/' + outname + "_countlinks");
    float count_1 = 1.0f, count_2 = 1.0f, normalize_factor = 1.0f;
    for (auto const & k: contiglinks) {
        float frac_links = k.second;
        if (contigs_map.find(k.first.first) != contigs_map.end()) {
            count_1 = contigs_map[k.first.first];
        } else {
            std::cerr << k.first.first << " is not present in contigs_map constructed. Revisit the code \n";
        }
        if (contigs_map.find(k.first.second) != contigs_map.end()) {
            count_2 = contigs_map[k.first.second];
        } else {
            std::cerr << k.first.second << " is not present in contigs_map constructed. Revisit the code \n";
        }
        normalize_factor = (count_1 < count_2) ? count_1 : count_2;
        if (frac_links > normalize_factor) {
            frac_links = normalize_factor;
        }
        frac_links /= normalize_factor;

        // output file format: (i) pair_contig 1, (ii) pair_contig 2, 
        // (iii) fraction of value obtained from the split-mapped read, (iv) iii normalized by the total count of shortest contig in the pair
        linkfile << k.first.first << " " << k.first.second << " " << k.second << " " << frac_links << "\n";
    }
    linkfile.close();
}


// TO DO LIST
// Flags in sam files are considered only as per bowtie2 output. Need to generalize or check compatibility for bwa-mem and minimap2 output
// 

int main(int argc, char *argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    // never use scanf or any library which would use scanf, but it makes input much faster
    std::ios::sync_with_stdio(false);

    if (argc == 1 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        std::cerr << "aligner command | ./aligner2counts working_dir outputname" << "\n";
        return 1;
    }
    std::string tmp_dir = argv[1];
    std::string outname = "";
    if (argv[2]) {
        outname = argv[2];
    }

    // if(argc < 3 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
    //     std::cout << "provide assembly and read files \n";
    //     std::cout << "bam2counts contigs_index reads.fq working_dir threads [default, 1]" << "\n";
    //     return 1;
    // }
    // std::string contigs = argv[1]; 
    // std::string reads = argv[2];
    // std::string tmp_dir = argv[3];
    // std::string threads = "1"; // when unspecified, run as single thread job;

    // if (argv[4]) {
    //     threads = argv[4];
    // }

    unsigned int minlength = 2000;
    std::unordered_map<std::string, float> contigs_map;
    std::unordered_map<std::pair<std::string, std::string>, float, PairHash> contiglinks;

    // std::string command = "bowtie2 -q -a --fr -x " + contigs + " --interleaved " + reads + " -p " + threads; // for interleaved reads;
    
    // FILE* pipe = popen(command.c_str(), "r");

    std::string line;

    // if (!pipe) {
    //     std::cerr << "Failed to open pipe for command: " << command << std::endl;
    //     return 1;
    // }

    // char buffer[1001];
    // buffer[1000] = 0;

    std::vector<al_data> parsedaln;
    std::string currentread_id, contig_id, cigar_str, qual_str, md_str;
    float sequenceidentity = 97.0f;
    unsigned int bitflag;
    currentread_id = "";
    contig_id = "";

    std::ofstream samfile;
    samfile.open(tmp_dir + '/' + outname + ".sam");
    std::ofstream eachcount;
    eachcount.open(tmp_dir + '/' + outname + "_eachcount");

    // while (!feof(pipe)) {
    //     if (fgets(buffer, 1000, pipe)) {
    //         line = buffer;
    while(std::getline(std::cin, line)) {
        samfile << line << "\n"; // write all line before applying any conditions and proceed with continue
        if (line.size() > 0) {
            if (line[0] == '@') {
                if ((line.rfind("@HD",0) == 0)) { // check if input alignment is unsorted
                    if ((line.find("SO:unsorted") == std::string::npos) && line.find("SS") != std::string::npos) {
                        std::cerr << "Input alignment file is sorted. Please provide unsorted file or ordered by queryname\n";
                        return 1;
                    }
                } else { construct_map(contigs_map, line, minlength); };                
            } else {
                if (contigs_map.size() == 0) {
                    std::cerr << "Input sam/bam file doesn't has header. Please provide input file with header \n";
                    return 1;
                }
                std::istringstream iss(line);
                iss >> currentread_id;

                std::string field;
                iss >> bitflag;
                iss >> contig_id;

                if ((contig_id == "*") || (contigs_map.find(contig_id) == contigs_map.end())) {
                    // No reference or not of minimum length, don't process the alignment;
                    continue;
                }

                iss >> field >> field;
                iss >> cigar_str;
                if (cigar_str.find_first_of("IDNP*") != std::string::npos) {
                    // indel, don't process the alignment;
                    if (bitflag & PAIRED_FLAG) {
                        if (!parsedaln.empty()) {
                            parsedaln.erase(std::remove_if(parsedaln.begin(), parsedaln.end(),
                            [contig_id](const al_data& element) {
                                return element.contig == contig_id;
                            }),
                            parsedaln.end());
                        }  
                    }
                    continue;
                }

                iss >> field >> field >> field >> field;
                iss >> qual_str;
                iss >> field >> field;
                if (field.find("XS") != std::string::npos) {
                    // suboptimal alignment score is given;
                    iss >> field >> field >> field >> field >> field >> md_str;
                } else {
                    iss >> field >> field >> field >> field >> md_str;
                }

                md_str = md_str.substr(5, md_str.length());

                auto alnpos = get_alnpos(cigar_str);

                if (std::get<1>(alnpos) == 0) {
                    // clip at both ends, don't process the alignment;
                    continue;
                }

                auto alnstats = get_seqid_alncov(alnpos, qual_str, md_str);

                if ((!std::isnan(alnstats.first)) && (alnstats.second >= 70.0f)) {
                    // get read direction
                    unsigned int pair_flag = 0;
                    if (bitflag & FORWARD_FLAG) {
                        pair_flag = 1;
                    } else if (bitflag & REVERSE_FLAG) {
                        pair_flag = 2;
                    } else {
                        std::cerr << "input reads are not paired\n";
                        return 1;
                    }
                    
                    // get if alignment is paired
                    bool proper_pair = false;
                    if (bitflag & PAIRED_FLAG) {
                        proper_pair = true;
                    } else {;}

                    if (parsedaln.empty()) {
                        parsedaln.push_back({currentread_id, contig_id, pair_flag, proper_pair, false, alnstats.first});
                    } else { // store alignments of current read
                        if (parsedaln.rbegin()->read == currentread_id) {
                            storealnstats(parsedaln,alnstats, currentread_id, contig_id, pair_flag, proper_pair);
                        } else { // new read alignment
                            counting(parsedaln, contigs_map, contiglinks, sequenceidentity, eachcount);
                            parsedaln.clear(); // clean alignment data for current read
                            parsedaln.push_back({currentread_id, contig_id, pair_flag, proper_pair, false, alnstats.first});
                        }
                    }
                }
            }
        }
    }
    // while end
    counting(parsedaln, contigs_map, contiglinks, sequenceidentity, eachcount); // process last read alignment
    parsedaln.clear();

    fractionate_countlinks(contiglinks, contigs_map, tmp_dir, outname);

    std::ofstream outfile;
    outfile.open(tmp_dir + '/' + outname + "_count");
    for (auto const & k: contigs_map) {
        outfile << k.first << " " << outname << " " << k.second << "\n";
    }

    outfile.close();
    samfile.close();
    eachcount.close();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "alignment file processed in " << duration.count() << " seconds\n";

    return EXIT_SUCCESS;
}
