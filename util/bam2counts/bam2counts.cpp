#include "pybind11/include/pybind11/pybind11.h"
#include <api/BamReader.h>
#include <api/BamMultiReader.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>
#include <dirent.h>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

struct al_data{
    std::string read;
    int32_t contig;
    unsigned int pair_dir; 
    bool paired;
    float sequence_identity;
};

std::pair<float, float> get_seqid_aligncov(BamTools::BamAlignment al) {
    if (al.CigarData.size() <= 2 && al.CigarData.size() != 0 && al.RefID != -1) {
        unsigned int matches = 0;
        unsigned int mismatches = 0;
        std::string mismatch_string;
        al.GetTag("MD", mismatch_string);
        std::stringstream ss(mismatch_string);
        unsigned int future_matches = 0;
        ss >> future_matches;

        unsigned int start = 0;
        if(al.CigarData[0].Type == 'S') {
            start = al.CigarData[0].Length;
        }
        unsigned int end = al.Length;
        // std::cout << al.Length << " length\n";
        if ((al.CigarData.size() > 1) && al.CigarData[1].Type == 'S') {
            end -= al.CigarData[1].Length;
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
            if (al.Qualities[i] >= 20 + 33) {
                if(is_match){
                    matches++;
                } else {
                    mismatches++;
                }
            }
        }

        float seq_id, alignment_coverage;

        seq_id = (float)matches/(float)(matches+mismatches)*100;
        alignment_coverage = (float)alignment_length / (float)al.Length * 100; // account for variable read length
        return std::make_pair(seq_id, alignment_coverage);
    }

    else {
        return std::make_pair(0.0f,0.0f);
    }
}

void construct_umap(const BamTools::BamReader& reader, std::unordered_map<int, float>& umap, const int& minlength, bool flag, std::string& tmp_dir) {
    BamTools::SamHeader header;
    header = reader.GetHeaderText();
    std::stringstream ss(header.ToString());
    std::string to, contig_id;
    std::ofstream selected_contigs;
    
    float init_count = 0.0f;
    if (header.ToString().size() != 0) {
        unsigned int ref_id = 0;
        while(std::getline(ss,to,'\n')) {
            std::string g, contig_id, length;
            std::stringstream st(to);
            st >> g;

            if (g.rfind("@SQ",0) == 0) {
                st >> contig_id;
                st >> length;
                int pos = length.find(":");
                length = length.substr(pos+1);
                
                if (std::stoi(length) >= minlength) { // check length of contigs and insert to umap only if above minimum length for contigs
                    if (!flag) {
                        umap.insert({ref_id, init_count});
                    } else {
                        std::string selected_contigs= tmp_dir + "/selected_contigs";
                        std::fstream writetofile;
                        writetofile.open(selected_contigs, std::fstream::in | std::fstream::out | std::fstream::app);
                        if (!writetofile) {
                            writetofile << ref_id << " " << contig_id.substr(contig_id.find(":")+1,-1) << " " << length << "\n"; // write index, name and length of selected contigs to a file
                        } else {
                            writetofile << ref_id << " " << contig_id.substr(contig_id.find(":")+1,-1) << " " << length << "\n"; // write index, name and length of selected contigs to a file
                        }
                        writetofile.close();    
                    }
                }
                ref_id++;
            }
        }
        ref_id = 0;
    }
    else {
        /* could not parse header of bamfile */
        perror ("Input bamfile is not correct");
        exit(1);
    }
}


void obtain_readcounts(std::string bamfile, const int flag, std::string input_dir, std::string tmp_dir, const int minlength, float sequenceidentity) {

    auto start = std::chrono::high_resolution_clock::now();

    std::unordered_map<int, float> umap;
    
    BamTools::BamReader reader;
    BamTools::BamAlignment aln;
    if (reader.Open(input_dir + "/" + bamfile)) {
        std::stringstream text;
        text << "processing " << bamfile << " to get fractional read counts for contigs\n";
        std::cout << text.str();
        std::cout.flush();
        // unsigned int read_length = 0;

        // while (reader.GetNextAlignment(aln)) {
        //     /* get read length */
        //     if (aln.CigarData.size() == 1 && read_length == 0) {
        //         read_length = aln.Length;
        //     }
        //     std::cout << read_length << " read_length\n";
        //     break;
        // }
        if (flag) {
            construct_umap(reader, umap, minlength, flag, tmp_dir);
        }
        reader.Rewind();
        
        std::vector<al_data> parsed_al;
        size_t lastindex = bamfile.find_last_of("."); 
        std::string out = bamfile.substr(0, lastindex); 
        std::tuple<std::string, std::string, int32_t> assign_qstring;
        std::unordered_map<int, float> umap;

        construct_umap(reader, umap, minlength, false, tmp_dir);

        std::ofstream outfile;
        outfile.open(tmp_dir + "/" + out +"_count");
        // std::ofstream eachcount;
        // eachcount.open(tmp_dir + "/" + out +"_eachcount");

        while (reader.GetNextAlignment(aln)) {
    
        // while start

            if (!aln.Qualities.empty()) {
                assign_qstring = std::make_tuple(aln.Name, aln.Qualities, aln.Length);
            }
            else {
                if (aln.Name == std::get<0>(assign_qstring)) {
                    aln.Qualities = std::get<1>(assign_qstring);
                    aln.Length = std::get<2>(assign_qstring);
                }
            }

            auto check_contig = umap.find(aln.RefID);

            if (check_contig != umap.end()) {
                std::pair<float, float> alignment_stat;
                alignment_stat = get_seqid_aligncov(aln);

                if (alignment_stat.second >= 70.0f) { // filter the alignment by sequence identity (97%) and read coverage (70%)
                    
                    unsigned int pair_flag = 0;

                    if (aln.IsFirstMate()) { // set flag for direction of the read pair
                        pair_flag = 1;
                    }
                    else if (aln.IsSecondMate()) {
                        pair_flag = 2;
                    } else {};

                    if (parsed_al.empty()) {
                        parsed_al.push_back({aln.Name, aln.RefID, pair_flag, aln.IsProperPair(), alignment_stat.first});
                    }
                    
                    else {
                        if (parsed_al.rbegin()->read == aln.Name) { // check read name is same
                            auto ic = std::find_if(parsed_al.rbegin(), parsed_al.rend(),[&](const al_data& a) {return a.contig == aln.RefID;});
                            if (ic != parsed_al.rend()){ 
                                if (ic->pair_dir != pair_flag){ // if mate pair alignment is found
                                    ic->sequence_identity = (ic->sequence_identity + alignment_stat.first) / 2.0f;
                                    ic->paired = aln.IsPaired();
                                }
                                else { // update sequence identity if the same read (of the same direction) mapped to the same contig with higher sequence identity
                                    if (ic->sequence_identity < alignment_stat.first) {
                                        ic->sequence_identity = alignment_stat.first; 
                                    } else {continue;}
                                }
                            }
                            else {
                                if (aln.IsProperPair() == 1) { // if read mapped to different contig, add to the vector
                                    parsed_al.push_back({aln.Name, aln.RefID, pair_flag, aln.IsProperPair(), alignment_stat.first});;
                                }
                                else { // if read mapped to different contig and mate pair didn't mapped, add to the vector only sequence identity is greater than or equal 
                                    if (parsed_al.rbegin()->sequence_identity <= alignment_stat.first) {
                                        parsed_al.push_back({aln.Name, aln.RefID, pair_flag, aln.IsProperPair(), alignment_stat.first});     
                                    } else {continue;}
                                }
                            }  
                        }

                        else { // new read alignment

                            auto it = std::max_element(parsed_al.begin(), parsed_al.end(),[](const al_data& a,const al_data& b) { return a.sequence_identity < b.sequence_identity;});
                            
                            if (it->sequence_identity >= sequenceidentity) {
                                parsed_al.erase(std::remove_if(parsed_al.begin(),parsed_al.end(), [&](const al_data& a) {return it->sequence_identity > a.sequence_identity;}), parsed_al.end());

                                if (parsed_al.size() > 0) {
                                    // for (size_t j = 0; j < parsed_al.size(); j++) {
                                    //     eachcount << sequenceidentity << " " << parsed_al[j].read << " " << parsed_al[j].contig << " " << parsed_al[j].pair_dir << " " << parsed_al[j].paired << " " << parsed_al.size() << " " << parsed_al[j].sequence_identity << "\n";
                                    // }
                                    unsigned int paired_count = std::count_if(parsed_al.begin(), parsed_al.end(),[](const al_data& a) { return a.paired == 1;});
                                    unsigned int non_paired_count = parsed_al.size() - paired_count;

                                    float val1 = 1.0f / parsed_al.size() ;
                                    float val2 = val1 / 2.0f;
                                    float val3 = val2 * non_paired_count;
                                    float val4 = val3 / paired_count;
                                    for (size_t r = 0; r < parsed_al.size(); r++) {
                                        auto it = umap.find(parsed_al[r].contig);
                                        if (parsed_al[r].paired) {
                                            it->second = it->second + val1 + val4 ;
                                        }
                                        else {
                                            it->second = it->second + val2;
                                        }
                                        
                                    }
                                }
                            }
                            parsed_al.clear();
                            parsed_al.push_back({aln.Name, aln.RefID, pair_flag, aln.IsProperPair(), alignment_stat.first});
                        }
                    }

                }
            }
        
        }
        // while end

        parsed_al.clear();
        for (auto const & k: umap) {
            outfile << k.first << " " << out << " " << k.second << "\n";
        }
        outfile.close();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        std::stringstream text_out;
        text_out << "completed writing counts for sample " << out << " in " << duration.count() << " seconds\n";
        std::cout << text_out.str();
        std::cout.flush();
    }

    else {
        throw std::invalid_argument("Input error! please check input");
    }
    umap.clear();
    reader.Close();
}

namespace py = pybind11;

PYBIND11_MODULE(bam2counts, m) {
m.def("obtain_readcounts", &obtain_readcounts, "obtain fractional read counts from each metagenomic sample", py::arg("bamfile"), py::arg("flag"), py::arg("input_dir"), py::arg("tmp_dir"), py::arg("minlength"), py::arg("sequence_identity"));
}
