#include <api/BamReader.h>
#include <api/BamMultiReader.h>
#include <api/BamWriter.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <dirent.h>

int main(int argc, char* argv[]) {
   
    std::string directory = argv[1];
    unsigned int contig_length_cutoff = std::stoi(argv[2]);
    std::string tmp = argv[3];
    std::string bamfile, contig_id;
    auto start = std::chrono::high_resolution_clock::now();
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(directory)) != NULL) {
        /* collect all bam files in the directory */
        while ((ent = readdir(dir)) != NULL) {
            if(std::strstr(ent->d_name, ".bam")) {
                bamfile = ent->d_name;
                break;
            }
        }
        closedir (dir);
    } else {
    /* could not open directory */
        perror ("Input folder doesn't have bam files");
        return EXIT_FAILURE;
    }
    
    BamTools::BamReader reader;
    BamTools::SamHeader header;
    std::stringstream ss(header.ToString());

    std::ofstream selected_contigs;
    selected_contigs.open(tmp + "selected_contigs");
    if (reader.Open(directory + bamfile)) {
        header = reader.GetHeaderText();
        if (header.ToString().size() != 0) {
            unsigned int ref_id = 0;
            std::string line;
            std::string g, length;
            ss >> line;
            while(line.rfind("@SQ",0) == 0) {
                
                std::stringstream st(line);
                st >> g;

                // if (g.rfind("@SQ",0) == 0) {
                st >> g;
                contig_id = g;
                int pos = g.find("C");
                g = g.substr(pos+1);
                st >> length;
                pos = length.find(":");
                length = length.substr(pos+1);
                if (std::stoi(length) >= contig_length_cutoff) { // check length of contigs and insert to umap only if above contig_size_cutoff 
                    selected_contigs << ref_id << " " << contig_id.substr(contig_id.find(":")+1,-1) << " " << length << "\n";
                }
                ref_id++;
                // }
                ss >> line;
            }
        } else {
            /* could not parse header of bamfile */
            perror ("Input bamfile is not correct");
            return EXIT_FAILURE;
        }
    } else {
            /* could not open bamfile */
            perror ("Unable to open input bamfile. Please check if bamfile exist it directory");
            return EXIT_FAILURE;
    }
    selected_contigs.close();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << duration.count() << " seconds\n";
    return EXIT_SUCCESS;
}