#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include "simd_T/simd.h"

#include "loggamma.h"


template<typename F>
typename SIMD<F>::type log_product_reg(typename SIMD<F>::type x, typename SIMD<F>::type x_prime,typename SIMD<F>::type alpha) {
	auto x_sum = SIMD<F>::add(alpha, x);
	auto x_prime_sum = SIMD<F>::add(alpha, x_prime);
	auto both_x_sum = SIMD<F>::add(x_sum, x_prime);
	auto lgamma_x_sum = log_gamma_register<F>(x_sum);
	auto lgamma_x_prime_sum = log_gamma_register<F>(x_prime_sum);
	auto lgamma_both_x_sum = log_gamma_register<F>(both_x_sum);
	auto sum = SIMD<F>::add(lgamma_x_sum, lgamma_x_prime_sum);
	return SIMD<F>::sub(sum, lgamma_both_x_sum);

}

// Calculates log prod (gamma(x_n + a_n) gamma(x_prime_n + a_n)) / gamma(xn + x_prime_n + a_n)
template<typename F>
F log_product(F *x, F *x_prime, F *alpha, size_t N) {
	size_t simd_blocks = N / SIMD<F>::count;

	auto sum = SIMD<F>::set((F)0.0);

    size_t index = 0;
    for(; index < simd_blocks; index++) {
        auto x_reg = SIMD<F>::loadU(x + index * SIMD<F>::count);
        auto x_prime_reg = SIMD<F>::loadU(x_prime + index * SIMD<F>::count);
        auto alpha_reg = SIMD<F>::loadU(alpha + index * SIMD<F>::count);
        
        sum = SIMD<F>::add(sum, log_product_reg<F>(x_reg, x_prime_reg, alpha_reg));
    }
    
    

    //Do the last few elements

    size_t remainder_elements = N % SIMD<F>::count;
    
    
    auto x_reg = SIMD<F>::set((F)1.0);
    auto x_prime_reg = SIMD<F>::set((F)1.0);
    auto alpha_reg = SIMD<F>::set((F)1.0);
    std::memcpy(&x_reg, x + index * SIMD<F>::count, remainder_elements * sizeof(F));
    std::memcpy(&x_prime_reg, x_prime + index * SIMD<F>::count, remainder_elements * sizeof(F));
    std::memcpy(&alpha_reg, alpha + index * SIMD<F>::count, remainder_elements * sizeof(F));
    
    auto sum_remainder = log_product_reg<F>(x_reg, x_prime_reg, alpha_reg);
    
    return SIMD<F>::hsum(sum) + SIMD<F>::partial_hsum(sum_remainder, remainder_elements);
}

// // log gamma for high counts
// template<typename F>
// F std_log_gamma_product(F *R, F *R_prime, double alpha_sum, size_t N) {
//     F output=0.0f;
// 	for(unsigned j = 0; j < N; j++) {
// 		double lgamma_R_alpha = std::lgamma(R[j] + 15.048f);
// 		double lgamma_R_prime_alpha = std::lgamma(R_prime[j] + 15.048f);
//         double lgamma_R_R_prime_alpha = std::lgamma(R[j] + R_prime[j] + 15.048f);
// 		output += lgamma_R_alpha + lgamma_R_prime_alpha - lgamma_R_R_prime_alpha;
//     }
//     return output;
// }


// Calculate log of sum of exponentials

double calc_logsumexp(double x, double y) {
   double _min = std::min(x,y);
   double _max = std::max(x,y);

    //_max it much bigger than _min, so we return just _max
    if (_max > _min + 350) {
        return _max;
    }

    else {
      return _max + std::log(std::exp(_min - _max) + 1.0f);
    }

}

// Calculate log gamma of dirichlet priors
template<typename F>
F calc_sumdirichletlgamma(F *alpha, size_t N) {
    double sum_dirichlet_lgamma = 0.0f;
    for (size_t n = 0; n < N; n++) {
        sum_dirichlet_lgamma += log_gamma(alpha[n]);
    }
    return sum_dirichlet_lgamma;
}

template<typename F>
F calc_lgammadirichletsum(F *alpha, size_t N) {
    double lgamma_dirichlet_sum = 0.0f;
    for (size_t n = 0; n < N; n++) {
        lgamma_dirichlet_sum += alpha[n];
    }
    return log_gamma(lgamma_dirichlet_sum);
}


namespace py = pybind11;

py::array_t<double> compute_readcountdist(size_t query_index, py::array_t<double, py::array::c_style> &read_counts, py::array_t<double, py::array::c_style> &R_c, double alpha_sum, py::array_t<double, py::array::c_style> &alpha, double q_read) {
    
    py::buffer_info read_counts_buf = read_counts.request();
    py::buffer_info R_c_buf = R_c.request();
    py::buffer_info alpha_buf = alpha.request();

    size_t N = read_counts_buf.shape[1];
    size_t C = read_counts_buf.shape[0];

    auto read_counts_ptr = static_cast<double*>(read_counts_buf.ptr);
    auto R_c_ptr = static_cast<double*>(R_c_buf.ptr);
    auto alpha_ptr = static_cast<double*>(alpha_buf.ptr);

    auto distance = py::array_t<double>(C);
    py::buffer_info distance_buf = distance.request();
    auto distance_ptr = static_cast<double*>(distance_buf.ptr);
    
//    double q_read = std::exp(-8.0);

    // obtain log gamma of dirichlet priors
    double sum_dirichlet_lgamma_reads = calc_sumdirichletlgamma(alpha_ptr, N);
    double lgamma_dirichlet_sum_reads = calc_lgammadirichletsum(alpha_ptr, N);

    auto current_contig_read = read_counts_ptr + query_index * N;
    auto alpha_sum_ptr = &alpha_sum;

    #pragma omp parallel for
    for (size_t c1 = 0; c1 < C; c1++) {

        auto prime_contig_read = read_counts_ptr + c1 * N;

        double term_1 = log_product(current_contig_read, prime_contig_read, alpha_ptr, N) - sum_dirichlet_lgamma_reads;
        double term_2 = log_product(R_c_ptr + query_index, R_c_ptr + c1, alpha_sum_ptr, 1) - lgamma_dirichlet_sum_reads;
        double distance_reads = calc_logsumexp(std::log(1.0f), term_1-term_2 + std::log((1.0f-q_read)/q_read));
        
        distance_ptr[c1] = distance_reads;

    }
    return distance;

}


py::array_t<double> compute_kmercountdist(size_t query_index, py::array_t<double, py::array::c_style> &kmer_counts, py::array_t<double, py::array::c_style> &R_k, py::array_t<double, py::array::c_style> &alpha_kmers, py::array_t<double, py::array::c_style> &alpha_perkmers, double q_kmer) {
    
    py::buffer_info kmer_counts_buf = kmer_counts.request();
    py::buffer_info R_k_buf = R_k.request();
    py::buffer_info alpha_kmers_buf = alpha_kmers.request();
    py::buffer_info alpha_perkmers_buf = alpha_perkmers.request();

    int tetramer_count = 256;
    int trimer_count = 64;

    size_t C = kmer_counts_buf.shape[0];

    if ((size_t)kmer_counts_buf.size != C * tetramer_count) {
        throw std::runtime_error("numpy array of kmer counts does not match with contigs * 256 tetramers dimension");
    }

    auto kmer_counts_ptr = static_cast<double*>(kmer_counts_buf.ptr);
    auto R_k_ptr = static_cast<double*>(R_k_buf.ptr);
    auto alpha_kmers_ptr = static_cast<double*>(alpha_kmers_buf.ptr);
    auto alpha_perkmers_ptr = static_cast<double*>(alpha_perkmers_buf.ptr);

    auto distance = py::array_t<double>(C);
    py::buffer_info distance_buf = distance.request();
    auto distance_ptr = static_cast<double*>(distance_buf.ptr);

//    double q_kmer = 0.6;

    // obtain log gamma of dirichlet priors
    double sum_dirichlet_lgamma_perkmers = calc_sumdirichletlgamma(alpha_perkmers_ptr, tetramer_count);
    double sum_dirichlet_lgamma_kmers = calc_sumdirichletlgamma(alpha_kmers_ptr, trimer_count);

    // std::cout << "sum_dirichlet_lgamma_perkmers " << sum_dirichlet_lgamma_perkmers << " sum_dirichlet_lgamma_kmers " << sum_dirichlet_lgamma_kmers << "\n";
    auto current_contig_kmer = kmer_counts_ptr + query_index * tetramer_count;
    auto current_contig_R_k = R_k_ptr + query_index * trimer_count;

    #pragma omp parallel for
    for (size_t c1 = 0; c1 < C; c1++) {

        auto prime_contig_kmer = kmer_counts_ptr + c1 * tetramer_count;
        auto prime_contig_R_k = R_k_ptr + c1 * trimer_count;        

        double term_1 = log_product(current_contig_kmer, prime_contig_kmer, alpha_perkmers_ptr, tetramer_count) - sum_dirichlet_lgamma_perkmers;
        double term_2 = log_product(current_contig_R_k, prime_contig_R_k, alpha_kmers_ptr, trimer_count) - sum_dirichlet_lgamma_kmers;
        double distance_kmers = calc_logsumexp(std::log(1.0), term_1-term_2 + std::log((1.0-q_kmer)/q_kmer));
        distance_ptr[c1] = distance_kmers;
    }
    return distance;

}


py::array_t<double> compute_dist(size_t query_index, py::array_t<double, py::array::c_style> &read_counts, py::array_t<double, py::array::c_style> &kmer_counts, py::array_t<double, py::array::c_style> &R_c, py::array_t<double, py::array::c_style> &R_k,double alpha_sum, py::array_t<double, py::array::c_style> &alpha, py::array_t<double, py::array::c_style> &alpha_kmers, py::array_t<double, py::array::c_style> &alpha_perkmers, double q_read, double q_kmer){
    
    py::buffer_info read_counts_buf = read_counts.request();
    py::buffer_info kmer_counts_buf = kmer_counts.request();
    py::buffer_info R_c_buf = R_c.request();
    py::buffer_info R_k_buf = R_k.request();
    py::buffer_info alpha_buf = alpha.request();    
    py::buffer_info alpha_kmers_buf = alpha_kmers.request();
    py::buffer_info alpha_perkmers_buf = alpha_perkmers.request();

    int tetramer_count = 256;
    int trimer_count = 64;


    size_t N = read_counts_buf.shape[1];
    size_t C = read_counts_buf.shape[0];

    if ((size_t)kmer_counts_buf.size != C * tetramer_count) {
        throw std::runtime_error("numpy array of kmer counts does not match with contigs * 256 tetramers dimension");
    }
    
    auto read_counts_ptr = static_cast<double*>(read_counts_buf.ptr);
    auto kmer_counts_ptr = static_cast<double*>(kmer_counts_buf.ptr);
    auto R_c_ptr = static_cast<double*>(R_c_buf.ptr);
    auto R_k_ptr = static_cast<double*>(R_k_buf.ptr);
    auto alpha_ptr = static_cast<double*>(alpha_buf.ptr);
    auto alpha_kmers_ptr = static_cast<double*>(alpha_kmers_buf.ptr);
    auto alpha_perkmers_ptr = static_cast<double*>(alpha_perkmers_buf.ptr);

    auto distance = py::array_t<double>(C);
    py::buffer_info distance_buf = distance.request();
    auto distance_ptr = static_cast<double*>(distance_buf.ptr);
    
//    double q_read = std::exp(-8.0);
//    double q_kmer = std::exp(-8.0);

    // obtain log gamma of dirichlet priors
    double sum_dirichlet_lgamma_reads = calc_sumdirichletlgamma(alpha_ptr, N);
    double lgamma_dirichlet_sum_reads = calc_lgammadirichletsum(alpha_ptr, N);
    double sum_dirichlet_lgamma_perkmers = calc_sumdirichletlgamma(alpha_perkmers_ptr, tetramer_count);
    double sum_dirichlet_lgamma_kmers = calc_sumdirichletlgamma(alpha_kmers_ptr, trimer_count);

    auto current_contig_read = read_counts_ptr + query_index * N;
    auto alpha_sum_ptr = &alpha_sum;
    auto current_contig_kmer = kmer_counts_ptr + query_index * tetramer_count;
    auto current_contig_R_k = R_k_ptr + query_index * trimer_count;

    #pragma omp parallel for
    for (size_t c1 = 0; c1 < C; c1++) {

        auto prime_contig_read = read_counts_ptr + c1 * N;
        auto prime_contig_kmer = kmer_counts_ptr + c1 * tetramer_count;
        auto prime_contig_R_k = R_k_ptr + c1 * trimer_count;
        
        double term_1 = log_product(current_contig_read, prime_contig_read, alpha_ptr, N) - sum_dirichlet_lgamma_reads;
        double term_2 = log_product(R_c_ptr + query_index, R_c_ptr + c1, alpha_sum_ptr, 1) - lgamma_dirichlet_sum_reads;
        double distance_reads = calc_logsumexp(std::log(1.0), term_1-term_2 + std::log((1.0-q_read)/q_read));

        double term_3 = log_product(current_contig_kmer, prime_contig_kmer, alpha_perkmers_ptr, tetramer_count) - sum_dirichlet_lgamma_perkmers;
        double term_4 = log_product(current_contig_R_k, prime_contig_R_k, alpha_kmers_ptr, trimer_count) - sum_dirichlet_lgamma_kmers;
        double distance_kmers = calc_logsumexp(std::log(1.0), term_3-term_4 + std::log((1.0-q_kmer)/q_kmer));

        distance_ptr[c1] = distance_reads + distance_kmers;
    }
    return distance;
}

template<typename F>
py::array_t<F> log_gamma_avx2(const py::array_t<F, py::array::c_style> &x) {
    py::buffer_info buffer_inf = x.request();

    size_t L = buffer_inf.size;

    auto *output = allocAligned<F>(L);
    py::capsule free_when_done(output, [](void *f) {
        auto foo = reinterpret_cast<F*>(f);
        freeAligned(foo);
    });
    auto ptr = reinterpret_cast<F*>(buffer_inf.ptr);
    log_gamma_simd(ptr, output, L);
    return py::array_t<F>(buffer_inf.shape, buffer_inf.strides, output, free_when_done);
}

PYBIND11_MODULE(metadevol_distance, m) {
//Init gamma precalc
//calc_precalc<float>();

m.def("compute_dist", &compute_dist, "compute distance between contigs", py::arg("query_index"), py::arg("read_counts"), py::arg("kmer_counts"), py::arg("R_c"), py::arg("R_k"), py::arg("alpha"), py::arg("alpha_sum"), py::arg("alpha_kmers"), py::arg("alpha_perkmers"), py::arg("q_read"), py::arg("q_kmer"));
m.def("compute_readcountdist", &compute_readcountdist, "compute distance between contigs using read counts", py::arg("query_index"), py::arg("read_counts"), py::arg("R_c"), py::arg("alpha"), py::arg("alpha_sum"), py::arg("q_read"));
m.def("compute_kmercountdist", &compute_kmercountdist, "compute distance between contigs using kmer counts", py::arg("query_index"), py::arg("kmer_counts"), py::arg("R_k"), py::arg("alpha_kmers"), py::arg("alpha_perkmers"), py::arg("q_kmer"));
m.def("log_gamma_avx2", &log_gamma_avx2<double>, "compute log gamma of an array", py::arg("arrays"));
}

