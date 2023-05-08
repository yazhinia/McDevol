#include "simd_T/simd.h"

#include <cmath>
#include <cstring>

const double LS2PI = 0.91893853320467274178;

template<typename F, bool one_over_12 = true>
F log_gamma(F x) {
	F x_p_1 = x + 1;
	//Stirling gamma(x+2)
	F q = (x_p_1 + (F)0.5) * std::log(x_p_1) - x_p_1 + (F)LS2PI;
	
	if constexpr(one_over_12) {
		q += (F)1.0/((F)12.0*x_p_1);
	}
	
	return q - std::log(x*(x+1)); // correction for shifting by 2
}

template<typename F, bool one_over_12 = true>
typename SIMD<F>::type log_gamma_register(typename SIMD<F>::type x) {
    auto one = SIMD<F>::set(1.0);
    auto half = SIMD<F>::set(0.5);
    auto ls2pi = SIMD<F>::set((F)LS2PI);
    auto x_plus_1 = SIMD<F>::add(one, x);
    auto log_x_plus_1 = SIMD<F>::log(x_plus_1);
    auto x_plus_one_half = SIMD<F>::add(x_plus_1, half);
    auto q = SIMD<F>::mul(x_plus_one_half, log_x_plus_1);
    q = SIMD<F>::sub(q, x_plus_1);
    q = SIMD<F>::add(q, ls2pi);
    
    if constexpr(one_over_12) {
		auto rcp = SIMD<F>::rcp(x_plus_1);
		auto rcp_12 = SIMD<F>::set((F)1.0 / (F)12.0);
		q = SIMD<F>::add(q, SIMD<F>::mul(rcp, rcp_12));
	}
    
    auto prod = SIMD<F>::mul(x, x_plus_1);
    auto log_prod = SIMD<F>::log(prod);
    
    return SIMD<F>::sub(q, log_prod);
}

template<typename F, bool one_over_12 = true>
void log_gamma_simd(F *input, F *output, size_t L) {
    size_t simd_blocks = L / SIMD<F>::count;

    size_t index = 0;
    for(; index < simd_blocks; index++) {
        auto data = SIMD<F>::loadU(input + index * SIMD<F>::count);
        auto result = log_gamma_register<F, one_over_12>(data);
        SIMD<F>::store(output + index * SIMD<F>::count, result);
    }

    //Do the last few elements

    size_t remainder_elements = L % SIMD<F>::count;
    auto data = SIMD<F>::set(1.0);
    std::memcpy(&data, input + index * SIMD<F>::count, remainder_elements * sizeof(F));
    data = log_gamma_register<F, one_over_12>(data);
    std::memcpy(output + index * SIMD<F>::count, &data, remainder_elements * sizeof(F));
}

template<typename F, bool one_over_12 = true>
F log_gamma_simd_wrapper(F x) {
	auto reg = SIMD<F>::set(x);
	auto result = log_gamma_register<F, one_over_12>(reg);
	return SIMD<F>::extract_first(result);
}
