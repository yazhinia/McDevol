#include<cmath>
#include<iostream>

#include "loggamma.h"

int main() {
   

    float min_value = 0.2;
    float max_value = 1e6;

    double max_abs_err = -10;
    float where = -10;

    float x = min_value;

    while(x <= max_value) {
        double real = std::lgamma((double)x);
        double approx = log_gamma<float, true>(x);
        double abs_err = std::abs(real - approx);
        
        
		if(abs_err > max_abs_err) {
			max_abs_err = abs_err;
			where = x;
		}
      

        x = std::nextafter<float>(x, max_value + 1e10f);

        
    }

    std::cout << "Max abs error: " << max_abs_err << std::endl;
    std::cout << "Where: " << where << std::endl;
}
