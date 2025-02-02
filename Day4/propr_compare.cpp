#include <Rcpp.h>
#include <math.h>
#include <chrono>  // Add this for chrono
using namespace Rcpp;

// Function to compare results between implementations
bool compare_results(NumericVector rcpp_result, float* cpu_result, int length) {
    double max_diff = 0.0;
    for(int i = 0; i < length; i++) {
        double diff = std::abs(rcpp_result[i] - cpu_result[i]);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff < 1e-5; // Allow small floating point differences
}

// Rcpp implementation using built-in var() function
NumericVector lrv(NumericMatrix & Y) {
    // Output a half-matrix
    NumericMatrix X = clone(Y);
    int nfeats = X.ncol();
    int llt = nfeats * (nfeats - 1) / 2;
    NumericVector result(llt);
    int counter = 0;

    for(int i = 1; i < nfeats; i++) {
        for(int j = 0; j < i; j++) {
            result(counter) = var(log(X(_, i) / X(_, j)));
            counter += 1;
        }
    }
    
    return result;
}

// CPU implementation with manual variance calculation
float* compute_log_variance_ratio_cpu(const float* Y, int nb_samples, int nb_genes) {
    // Output array to store variances for each pair
    int num_pairs = (nb_genes * (nb_genes - 1)) / 2;
    float* variances = new float[num_pairs];
    int counter = 0;

    // For each pair of genes
    for(int i = 1; i < nb_genes; i++) {
        for(int j = 0; j < i; j++) {
            float mean = 0.0f;
            float variance = 0.0f;
            
            // First pass: compute mean of log ratios
            for(int k = 0; k < nb_samples; k++) {
                float ratio = Y[k + i * nb_samples] / Y[k + j * nb_samples];
                mean += log(ratio);
            }
            mean /= nb_samples;
            
            // Second pass: compute variance
            for(int k = 0; k < nb_samples; k++) {
                float ratio = Y[k + i * nb_samples] / Y[k + j * nb_samples];
                float diff = log(ratio) - mean;
                variance += diff * diff;
            }
            
            // Divide by (N-1) for sample variance
            variances[counter] = variance / (nb_samples - 1);
            counter++;
        }
    }

    return variances;
}

// Benchmark function to compare implementations
// [[Rcpp::export]]
List benchmark_lrv(NumericMatrix Y) {
    // Time Rcpp implementation
    auto start_rcpp = std::chrono::high_resolution_clock::now();
    NumericVector rcpp_result = lrv(Y);
    auto end_rcpp = std::chrono::high_resolution_clock::now();
    auto duration_rcpp = std::chrono::duration_cast<std::chrono::microseconds>(end_rcpp - start_rcpp);
    
    // Convert NumericMatrix to float array for CPU implementation
    int nb_samples = Y.nrow();
    int nb_genes = Y.ncol();
    float* Y_float = new float[nb_samples * nb_genes];
    for(int i = 0; i < nb_genes; i++) {
        for(int j = 0; j < nb_samples; j++) {
            Y_float[j + i * nb_samples] = Y(j, i);
        }
    }
    
    // Time CPU implementation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float* cpu_result = compute_log_variance_ratio_cpu(Y_float, nb_samples, nb_genes);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    // Compare results
    int num_pairs = (nb_genes * (nb_genes - 1)) / 2;
    bool results_match = compare_results(rcpp_result, cpu_result, num_pairs);
    
    // Clean up
    delete[] Y_float;
    delete[] cpu_result;
    
    // Return timing results and comparison
    List result = List::create(
        Named("rcpp_time_us") = duration_rcpp.count(),
        Named("cpu_time_us") = duration_cpu.count(),
        Named("results_match") = results_match
    );
    
    // Convert List elements to double before printing
    double rcpp_time = as<double>(result["rcpp_time_us"]);
    double cpu_time = as<double>(result["cpu_time_us"]);
    bool match = as<bool>(result["results_match"]);
    
    Rcout << "Rcpp time: " << rcpp_time << " us" << std::endl;
    Rcout << "CPU time: " << cpu_time << " us" << std::endl;
    Rcout << "Results match: " << match << std::endl;
    
    return result;
}
