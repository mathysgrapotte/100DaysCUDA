# Load the compiled function
sourceCpp("propr_compare.cpp")

# Create test data
Y <- matrix(runif(80 * 10), nrow=80)  # matches your dimensions

# Run benchmark
result <- benchmark_lrv(Y)

# Print results
print(result)