#include <immintrin.h>
#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Vectorized weighted moving average (Fused Multiply-Add) using AVX2 SIMD instructions
float avx2_weighted_forecast(const float* demand, const float* weights, int size) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    
    // Process 8 floats at a time using AVX2 registers
    for (; i <= size - 8; i += 8) {
        __m256 d = _mm256_loadu_ps(&demand[i]);
        __m256 w = _mm256_loadu_ps(&weights[i]);
        sum = _mm256_fmadd_ps(d, w, sum); // Fused Multiply-Add: (d * w) + sum
    }
    
    // Horizontal addition of the 8 float elements inside the AVX2 register
    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, sum);
    float total = temp[0] + temp[1] + temp[2] + temp[3] + 
                  temp[4] + temp[5] + temp[6] + temp[7];
    
    // Handle remaining elements if size is not a multiple of 8
    for (; i < size; ++i) {
        total += demand[i] * weights[i];
    }
    
    return total;
}

// Function to process a batch of SKUs
// demand_matrix is a 2D numpy array (num_items x history_len)
// weights is a 1D numpy array of length history_len
py::array_t<float> forecast_batch(py::array_t<float> demand_matrix, py::array_t<float> weights) {
    py::buffer_info demand_info = demand_matrix.request();
    py::buffer_info weight_info = weights.request();
    
    if (demand_info.ndim != 2) {
        throw std::runtime_error("Demand matrix must be a 2D array.");
    }
    
    int num_items = demand_info.shape[0];
    int history_len = demand_info.shape[1];
    
    if (weight_info.size != history_len) {
        throw std::runtime_error("Weights array size must match demand history length.");
    }

    // Allocate result array
    auto result = py::array_t<float>(num_items);
    py::buffer_info result_info = result.request();
    
    float* demand_ptr = static_cast<float*>(demand_info.ptr);
    float* weight_ptr = static_cast<float*>(weight_info.ptr);
    float* result_ptr = static_cast<float*>(result_info.ptr);
    
    // Parallelize SKU batch calculations across available threads using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < num_items; ++i) {
        result_ptr[i] = avx2_weighted_forecast(&demand_ptr[i * history_len], weight_ptr, history_len);
    }
    
    return result;
}

PYBIND11_MODULE(forecaster, m) {
    m.doc() = "AVX2 SIMD Accelerated Inventory Forecasting Kernel";
    m.def("forecast_batch", &forecast_batch, "Run batch forecasting using optimized AVX2 SIMD operations");
}
