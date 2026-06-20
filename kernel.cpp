#include <immintrin.h>
#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Vectorized Double Exponential Smoothing (Holt's Linear Trend Model) using AVX2 SIMD
// Processes 8 SKUs in parallel (Structure of Arrays approach)
void avx2_holt_forecast_8x(
    const float* demands_flat, // Array of size 8 * history_len (demands for 8 SKUs interleaved or structured)
    int history_len,
    float alpha,
    float beta,
    float forecast_horizon,
    float* results             // Outputs for 8 SKUs
) {
    // Coefficients as vector constants
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 one_minus_alpha_vec = _mm256_set1_ps(1.0f - alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    __m256 one_minus_beta_vec = _mm256_set1_ps(1.0f - beta);
    __m256 horizon_vec = _mm256_set1_ps(forecast_horizon);

    // Initialize levels (L_0) to first demand points, and trends (T_0) to zero
    // demands_flat layout: demands[sku_index * history_len + time_index]
    alignas(32) float initial_demands[8];
    alignas(32) float second_demands[8];
    for (int sku = 0; sku < 8; ++sku) {
        initial_demands[sku] = demands_flat[sku * history_len + 0];
        second_demands[sku] = demands_flat[sku * history_len + 1];
    }

    __m256 level = _mm256_loadu_ps(initial_demands);
    __m256 y_1 = _mm256_loadu_ps(second_demands);
    
    // Initial trend estimate (T_0 = Y_1 - Y_0)
    __m256 trend = _mm256_sub_ps(y_1, level);

    // Run Holt's recurrence relation over time series
    for (int t = 1; t < history_len; ++t) {
        alignas(32) float current_y[8];
        for (int sku = 0; sku < 8; ++sku) {
            current_y[sku] = demands_flat[sku * history_len + t];
        }
        __m256 y = _mm256_loadu_ps(current_y);

        // L_t = alpha * Y_t + (1 - alpha) * (L_{t-1} + T_{t-1})
        __m256 next_level = _mm256_fmadd_ps(
            alpha_vec, 
            y, 
            _mm256_mul_ps(one_minus_alpha_vec, _mm256_add_ps(level, trend))
        );

        // T_t = beta * (L_t - L_{t-1}) + (1 - beta) * T_{t-1}
        __m256 level_diff = _mm256_sub_ps(next_level, level);
        __m256 next_trend = _mm256_fmadd_ps(
            beta_vec, 
            level_diff, 
            _mm256_mul_ps(one_minus_beta_vec, trend)
        );

        level = next_level;
        trend = next_trend;
    }

    // Forecast: F_{t+m} = L_t + m * T_t
    __m256 forecast = _mm256_fmadd_ps(horizon_vec, trend, level);
    _mm256_storeu_ps(results, forecast);
}

// pybind11 module function to process a batch of SKUs
// demand_matrix shape: (num_items, history_len)
py::array_t<float> forecast_batch_double_exponential(
    py::array_t<float> demand_matrix, 
    float alpha, 
    float beta, 
    float forecast_horizon
) {
    py::buffer_info demand_info = demand_matrix.request();
    if (demand_info.ndim != 2) {
        throw std::runtime_error("Demand matrix must be a 2D array.");
    }

    int num_items = demand_info.shape[0];
    int history_len = demand_info.shape[1];
    
    if (history_len < 2) {
        throw std::runtime_error("History length must be at least 2 steps for trend analysis.");
    }

    auto result = py::array_t<float>(num_items);
    py::buffer_info result_info = result.request();

    float* demand_ptr = static_cast<float*>(demand_info.ptr);
    float* result_ptr = static_cast<float*>(result_info.ptr);

    // Process blocks of 8 SKUs using OpenMP threads and AVX2 registers
    #pragma omp parallel for
    for (int i = 0; i < num_items; i += 8) {
        int rem = num_items - i;
        if (rem >= 8) {
            // Run optimized AVX2 SIMD Holt forecast for 8 parallel SKUs
            avx2_holt_forecast_8x(
                &demand_ptr[i * history_len], 
                history_len, 
                alpha, 
                beta, 
                forecast_horizon, 
                &result_ptr[i]
            );
        } else {
            // Scalar fallback cleanup for remaining SKUs
            for (int r = 0; r < rem; ++r) {
                int idx = i + r;
                float* sku_demand = &demand_ptr[idx * history_len];
                
                float level = sku_demand[0];
                float trend = sku_demand[1] - sku_demand[0];
                
                for (int t = 1; t < history_len; ++t) {
                    float next_level = alpha * sku_demand[t] + (1.0f - alpha) * (level + trend);
                    float next_trend = beta * (next_level - level) + (1.0f - beta) * trend;
                    level = next_level;
                    trend = next_trend;
                }
                
                result_ptr[idx] = level + forecast_horizon * trend;
            }
        }
    }

    return result;
}

PYBIND11_MODULE(forecaster, m) {
    m.doc() = "Enterprise Grade AVX2 Double Exponential Smoothing Forecast Kernel";
    m.def("forecast_batch", &forecast_batch_double_exponential, 
          "Run Double Exponential (Holt's Linear) batch forecast using AVX2 SIMD operations",
          py::arg("demand_matrix"),
          py::arg("alpha") = 0.2,
          py::arg("beta") = 0.1,
          py::arg("forecast_horizon") = 1.0
    );
}
