# Enterprise Manufacturing Demand Forecasting & Inventory Optimization Platform

Hi, Aishwarya here. This is an enterprise-grade demand forecasting and safety stock optimization platform designed for high-throughput manufacturing lines and exascale cluster execution.

Instead of slow, single-threaded Python loops or database-heavy calculations, this system uses an **AVX2 SIMD vectorized C++ compute engine** that performs parallel Double Exponential Smoothing (Holt's Linear Trend Model) across 8 SKUs simultaneously. It incorporates real-time statistical calculations for inventory safety targets, integrated with MongoDB, distributed via MPI, and visualized in a premium dark-themed Streamlit dashboard.

---

## High-Performance Architectural Optimizations

### 1. Vectorized C++ SIMD Core (Structure of Arrays)
* **Double Exponential Smoothing (Holt's Model)**: Dynamically computes trend-adjusted demand projections to capture shifting manufacturing volumes.
* **8x SKU Parallelism**: Utilizes 256-bit Intel AVX2 registers (`__m256`) to process levels, trends, and forecasts for 8 SKUs in parallel.
* **Pybind11 Zero-Copy Binding**: maps memory pointers (`static_cast<float*>`) directly between NumPy array buffers and C++ vectors to eliminate memory copy latency.
* **OpenMP Multi-threading**: Automatically maps SKU chunks across all available hardware threads.

### 2. Industry-Grade Supply Chain Metrics
* **Safety Stock**: Computed on-the-fly using:
  $$Safety\ Stock = Z \times \sigma_d \times \sqrt{\text{Lead Time}}$$
  Where $Z$ represents the service level coefficient (e.g., 1.65 for 95% service level confidence), $\sigma_d$ is the standard deviation of daily demand, and $Lead\ Time$ is the replenishment window.
* **Reorder Point (ROP)**: Computed dynamically to flag shortage risks:
  $$ROP = (Average\ Daily\ Demand \times Lead\ Time) + Safety\ Stock$$
* **Dynamic Alert Flags**: Flags SKUs as `SAFE`, `REORDER`, or `CRITICAL` depending on active stock levels relative to ROP thresholds.

### 3. Distributed Scaling (MPI)
* Shards the inventory database records across independent processes via MPI (`mpi4py`) to run in parallel on large-scale HPC/exascale clusters.

---

## Setup & Compilation

### 1. Build C++ Kernel Module
Make sure you have CMake, a C++14 compliant compiler (GCC/Clang/MSVC) and AVX2-enabled processor.
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
# Copy the compiled binaries to the parent folder
cp forecaster*.so ..
cd ..
```

### 2. Run Distributed Workloads (MPI)
Execute the parallel computing engine over 4 cluster ranks:
```bash
mpirun -np 4 python worker.py
```
*Note: If MongoDB isn't running locally, the script automatically switches to generating simulated manufacturing historical demand data so you can test it immediately.*

### 3. Start the Web Dashboard
Launch the dashboard for real-time adjustments (Service Level and Lead Time settings recalculate thresholds instantly on the graphs):
```bash
streamlit run dashboard.py
```

---

## Interview Presentation Notes (Q&A Ready)
This project is structured specifically to showcase high-performance systems engineering expertise:
* **Memory Management**: Shows understanding of zero-copy binding, memory alignment (`alignas(32)`), and cache friendliness.
* **Vector Math**: Demonstrates real-world usage of Intel intrinsics (`_mm256_fmadd_ps`) for Fused Multiply-Add loops.
* **Cluster Orchestration**: Demonstrates how to write code for distributed memory systems using MPI, bypassing python's single-node limitations.

If you want to discuss latency audits or HPC optimizations, feel free to reach out for a cofeeee tech chat!
