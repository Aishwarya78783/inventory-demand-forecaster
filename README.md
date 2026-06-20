# Ai-inventory-demand-forecasting-dashboard

Hi, Aishwarya here. This is a high-performance inventory demand forecasting engine optimized for HPC and exascale cluster environments. 

Instead of relying on slow Python loops or pandas calculations for thousands of SKUs, I wrote a C++ compute kernel vectorized with AVX2 SIMD instructions (FMA - Fused Multiply-Add) to get maximum performance out of the CPU. It uses OpenMP for multi-threading, MongoDB to store transaction history, MPI to distribute workloads across nodes, and a sleek Streamlit dashboard to show it .

## Tech Stack
* **Core math**: C++ (AVX2 SIMD, OpenMP)
* **Bindings**: pybind11
* **Distributed engine**: mpi4py (MPI)
* **Database**: MongoDB
* **Dashboard**: Streamlit + Matplotlib (custom dark theme)

---

## How to Compile & Run Locally

### 1. Build the C++ AVX2 Kernel
Make sure you have CMake and a compiler supporting AVX2.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
# Copy the compiled .so or .pyd file to the main folder
cp forecaster*.so ..
cd ..
```

### 2. Start the Distributed Forecaster (MPI)
To run the forecasting job distributed across multiple worker processes:
```bash
mpirun -np 4 python worker.py
```
*Note: If MongoDB isn't running locally, the worker script will automatically generate simulated SKU data so you can test it.*

### 3. Run the Dashboard
Launch the Streamlit frontend to visualize historical data and predictions:
```bash
streamlit run dashboard.py
```

---

## Performance optimizations in this repo:
* **AVX2 SIMD Vectorization**: Calculates 8 floating-point operations per cycle inside the C++ FMA loop.
* **OpenMP Threading**: Distributes SKU batches across all available CPU cores.
* **MPI Workload Scattering**: Master node shards the SKU list and scatters them across cluster nodes to prevent memory bottlenecks.
* **MongoDB Bulk Operations**: Minimizes database trip latency by writing forecasts in bulk updates.

Open for a cofeeee tech chat if you want to discuss HPC tuning or latency audits!
