# Use official Python 3.9 base image
FROM python:3.9-slim

# Install system utilities, GCC/G++ compilation tools, and OpenMPI development libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory inside the container
WORKDIR /app

# Copy python dependencies file and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy C++ source and CMake configure settings to build the binary extension module
COPY CMakeLists.txt kernel.cpp ./

# Build the C++ extension module using CMake
RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . --config Release && \
    cp forecaster*.so .. && \
    cd .. && \
    rm -rf build

# Copy remaining python logic scripts
COPY worker.py dashboard.py ./

# Expose port 8501 for Streamlit access
EXPOSE 8501

# Command to run the dashboard as the default service
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
