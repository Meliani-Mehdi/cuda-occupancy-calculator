# GPU_sim — CUDA Kernel Execution Simulator

## Overview
**GPU_sim** is a C-based simulation tool designed to **model and visualize how CUDA kernels are launched and executed** across GPU streaming multiprocessors (SMs).  
It provides an educational, configurable way to understand GPU scheduling, resource management, and kernel execution flow, without needing a real CUDA-capable device.

This project was developed as part of an academic exploration into **GPU architecture modeling** and **CUDA kernel scheduling**.

---

## Features

- **JSON-based configuration** for defining custom GPU architectures and kernel properties.  
- **Dynamic simulation** of kernel launches across multiple GPUs.  
- **HTML output reports** for visualizing GPU utilization under different resource scenarios.
- 
---

## Project Structure

```
cuda_project/
├── build/                     # Compiled object files
├── code/
│   ├── cJSON.c / cJSON.h      # JSON parsing library
│   ├── cuda_arch.c / .h       # GPU architecture definitions and functions
│   ├── GPU_sim.c              # Main simulation engine
│   └── queue.h                # Helper header
├── config.json                # Configuration for GPUs and kernels
├── results/                   # HTML simulation outputs
│   ├── GPU_high_resources.html
│   ├── GPU_mid_resources.html
│   └── GPU_low_resources.html
├── gpu_style.css              # Visualization styling for reports
├── Makefile                   # Build automation script
└── GPU_sim                    # Compiled executable
```

---

## ⚙️ How It Works

### 1. **Configuration Input**
The simulator reads from `config.json`, which defines:
- A list of **GPUs** with architectural parameters (memory, SM count, registers, etc.)
- A list of **kernels** to be launched, each with its own block/thread configuration and resource requirements.

Example JSON structure:
```json
{
  "gpus": [
    {
      "name": "RTX_3080",
      "memory_bytes": 10485760,
      "shared_mem_per_sm": 98304,
      "registers_per_sm": 65536,
      "max_warps_per_sm": 64,
      "max_blocks_per_sm": 32,
      "num_sms": 68
    }
  ],
  "kernels": [
    {
      "name": "MatrixMul",
      "number_of_blocks": 256,
      "threads_per_block": 512,
      "shared_mem_used_in_bytes_per_block": 2048,
      "registers_per_thread": 32,
      "stream_id": 0
    }
  ]
}
```

---

### 2. **Simulation Flow**
- `GPU_sim.c` loads GPU and kernel data from JSON.  
- For each GPU:
  - Each kernel is launched and simulated via `launch_one_kernel()`.  
  - GPU resource usage (shared memory, registers, warps, etc.) is updated.  
  - `print_GPU_info()` displays summary statistics in the terminal.  
  - `export_GPU_to_HTML()` creates an HTML visualization under `/results/`.  

---

### 3. **Output Reports**
Each GPU simulation produces an HTML report such as:
```
results/
├── GPU_high_resources.html
├── GPU_mid_resources.html
└── GPU_low_resources.html
```
These visualize the **resource utilization** and **execution results** in a easy to read way, styled using `gpu_style.css`.

---


## Build Instructions

Make sure you have a C compiler (e.g. `gcc`) installed.

```bash
# Compile the project
make

# Run the simulation
./GPU_sim

# Compile and run
make run

# Clean build files
make clean

# Clear and reset results directory
make clear
```

All object files will be stored under the `/build` directory.

---

## 📊 Sample HTML Output Preview

Each HTML file contains:
- GPU details (name, SM count, memory, etc.)
- Graphical view of SM utilization and resource occupancy

The `gpu_style.css` ensures consistency and readability across reports.


---
