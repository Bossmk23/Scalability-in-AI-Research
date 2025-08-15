# Scalability in AI Research

This repository documents my work from **Day 121–135** of the AI Research journey, focused on **Scalability** — enabling AI models to train efficiently on large datasets and architectures through high-performance computing (HPC) and distributed training strategies.

##  Objective
The main goal was to **explore, implement, and benchmark scalability strategies** for AI training workflows, leveraging **parallel processing, GPU/TPU acceleration, and distributed training frameworks**.

---

##  Project Structure
```
Final Report/
└── Scalability_in_AI_Research_final_doc.docx.pdf # Comprehensive research report
HPC_task_file/
├── HPC_PROFILING_STARTER.ipynb # Profiling code for HPC experiments
├── hpc_strategies.md # Summary of HPC scaling strategies
├── profiler_output.txt # Sample profiler output
└── README.md # Notes related to HPC experiments
distributed_training/
├── single_gpu_example.ipynb # Baseline single-GPU training
├── ddp_example1_py.ipynb # Distributed Data Parallel example
└── performance_comparison.md # Benchmarks: single vs distributed
```

---

##  Topics Covered

- **High-Performance Computing (HPC)**
  - Parallel processing & workload distribution
  - GPU/TPU utilization
  - Distributed training with frameworks like **Horovod** and **PyTorch DDP**
- **Challenges**
  - Scaling experiments while managing resource allocation efficiently
- **Hands-on Tasks**
  - Setting up distributed training experiments
  - Profiling and measuring performance improvements
  - Creating architecture diagrams for distributed setups

---

##  How to Use

###  Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Scalability-in-AI-Research.git
cd Scalability-in-AI-Research

# Install dependencies
pip install -r requirements.txt


