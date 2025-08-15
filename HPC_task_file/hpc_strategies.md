
# High-Performance Computing (HPC) & Resource Optimization
**Focus:** Understanding hardware and parallelization techniques — *Written for beginners, usable by pros.*

---

## 0) TL;DR (Executive Summary)
- **Start on a GPU** (PyTorch or TensorFlow). Profile first, then optimize.
- **Use TPU** when you’re on **TensorFlow/JAX** and need **massive scale** (huge models or very large batch sizes).
- **Profile time & memory** with `torch.profiler`; export **TensorBoard traces**. Fix the slowest ops first.
- **Scale out** with **Distributed Data Parallel (DDP)**; prefer *more GPUs per node* before *more nodes* (to reduce network overhead).
- **Cloud**: Use **AWS (EC2: g4/g5/p4/p5)** for GPUs, **GCP TPU** for TPU Pods. Spot instances can save cost for experiments.

---

## 1) Choosing Hardware: GPUs vs TPUs
### 1.1 GPUs — The Flexible Workhorse
- Works with **PyTorch**, **TensorFlow**, JAX (via CUDA backend), and most other DL stacks.
- Massive ecosystem: **CUDA**, cuDNN, TensorRT, Nsight Systems, nvprof, PyTorch/TensorFlow profilers.
- Best for **research**, **experimentation**, **mixed workloads**, **custom ops**.
- Available everywhere: local rigs, on-prem clusters, AWS, Azure, GCP.

**When to pick GPUs**
- You use **PyTorch** (default choice).
- You want **maximum library/tooling compatibility**.
- You need **mixed precision** (FP16/bfloat16) and **Tensor Cores** (Ampere+).

### 1.2 TPUs — The Scale Specialist
- Google’s **ASIC** tuned for tensor ops; excels at **bfloat16/low precision** with **XLA**.
- Tightest integration with **TensorFlow** and **JAX**.
- Great for **very large models/datasets** and **pod-level** scaling.

**When to pick TPUs**
- Your stack is **TensorFlow/JAX** and you need **high throughput**.
- You can access **TPU v4/v5e** or **TPU Pods** on **GCP**.
- You want predictable scaling with XLA-compilable graphs.

### 1.3 Quick Reference
| Hardware | Best For | Frameworks | Notes |
|---|---|---|---|
| **GPU** | Research, prototyping, production | PyTorch, TF, JAX | Widest ecosystem; great debuggability |
| **TPU** | Massive-scale TF/JAX training | TF, JAX (XLA) | Best throughput at scale; GCP-centric |

---

## 2) Profiling: Find Bottlenecks (Time & Memory)
- Use `torch.profiler` to measure per-operator time and memory.
- Export to TensorBoard for timeline view.
- Focus optimization efforts on top bottlenecks.

---

## 3) Parallelism & Scaling
- **Single-GPU**: Use mixed precision, larger batches, optimize dataloaders.
- **Multi-GPU (Single Node)**: Use DDP via `torchrun`, backend NCCL.
- **Multi-Node**: Prioritize GPU-per-node before adding more nodes.

---

## 4) Cloud Basics
- **AWS GPUs**: p3, p4, g4, g5 instances.
- **GCP TPUs**: v4, v5e, TPU Pods.
- Spot instances for cost savings.

---

## 5) Concrete Steps
- Run profiling in Colab using GPU.
- Save logs and screenshots.
- Update results section.

---

## 6) Results
**Device**: NVIDIA Tesla T4  
**Python**: 3.11.3  
**PyTorch**: 2.6.0+cu124  
**torchvision**: 0.21.0+cu124  
**CUDA Version**: 12.4  
**CUDA Capability**: 7.5  
**Peak GPU Memory**: 0.273 GB  

**Top Ops by CUDA Time**:
1. `aten::conv2d` → 174.829 ms (70.61% of total CUDA time)
2. `volta_sgemm_128x64_nn` → 62.983 ms (25.44%)
3. `_5x_cudnn_volta_scudnn_128x64_relu_xregs_large_nn_v1` → 33.144 ms (13.39%)
4. `aten::batch_norm` → 28.645 ms (11.57%)
5. `aten::relu_` / `aten::clamp_min_` → ~23.840 ms each (~9.63%)

**Observations**:
- **Convolution layers** dominate CUDA execution time (~70%).
- Batch norm and activation layers also contribute significantly (~20% combined).
- Peak memory usage was low (0.273 GB), suggesting batch size can be increased for better GPU utilization.

---

## 7) Recommendations
- Profile before optimizing.
- Increase batch size where memory allows.
- Consider mixed precision to boost throughput.
- Optimize convolution-heavy parts with fused kernels or more efficient architectures.
- Use TPUs for large-scale TensorFlow/JAX training.

---
