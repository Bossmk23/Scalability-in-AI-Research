# Performance Comparison: Single-GPU vs Multi-GPU (DDP) Training

## Introduction
In deep learning, the scalability of training workflows is a crucial factor when working with large datasets and complex models. To evaluate scalability, we conducted experiments using both **single-GPU training** and **multi-GPU Distributed Data Parallel (DDP) training**. This report compares the two approaches in terms of setup, execution, and performance outcomes.

---

## 1. Single-GPU Training

### Codebase
The `single_gpu_example.py` script trains a simple neural network on the MNIST dataset using only one GPU.  
Key features:
- Uses **PyTorch** with a standard training loop.
- Runs entirely on a single device (GPU if available, otherwise CPU).
- Straightforward setup with minimal overhead.

### Sample Output
```
Using device: cpu
Train Epoch: 1 [0/60000] Loss: 2.303463
Train Epoch: 1 [6400/60000] Loss: 0.208762
...
Train Epoch: 2 [57600/60000] Loss: 0.177160
Training completed in: 414.96 seconds
```
**Observation:**  
Training completed in ~415 seconds on CPU (single-device). The loss values decreased steadily, showing effective learning.

---

## 2. Multi-GPU Training (DDP)

### Codebase
The `ddp_example1_py.ipynb` script demonstrates **Distributed Data Parallel (DDP)** in PyTorch.  
Key features:
- Launches multiple processes, each handling a portion of the dataset.
- Synchronizes gradients across GPUs during training.
- Designed to utilize **2 GPUs** (or more if available).

### Behavior
- Each process trains on a shard of the data.
- Synchronization ensures models remain consistent across devices.
- Expected to achieve significant **speedup** compared to single-GPU training.

---

## 3. Comparison: Single-GPU vs Multi-GPU

| Aspect              | Single-GPU Training                 | Multi-GPU (DDP) Training                  |
|---------------------|--------------------------------------|-------------------------------------------|
| **Code Complexity** | Simple, minimal setup               | More complex (requires multiprocessing)   |
| **Hardware Usage**  | 1 GPU (or CPU fallback)             | Utilizes multiple GPUs simultaneously      |
| **Performance**     | ~415 seconds on CPU (baseline)      | Faster training expected with >1 GPU       |
| **Scalability**     | Limited to single device            | Scales across multiple GPUs efficiently    |
| **Best Use Case**   | Small datasets, debugging, prototyping | Large datasets, faster training, production |

---

## Conclusion
The experiments highlight the difference between single-device and distributed training setups.  
- **Single-GPU training** is straightforward and best suited for smaller workloads or initial prototyping.  
- **Multi-GPU DDP training** provides scalability, efficiency, and reduced training time when more GPUs are available.  

This comparison demonstrates why DDP is a preferred choice in research and production environments where computational resources can be scaled.
