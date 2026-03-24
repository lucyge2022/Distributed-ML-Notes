# GPU Concepts for ML

---

## 1. CPU vs GPU

| | CPU | GPU |
|---|---|---|
| Good at | Running OS, loading files, complex logic and branching, general purpose tasks | Matrix multiplication, same operation on millions of numbers, parallel computation |
| Bad at | Doing 10,000 simple things simultaneously | Complex logic, general purpose tasks |

> Matrix multiplication is **exactly** what ML needs — which is why GPUs dominate training.

---

## 2. RAM vs VRAM (内存 vs 显存)

| | RAM (System Memory) | VRAM (Video RAM) |
|---|---|---|
| Attached to | CPU | GPU |
| Typical size | 16–32GB (laptop) | 8–24GB (consumer), 40–80GB (A100) |
| Stores | OS, applications, files in use | Model weights, gradients, training data batches |

**Key rule:**
- CPU can only work with RAM
- GPU can only work with VRAM
- They **cannot** directly access each other's memory

---

## 3. PCIe — The Highway Between CPU and GPU

PCIe is the physical connection bus between CPU and GPU on the motherboard.

```
CPU land ←──── PCIe highway ────→ GPU land
 (RAM)         (data travels)      (VRAM)
```

| Version | Bandwidth |
|---|---|
| PCIe 4.0 (common today) | ~32 GB/s |
| PCIe 5.0 (newer) | ~64 GB/s |
| VRAM internal speed | ~2,000 GB/s |

> PCIe is **60x slower** than VRAM internally — this is a significant bottleneck for data transfer between CPU and GPU.

---

## 4. NVLink — Direct GPU-to-GPU Connection

When multiple GPUs are in one machine, they need to exchange data (e.g. gradients after each iteration).

**Without NVLink (PCIe only):**
```
GPU1 → CPU → GPU2    ← must go through CPU as middleman
speed: ~32 GB/s
```

**With NVLink** (high-end GPUs only — A100, H100):
```
GPU1 ←── NVLink ──→ GPU2    ← direct connection, CPU bypassed
speed: ~600 GB/s             ← 18x faster than PCIe!
```

| GPU | NVLink? |
|---|---|
| Consumer (RTX 4090) | No — must use PCIe for GPU↔GPU |
| Pro (A100, H100) | Yes — direct GPU↔GPU at 600 GB/s |

> NVLink is what makes large-scale distributed ML training practical. AllReduce gradient sync between GPUs runs over NVLink on production hardware.

> **RDMA network** RDMA (IB/RoCE) is a high-speed network between compute nodes that lets machines read each other's GPU VRAM memory directly, bypassing the OS and CPU — much higher bandwidth than standard TCP/IP.

[TODO] add ucx related work here.

---

## 5. Bottlenecks of Training on a Single Machine

### Bottleneck 1 — VRAM Capacity

Memory required during training for a 10B parameter model:

```
weights:          40GB   (10B params × 4 bytes float32)
gradients:        40GB   (same shape as weights)
optimizer state:  80GB   (2× weights for Adam — stores m and v moments)
                 ──────
total:           160GB
```

Best single GPU VRAM: **80GB** (A100)

```
160GB > 80GB  →  model doesn't fit on one GPU ✗
```

### Bottleneck 2 — PCIe Bandwidth (Multi-GPU on same machine)

If the model is split across GPU1 and GPU2 without NVLink:
- They must communicate via PCIe: **32 GB/s**
- GPUs spend most of their time waiting for data to transfer
- PCIe becomes the starvation bottleneck

### Bottleneck 3 — Storage to RAM Speed

```
YouTube-8M dataset: hundreds of GBs on SSD
SSD read speed:     3 GB/s
GPU batch time:     0.1 seconds to process a batch
SSD load time:      1 second to load that batch

→ GPU sits idle 90% of the time waiting for data  ← I/O bottleneck
```
[TODO] add alluxiofs related work that from storage point of view to reduce IO waiting time to save GPU cycles

### Bottleneck 4 — RAM to VRAM Transfer (PCIe again)

The full data pipeline:

```
SSD ──────→ RAM ──────→ VRAM ──────→ GPU processes
  3 GB/s        32 GB/s      2000 GB/s

Slowest link = SSD → RAM = 3 GB/s
Everything else waits for this!
```

| Link | Speed |
|---|---|
| SSD → RAM | 3 GB/s |
| RAM → VRAM (PCIe) | 32 GB/s |
| VRAM internal | 2,000 GB/s |

> The entire pipeline runs at the speed of its slowest link — SSD read speed dominates.
