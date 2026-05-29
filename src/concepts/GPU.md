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

---

## 6. Reading `nvidia-smi` — GPU Utilization vs Power Draw

```
+-----------------------------------------------------------------------------------------+

| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2       |
|-----------------------------------------+----------------------+------------------------+

| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC   |
| Fan  Temp   Perf          Pwr:Draw / Limit |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M.   |
|=========================================+======================+========================+

|   0  NVIDIA A100-SXM4-80GB          On  | 00000000:00:04.0 Off |                    0   |
| N/A   42C    P0             143W / 400W |  45120MiB / 81920MiB |     87%      Default   |
|                                         |                      |                  Disabled|
+-----------------------------------------+----------------------+------------------------+
                                                                                          
+-----------------------------------------------------------------------------------------+

| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0  N/A  N/A      14832      C   python3                                     45110MiB |
+-----------------------------------------------------------------------------------------+
```

### What each field means

| Field | Value | Meaning |
|---|---|---|
| `Pwr:Draw / Limit` | 143W / 400W | Currently consuming 143W out of a 400W budget |
| `Memory-Usage` | 45120MiB / 81920MiB | ~55% of 80GB VRAM in use |
| `GPU-Util` | 87% | Over the last 1-second window, the GPU had at least one active kernel 87% of the time |

### Why Power Draw is the better busyness signal

**GPU-Util measures time occupancy, not compute intensity.**

It answers: *"Was the GPU doing anything?"* — not *"Was it doing it hard?"*

A GPU that processes one tiny kernel every millisecond and then idles for the rest of that millisecond will report 100% utilization. It is technically never idle, but it is barely working.

**Power Draw measures actual silicon activity.**

When Tensor Cores are running dense matrix multiplications at full throughput, they draw close to TDP (400W on an A100). When the GPU is waiting for data from CPU or doing lightweight work, power stays low regardless of what utilization reports.

```
GPU-Util 87%, Power 143W / 400W (36%)
→ GPU is rarely idle (high util) but is doing lightweight work each time it wakes up
→ bottleneck is likely: CPU overhead, data loading, or small batch sizes starving the Tensor Cores
```

### The Ferrari analogy

Think of the GPU as a Ferrari in city traffic:

- **GPU Utilization (87%)** = engine is on and wheels are rolling 87% of the time. Technically "utilizing" the car.
- **Power Draw (36%)** = fuel consumption. Crawling at 15 mph barely touches the gas pedal — almost no fuel burned even though the car is constantly in motion.

To reach 400W, you need the open racetrack: large batch sizes, no CPU bottleneck, Tensor Cores saturated with dense compute.

### What high util + low power tells you to fix

| Symptom | Likely cause | Fix |
|---|---|---|
| High util, low power | Small batches — GPU wakes, does tiny work, idles briefly | Increase batch size |
| High util, low power | CPU data preprocessing can't keep up | More DataLoader workers, prefetch |
| High util, low power | Python overhead between kernel launches | Move logic into CUDA kernels, use `torch.compile` |
| Low util, low power | GPU starved waiting for data | Fix I/O pipeline — SSD speed, caching, prefetch |
| High util, high power | GPU fully saturated | ✓ this is what you want |
