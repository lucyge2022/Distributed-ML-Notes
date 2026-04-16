# Distributed ML Notes

Engineering notes on distributed ML systems — written from an infrastructure perspective. The focus is on the system design problems underneath the ML abstractions: data movement at scale, distributed compute coordination, memory hierarchies, and the operational failure modes that textbooks skip.

Content keep updating...

live site at : https://lucyge2022.github.io/Distributed-ML-Notes/index.html
---

## Contents

### Concepts
Background knowledge that cuts across all chapters.
- [GPU, VRAM, PCIe & NVLink](./src/concepts/GPU.md) — hardware mental model: why GPUs, what VRAM is, PCIe bottleneck, NVLink
- [PyTorch Memory Management](./src/concepts/PyTorch-Memory-Management.md) — how PyTorch allocates and caches CUDA memory

### Chapter 2 — Data Ingestion Patterns
How raw data gets from disk into a training loop.
- [Dataset](./src/chapter-2/Dataset.md) — dataset composition, IDX binary format, tensor representation, data ingestion flow
- [Batching, Sharding & Caching](./src/chapter-2/Batching,Sharding,Caching.md) — Ray+Parquet vs WebDataset vs MosaicML MDS: formats, sharding strategies, shuffle, remote streaming

### Chapter 3 — Distributed Training Patterns
How training is parallelized across multiple machines.
- [Parameter Server Pattern](./src/chapter-3/PS-Pattern.md) — PS + workers architecture, gradient flow, staleness, fault tolerance
- [Worker-Only (AllReduce) Pattern](./src/chapter-3/AllReduce-Pattern.md) — Ring AllReduce, PyTorch DDP, fault tolerance, failure analysis

### Chapter 4 — Feature Store
How features are computed, stored, and served consistently between training and inference.
- [Feature Store](./src/chapter-4/Feature-Store.md) — offline vs online paths, train-serve skew, Chronon consistency measurement; illustrated with a user–restaurant recommendation system

---

## Supplemental Code

Runnable toy programs that accompany these notes: [distributed-ml-examples](https://github.com/lucyge2022/distributed-ml-examples)

| Example | What it covers |
|---|---|
| [ddp-testrun](https://github.com/lucyge2022/distributed-ml-examples/tree/main/ddp-testrun) | PyTorch DDP training with Ring AllReduce on MNIST |

---

## Built with

[mdBook](https://rust-lang.github.io/mdBook/) — the `src/SUMMARY.md` defines the book structure.

```bash
mdbook serve   # local preview at http://localhost:3000
mdbook build   # build static site to book/
```
