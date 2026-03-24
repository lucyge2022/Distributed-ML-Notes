# Chapter 3: Distributed Training Patterns

> **Topic Outline**<br>
> * Parameter server & Worker-only pattern<br>
> * Collective communication pattern<br>
> * Scalability(elasticity) / Reliability(FaultTolerance) pattern

---

## The Core Problem

Training a large neural network on a single machine is either impossible or impractically slow.
A 10B parameter model requires roughly 160GB of memory just for weights, gradients and optimizer
state — far exceeding a single GPU's VRAM. Even models that *fit* on one machine train too slowly
on large datasets to be practical.

The solution: distribute the work across multiple machines. But how?

There are two fundamentally different answers, and choosing between them depends on your model
size and hardware setup.

---

## 1. Pattern 1 — Parameter Server + Workers

### When to use it

Use this pattern when your **model is too large to fit on a single worker**. The model lives
on dedicated parameter server (PS) nodes, and worker nodes do all the computation.

> How large is considered large? How to estimate size of a model?

| Size | Example | Fits on... |
|---|---|---|
| ~100M–1B | Early BERT, small GPTs | Single GPU |
| ~7B–13B | Llama 2/3 small | 1–2 high-end GPUs |
| ~70B | Llama 3 70B | Needs multiple GPUs |
| ~500B+ | GPT-4 (estimated) | Needs many machines |

e.g. 10B parameter(with float32 precision) model size:

    the weights 10 B * 4 bytes = 40G
    the gradient (same with size of weights) 40G
    optimizer(Adam - tracks two extra values per parameter: first moment (m) and second moment (v), both float32) 10 B * 4 bytes * 2 = 80G
    -------
    total 40+40+80=160G


### Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Parameter Servers                   │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │   PS 1   │   │   PS 2   │   │   PS 3   │         │
│  │layer 1-3 │   │layer 4-6 │   │layer 7-9 │         │
│  │ (weights)│   │ (weights)│   │ (weights)│         │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘         │
└───────┼──────────────┼──────────────┼───────────────┘
        │   fetch params│              │
        ▼               ▼              ▼
┌─────────────────────────────────────────────────────┐
│                     Workers                         │
│                                                     │
│  ┌──────────────────────────────────────────────-┐  │
│  │                  Worker 1 workflow            │  │
│  │  for data chunk A                             │  │
│  │                                               │  │
│  │  fetch params layer 1-3 ◄─────────── PS 1     │  │
│  │  fetch params layer 4-6 ◄─────────── PS 2     │  │
│  │  fetch params layer 7-9 ◄─────────── PS 3     │  │
│  │                                               │  │
│  │  forward pass → loss → backprop → gradients   │  │
│  │                                               │  │
│  │  send gradients layer 1-3 ──────────► PS 1    │  │
│  │  send gradients layer 4-6 ──────────► PS 2    │  │
│  │  send gradients layer 7-9 ──────────► PS 3    │  │
│  └──────────────────────────────────────────────-┘  │
│                   (same for W2, W3...)              │
└─────────────────────────────────────────────────────┘
```

### What each role does

**Parameter Server (passive):**
- Stores its partition of model weights permanently
- Receives gradient updates from workers
- Applies updates: `new_params = old_params - (lr × gradients)`
- Sends fresh params back on request

**Worker (active):**
- Fetches the full model by pulling from ALL parameter servers
- Runs the complete forward pass through every layer
- Computes loss and backpropagates to get gradients
- Sends each gradient slice back to the PS that owns those params to update the model weights

> **Key insights:** 
> Workers do ALL the computation. Think of PS as a database, workers as the application servers.

### The gradient flow in detail

```
Worker fetches:
  PS1 params (layer 1-3) ──┐
  PS2 params (layer 4-6) ──┼──► Worker holds full model temporarily
  PS3 params (layer 7-9) ──┘
          ↓
  input data → layer1 → layer2 → ... → layer9 → prediction
          ↓
  loss = cross_entropy(prediction, true_label)
          ↓
  backprop: chain rule flows backwards through all 9 layers
          ↓
  full gradient matrix generated (same shape as full model!)
          ↓
  gradient slice [layer 1-3] ──► PS1 updates its params
  gradient slice [layer 4-6] ──► PS2 updates its params
  gradient slice [layer 7-9] ──► PS3 updates its params
```

**Why is the gradient always for the full model?**
Because computing gradients requires a complete forward pass first. You can't get a partial
gradient from a partial model — you need the prediction, which needs all layers.

## 2. Pattern 1 - Where is Bottleneck / State / Failure?

### 2.1 Bottleneck
1- What’s the result of increasing the number of workers or parameter servers?

    more param servers -> 
      [good]  smaller partition on each server
      [good]  reduce bottleneck on single server for serving
      [good]  more parallelism in update params
      [bad]   more routing overhead for workers
      [bad]   more network fan-out / round trips
      [bad]   possible uneven sharding causing nodes with large shards becomes a bottleneck (a hot node in traditioanl distibuted sys problem)

    more workers ->
      [good]  more parallelism in computing, faster computation 
      [good]  more dataset sharding
      [bad]   more communication overhead (more update conflict with peer workers)
      [bad]   decreased freshness on model and decreased accuracy (sometimes might use stale model for computation)
      [bad]   diminishing eventual outcome (communication cost overtakes computation gains)

### 2.2 State
2- Where does each component resides? What types of computational resources should we allocate to param servers?

    Param Servers(PS):
      Mem - critical, for faster param read/write
      CPU - for network i/o and gradient averaging (if it collects multiple gradients and avg and then apply, but most PS apply each gradient immediately)
      disk - for checkpointing (or you can checkpoint to external store such as Alluxio / S3 / HDFS, etc)
      GPU - no need
    Worker:
      Mem - yes, for model partiton (before feeding to GPU VRAM)
      CPU - yes, for possible dataset pre-processing before feeding to GPU
      Disk - yes, for loading dataset batches, or gets fed from external storage
      GPU - yes, critical for forward passing / backprop

> **Extended read** Get more insights from bottleneck in hardware during the training process, [TODO] link to md note

### 2.3 Failures 
3- What failures / exception could occur in the PS pattern of distributed training? What's the resolution

**2.3.1 PS Node Failures**

| Failure | Resolution |
|---|---|
| PS fully down | Restart from last checkpoint (all PS nodes checkpoint periodically to disk or remote storage like S3/HDFS) |
| One PS node down (partial) | Only workers needing that shard are blocked — other workers continue. Restart that PS from its checkpoint shard |
| PS memory OOM | Reduce model shard size per PS, add more PS nodes to spread the load |
| Checkpoint file corrupted | Keep last N checkpoints (e.g. N=3), fall back to N-1 |

**2.3.2 Worker Failures**

| Failure | Resolution |
|---|---|
| Worker permanently down | Reassign its dataset shard to surviving workers. PS never blocks — it just stops receiving gradients from that worker |
| Worker temporarily down | Wait with a timeout, then reassign if timeout exceeded |
| Slow worker (very slow) | In async PS this is naturally tolerated — other workers keep going. In sync PS a slow worker(straggler) blocks everyone → async is preferred for heterogeneous hardware |

**2.3.3 Network Failures**

| Failure | Resolution |
|---|---|
| Stale gradients | PS version-stamps each gradient batch; drops if `current_version - gradient_version > τ`, tells worker to refetch and redo |
| Slow fetch (worker ← PS) | Prefetch next batch's params while computing current batch (overlap compute and communication) |
| Slow push (worker → PS) | Retry with exponential backoff; forfeit if PS has already moved too many versions ahead (staleness check handles this) |
| Network partition (PS unreachable) | Workers retry with backoff; if PS stays unreachable past timeout, worker pauses and alerts |

> **Note on slow fetch:** a slow PS fetch is the most insidious failure in practice — it directly stalls the worker's entire compute pipeline since the forward pass can't start until all params arrive. Prefetching is the main mitigation.

**2.3.4 Split-brain (PS nodes diverge)**

If the network between PS nodes partially fails, two PS nodes might diverge on what the "current" model version is — workers pulling from PS1 get version 10, workers pulling from PS2 get version 9.
- Use a single coordinator to track global version (adds a bottleneck but prevents divergence)
- Accept it as a form of staleness and let τ handle it

**2.3.5 The staleness problem**

Workers run **asynchronously** — they don't wait for each other. This creates a race condition:

```
t=0:  W1 and W2 both fetch model at version v5

t=1:  W1 finishes fast → sends gradients → PS updates to v6

t=2:  W2 still computing using v5...
      PS is now at v6

t=3:  W2 finishes → sends gradients based on STALE v5
      ← these gradients could push params in the wrong direction!
```

Solution — version stamping with tolerance threshold τ:

```
Each gradient batch carries a version number.
PS checks: current_version - gradient_version ≤ τ ?
  YES → accept and apply
  NO  → drop, tell worker to refetch and redo
```

The parameter τ is your staleness tolerance knob — a key concept from the
[CMU Parameter Server paper (Li et al., 2013)](https://arxiv.org/abs/1412.6651).

---

### Real-life use case — storage optimization with Alluxio

In production PS training, the two biggest storage bottlenecks are checkpointing and dataset retrieval. A common solution is to place **Alluxio** as an intermediate caching layer between your workers/PS nodes and remote storage (S3, HDFS, GCS).

```
                        ┌─────────────┐
  PS nodes ─────────────►             │
                        │   Alluxio   ◄────── S3 / HDFS / GCS
  Workers  ─────────────►   (cache)   │         (remote store)
                        └─────────────┘
```

**1. Checkpointing**

PS nodes checkpoint their weight shards periodically. Writing directly to S3/HDFS introduces high latency on every checkpoint write, which stalls training if done synchronously.

With Alluxio:
- PS writes checkpoint to Alluxio (in-memory, fast — close to local disk speed)
- Alluxio asynchronously flushes to S3/HDFS in the background
- Training is never blocked waiting for remote storage I/O
- On recovery, PS reads the latest checkpoint from Alluxio cache (warm) rather than pulling from S3 cold

**2. Dataset retrieval**

Workers stream dataset batches each iteration. Pulling from S3/HDFS on every batch adds significant I/O latency that can starve the GPU.

> **[TODO]** Link to dataset retrieval and distributed data loading note — covers prefetching strategies, sharding, and Alluxio tiered storage in depth.

---

## Pattern 2 — Worker-Only (AllReduce)

→ Full notes: [Worker-Only (AllReduce) pattern.md](./Worker-Only%20(AllReduce)%20pattern.md)

---

## Comparison: When to Use Which

| | Parameter Server | AllReduce (Worker-Only) |
|---|---|---|
| Model fits on one GPU? | Not required | Required |
| Communication pattern | Worker ↔ PS | Worker ↔ Worker |
| Gradient sync | Async (can be stale) | Sync (always fresh) |
| Fault tolerance | Checkpoint PS | Any surviving worker |
| Staleness risk | Yes | No |
| Implementation | PyTorch RPC | PyTorch DDP |
| Best for | Very large models | Medium models, many workers |

> **My mental model:** Parameter Server is like a shared Google Doc — workers edit their
> section and changes sync back to a central store. AllReduce is like a team vote — 
> everyone submits their opinion and the group reaches one consensus answer together.

---

## Key Terms Quick Reference

| Term | Plain English |
|---|---|
| Parameter Server | Machine that stores model weights and applies updates |
| Worker | Machine that computes forward pass, loss and gradients |
| Gradient | Matrix of nudge values — same shape as model, tells each param which direction to move |
| AllReduce | Collective operation where all workers contribute and all receive the averaged result |
| Ring AllReduce | Efficient AllReduce using ring topology — O(N) messages instead of O(N²) |
| Staleness | Gradients computed from an outdated model version |
| τ (tau) | Staleness tolerance threshold — how many versions behind is acceptable |
| Checkpoint | Saved snapshot of model weights + optimizer state — the training save point |
| NCCL | NVIDIA's library implementing Ring AllReduce on GPUs |

---

## Further Reading

- [Parameter Server for Distributed Machine Learning — Li et al., CMU (2013)](https://arxiv.org/abs/1412.6651)
- [PyTorch Distributed Training docs](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- *Distributed Machine Learning Patterns* — Chapter 3 (the book these notes accompany)