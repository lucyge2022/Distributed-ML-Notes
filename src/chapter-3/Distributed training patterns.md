# Chapter 3: Distributed Training Patterns

> **My background when reading this:** I came in knowing basic Python and having heard
> words like "gradient" and "GPU" without really understanding them. These notes are
> written for engineers in the same position — curious, technical, but new to ML systems.

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

## Pattern 1 — Parameter Server + Workers

### When to use it

Use this pattern when your **model is too large to fit on a single worker**. The model lives
on dedicated parameter server (PS) nodes, and worker nodes do all the computation.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Parameter Servers                    │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │   PS 1   │   │   PS 2   │   │   PS 3   │         │
│  │layer 1-3 │   │layer 4-6 │   │layer 7-9 │         │
│  │ (weights)│   │ (weights)│   │ (weights)│         │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘         │
└───────┼──────────────┼──────────────┼───────────────┘
        │   fetch params│              │
        ▼               ▼              ▼
┌─────────────────────────────────────────────────────┐
│                     Workers                          │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │                  Worker 1                     │   │
│  │  data chunk A + fetched model params          │   │
│  │                                               │   │
│  │  forward pass → loss → backprop → gradients   │   │
│  │                                               │   │
│  │  gradients layer 1-3 ──────────────► PS 1    │   │
│  │  gradients layer 4-6 ──────────────► PS 2    │   │
│  │  gradients layer 7-9 ──────────────► PS 3    │   │
│  └──────────────────────────────────────────────┘   │
│                   (same for W2, W3...)               │
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
- Sends each gradient slice back to the PS that owns those params

> **Key insight I initially got wrong:** I assumed the parameter server did the forward pass
> and sent results to workers. It's actually the opposite — the PS is just storage.
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

### The staleness problem

Workers run **asynchronously** — they don't wait for each other. This creates a race condition:

```
t=0:  W1 and W2 both fetch model at version v5

t=1:  W1 finishes fast → sends gradients → PS updates to v6

t=2:  W2 still computing using v5...
      PS is now at v6

t=3:  W2 finishes → sends gradients based on STALE v5
      ← these gradients could push params in the wrong direction!
```

**Solution — version stamping with tolerance threshold τ:**

```
Each gradient batch carries a version number.
PS checks: current_version - gradient_version ≤ τ ?
  YES → accept and apply
  NO  → drop, tell worker to refetch and redo
```

The parameter τ is your staleness tolerance knob — a key concept from the
[CMU Parameter Server paper (Li et al., 2013)](https://arxiv.org/abs/1412.6651).

### Fault tolerance

If a worker fails mid-computation the PS simply stops receiving its gradients. Other workers
continue unaffected. The PS can checkpoint its parameters periodically so that if a PS node
fails, training resumes from the last checkpoint rather than from scratch.

---

## Pattern 2 — Worker-Only (AllReduce)

### When to use it

Use this pattern when **each worker machine has enough memory to hold a complete copy of
the model**. There are no parameter servers — workers communicate directly with each other.

### Prerequisite

```
Model fits entirely on one worker's GPU VRAM
  ↓
No need for parameter servers
  ↓
Every worker holds identical full model copy
  ↓
Workers only need to sync gradients after each batch
```

### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Worker 1   │     │   Worker 2   │     │   Worker 3   │
│              │     │              │     │              │
│ [full model] │     │ [full model] │     │ [full model] │
│ data chunk A │     │ data chunk B │     │ data chunk C │
│              │     │              │     │              │
│  gradients   │◄────►  gradients   │◄────►  gradients   │
│  [-0.03,..] │     │  [+0.02,..] │     │  [-0.01,..] │
└──────────────┘     └──────────────┘     └──────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    AllReduce: average all gradients
                    result: [-0.007, ...] same on ALL workers
                             │
                    all workers update identically
                    ← models stay in perfect sync!
```

### What AllReduce actually does

```
BEFORE AllReduce:
  W1 gradients: [-0.03, +0.01, +0.07, ...]
  W2 gradients: [+0.02, -0.04, +0.01, ...]
  W3 gradients: [-0.01, +0.02, -0.05, ...]

AFTER AllReduce (average):
  W1: [-0.007, -0.003, +0.01, ...]   ← identical
  W2: [-0.007, -0.003, +0.01, ...]   ← identical
  W3: [-0.007, -0.003, +0.01, ...]   ← identical

Every worker sees the gradient signal from ALL data chunks.
Parameter update = as if one machine saw all the data.
```

### AllReduce = Reduce + Broadcast

AllReduce can be decomposed into two simpler collective operations:

```
Step 1 — Reduce (gather and combine):
  W1, W2, W3 all send gradients to W1
  W1 averages them
  
Step 2 — Broadcast (distribute result):
  W1 sends averaged gradients back to W2, W3

Problem: W1 becomes a bottleneck — single point of failure,
         receives ALL traffic, does ALL computation.
```

### Ring AllReduce — the scalable solution

Instead of one central collector, workers form a ring and pass data around:

```
     W1
    /    \
  W4      W2
    \    /
     W3

Data flows clockwise in two phases:
  Phase 1 (ReduceScatter): partial sums accumulate around ring
  Phase 2 (AllGather):     complete results distributed around ring
```

**Why ring is better:**

| | Naive AllReduce | Ring AllReduce |
|---|---|---|
| Messages | O(N²) | O(N) |
| Bottleneck | Yes (center node) | No |
| Fault tolerance | Single point of failure | Distributed |
| Used by | Nobody in production | PyTorch DDP, NCCL |

With 100 workers: naive = 9,900 messages, ring = 198 messages.

### PyTorch implementation

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# initialize the process group — workers discover each other
dist.init_process_group(backend='nccl')  # nccl = NVIDIA's collective comms library

# wrap model — DDP handles Ring AllReduce automatically after each backward pass!
model = DDP(model)

# training loop is IDENTICAL to single-machine training
for batch_images, batch_labels in dataloader:
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')

    predictions = model(batch_images)           # forward pass
    loss = criterion(predictions, batch_labels) # loss

    optimizer.zero_grad()
    loss.backward()    # DDP invisibly runs Ring AllReduce here!
    optimizer.step()   # all workers update identically
```

### Fault tolerance in AllReduce

**Key advantage over Parameter Server:** every worker holds a complete model copy.
If one worker fails with no checkpoint saved, you can recover the latest model from
any surviving worker — because AllReduce guarantees all workers are always identical.

```
W4 fails with no checkpoint
  ↓
W1, W2, W3 all have identical latest model
  ↓
new worker joins, fetches model from W1
  ↓
training resumes!
```

For maximum safety, production systems also save async checkpoints to remote storage
(S3, GCS) so recovery is possible even if all workers fail simultaneously.

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