# 1. Pattern 2 — AllReduce (Data Parallel)

### When to use it

Use this pattern when **each worker machine has enough memory to hold a complete copy of
the model**. There are no parameter servers — workers communicate directly with each other.

### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Worker 1   │     │   Worker 2   │     │   Worker 3   │
│              │     │              │     │              │
│ [full model] │     │ [full model] │     │ [full model] │
│ data chunk A │     │ data chunk B │     │ data chunk C │
│              │     │              │     │              │
│  gradients   │◄────►  gradients   │◄────►  gradients   │
│  [-0.03,..]  │     │  [+0.02,..]  │     │  [-0.01,..]  │
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

Workflow of AllReduce:

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

**AllReduce = Reduce + Broadcast**

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

# 2. Where is Bottleneck / State / Failure

### 2.1 Bottleneck
1- What’s impact of increasing workers servers? what's the bottleneck during training?
Since this architecture relies heavily on underlying network infra for communication, major bottleneck comes from here.

    network overhead ->
      one worker sends gradients to all other workers -> O(N*N) fan out coming from broadcast
      slow worker -> blocking on slow worker to gather all gradient update ( but could easily bypass)

### 2.2 State

| Resource | Purpose |
|---|---|
| Mem | only for dataset-related pipelining/processing, but not for model params anymore, for GPU with NVLinks/RDMA, GPU ↔ GPU directly via NVLink / RDMA — never touches CPU RAM |
| CPU | network io & dataset related |
| Disk | dataset retrieval / model checkpointing for recovery, could coupled with external storage |
| GPU | forward / loss / backprop / gradient in-place update |

---

Some nuances compared to PS pattern:

**GPU VRAM holds (all of it, unlike PS pattern):**
- Model weights
- Activations (forward pass intermediates)
- Gradients
- Optimizer state (Adam's m and v moments) — unlike PS pattern where optimizer state lives on the PS

**Gradient updates happen in-place on GPU VRAM.** The full cycle stays on GPU:

| Step | Where |
|---|---|
| Forward pass | GPU |
| Loss computation | GPU |
| Backprop → gradients | GPU |
| NCCL AllReduce (gradient sync) | GPU ↔ GPU directly via NVLink / RDMA — never touches CPU RAM |
| `optimizer.step()` weight update | GPU, in-place |

---

### 2.3 Failures

**2.3.1 Worker Down**

AllReduce is a collective operation — every worker must participate in each round. A single unresponsive worker blocks the entire ring.

- Wait for a timeout threshold
- If still unresponsive, kick the worker and **reform the process group** (re-initialize `dist.init_process_group` with remaining workers)
- Training continues with the smaller group; the ring re-forms around the gap
- When the worker comes back online, it restarts from the last saved checkpoint and rejoins — it cannot rejoin mid-ring without a full resync

> **Key difference from PS:** in the PS pattern a slow/dead worker is silently ignored (PS just stops receiving its gradients). In AllReduce, one dead worker poisons the whole round — fault tolerance requires active group reformation.

**2.3.2 Network Partition**

The impact depends on the blast radius:

| Scenario | Resolution |
|---|---|
| Single worker isolated | Treat as worker down — kick, reform group, worker rejoins from checkpoint |
| Subset of workers partitioned | Reform with the surviving connected group; partitioned workers resync from last checkpoint when reconnected |
| Severe partition (network splits workers into two islands) | Both islands may independently continue with wrong gradients — must halt, wait for partition to heal, then resync all workers from last agreed-upon checkpoint to guarantee model consistency |

> In AllReduce, a network partition is more dangerous than in PS because there is no central authority tracking model version. Workers on each side of the partition diverge silently — everyone must roll back to the last checkpoint where all agreed.

**2.3.3 Dataset / Batch Corruption**

- IO error / data corruption
- Resolution: Skip the corrupted batch and advance to the next one
- Optionally log the bad batch index for offline inspection

**2.3.5 Straggler Worker**

Unlike PS (which is async and naturally tolerates stragglers), AllReduce is **synchronous** — the ring waits for the slowest worker every round.

- Short-term: acceptable, the ring just runs at straggler speed
- Long-term: if one worker is consistently 10x slower, kick it and reform the group
- Production mitigation: provision homogeneous hardware so no stragglers exist by design

---

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
