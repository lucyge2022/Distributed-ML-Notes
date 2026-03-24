# Pattern 2 — AllReduce (Worker-Only)

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
