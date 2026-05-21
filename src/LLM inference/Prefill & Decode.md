# Prefill vs Decode — LLM Inference Two-Phase Deep Dive

> Grounded in concrete numbers using Llama-3 70B + B200 GPU to build real intuition.

---

## 1. What is Prefill? What is Decode?

### Prefill

Processes all prompt tokens in one shot — fully parallel — and builds the KV cache.

```
Input:   prompt = ["where", "does", "the", "cat", "sit", "?"]
         shape  = (n, d)    n=6 tokens, d=model hidden dimension

Process: all n tokens fed into the model simultaneously
         each layer computes Q, K, V for every token
         K and V written into HBM KV cache

Output:  first generated token ("on")
         + fully populated KV cache (all n tokens, all layers)

Q = X × Wq → (n, d_k)  d_k = k/q dimension per head
K = X × Wk → (n, d_k)
V = X × Wv → (n, d_v)  d_v = d_k
say d = d_k * head_num

Attention = softmax(Q × Kᵀ / √d_k) × V
          = softmax((n,d) × (d,n) / √d) × (n,d)
          = softmax(n,n) × (n,d)
          = (n,d)   ← same shape as input to attention!         
O(n * n * d) = O(n^2)
```


**Key property: one-shot, fully parallel across all input tokens, O(n²) attention.**

---

### Decode

Autoregressively generates one token per step. Each step is a full forward pass over a single new token, looping until EOS.

```
step 1: input=(1,d) → forward pass → "on"
step 2: input=(1,d) → forward pass → "the"
step 3: input=(1,d) → forward pass → "mat"
... every step is a complete forward pass, but only for 1 new token
```

At each step, attention works as follows: the new token's Q (shape `1 × d_k`) attends over all historical K and V vectors read from the KV cache.

**Key property: sequential, one token per step, O(n) attention per step (n grows by 1 each step).**

---

## 2. Compute-Bound vs Memory-Bound

### Prefill — Compute-Bound

**Why?**

All n tokens attend to each other, forming an n×n matrix multiplication:

```
Q shape = (n, d_k)
K shape = (n, d_k)

Q × Kᵀ = (n, n)  ← n² multiplications

n = 1M tokens → 10¹² multiplications in attention alone
← GPU TFLOPS is the true bottleneck
← HBM bandwidth is comparatively adequate; the GPU is fully utilized
```

**Prefill speed calculation (1M token context, Llama-3 70B, B200):**

FLOP estimate (standard forward-pass rule of thumb):
**2 × num_params × num_tokens**

```
Theoretical its O(n^2) FLOP
but attention is only part of the model, actual we need multiplying with a const.

FLOP estimate (standard forward-pass rule of thumb):
  ≈ 2 × num_params × num_tokens
  

  = 2 × 70×10⁹ × 10⁶
  = 1.4 × 10¹⁷ FLOP  = 140 PFLOP

B200 compute = 4500 TFLOPS = 4.5 × 10¹⁵ FLOP/s

Prefill time      = 1.4×10¹⁷ / 4.5×10¹⁵  ≈  31 seconds

Prefill throughput = 1M / 31s             ≈  32,000 tokens/sec
```

Those 31 seconds are exactly why long-context **TTFT** (time to first token) is so high.

---

### Decode — Memory-Bound

**Why?**

Only 1 new token per step:

```
Q shape = (1, d_k)   ← just 1 row!
K shape = (n, d_k)   ← read from KV cache, n historical tokens

Q × Kᵀ = (1, n)      ← only n multiplications — compute is trivial!

But every step must read from HBM:
  Model weights:  70B × 2 bytes        = 140 GB  (full model, every step!)
  KV cache:       n tokens × per-token = 131 GB  (grows linearly with n)

← GPU compute sits >99% idle waiting for memory
← HBM bandwidth is the true bottleneck
```

**Why read the entire model weights every step?**

Each decode step is one full forward pass through all 32 layers. Every layer has Wq, Wk, Wv, Wo, and FFN weights that must be read from HBM to perform the matrix multiplications. Weights cannot be "cached in registers" — GPU registers are tens of MB; a 70B model is 140 GB.

**Decode speed calculation (1M token context, Llama-3 70B, B200):**

```
Memory read per step:
  Model weights  = 70B × 2 bytes (fp16)                      = 140 GB
  KV cache       = 1M × 80 layers × 8 heads × 128 dim × 2 × 2 = 131 GB
  Total          ≈ 271 GB per decode step

B200 HBM bandwidth = 8 TB/s

Time per step  = 271 GB / 8 TB/s  = 0.034 s  = 34 ms/token
34ms to generate one output token

Decode throughput  = 1 / 0.034    ≈ 29 tokens/sec   generate 29 output tokens per sec
```

**Prefill vs Decode side by side:**

```
Prefill:   32,000 tokens/sec  ← parallel,  compute-limited
Decode:        29 tokens/sec  ← sequential, bandwidth-limited

Gap: ~1,100×

← This is the root cause of why generation feels slow
← Magic's 100M context challenge in concrete terms:
     prefill alone  ≈ 3,100 seconds
     KV cache size  ≈ 13 TB
```

**Precise complexity of decode:**

Each step is O(n_current), but n_current grows by 1 each step:

```
step 1: O(n)
step 2: O(n+1)
step 3: O(n+2)
...
step m: O(n+m-1)

Total over m steps = O(m×n + m²)

When m << n (short output, long context): ≈ O(m×n)
When m ≈ n  (very long generation):       ≈ O(n²)

"Decode is O(n)" is a simplification that treats m as a constant.
More precisely: each step is O(n_current).
```

---

## 3. Full Data Flow: Prefill → Decode

### Where is the data at each stage?

```
Stage           Data                     Location              Operation
──────────────────────────────────────────────────────────────────────────
Input           tokenized prompt          CPU RAM               tokenization done

Before prefill  token embeddings          GPU HBM               embedding table lookup

During prefill  Q, K, V (per layer)       GPU HBM (transient)   X × Wq/Wk/Wv
                KV cache (per layer)      GPU HBM (persistent)  written, stays forever

After prefill   first token logits        GPU HBM               softmax → sample

Each decode     new token embedding       GPU HBM               embedding table lookup
step            new token K, V            GPU HBM               computed, appended to cache
                all historical K, V       HBM → registers       read from cache, do attention
                output logits             GPU HBM               sample → next token
```

### Decode step internals (step i in detail)

```
New token x_i, shape (1, d)

For each layer l in [0..31]:

  1. Linear projections (read model weights → compute in registers):
     Q_i = x_i × Wq_l   ← load Wq_l from HBM
     K_i = x_i × Wk_l   ← load Wk_l from HBM
     V_i = x_i × Wv_l   ← load Wv_l from HBM

  2. Append to KV cache (write HBM):
     kv_cache[l].K.append(K_i)
     kv_cache[l].V.append(V_i)

  3. Attention (read all historical KV — bandwidth bottleneck!):
     K_all = kv_cache[l].K   ← read n K vectors from HBM
     V_all = kv_cache[l].V   ← read n V vectors from HBM
     score = softmax(Q_i × K_allᵀ / √d_k)   ← shape (1, n), O(n) compute
     out   = score × V_all                   ← shape (1, d_v)

  4. FFN (read FFN weights):
     out = FFN(out)   ← load W_ffn from HBM

Output: logits → softmax → sample → token x_{i+1}
```

The step's compute (a few MFLOPS) completes almost instantly. The 34ms is entirely dominated by reading 271 GB from HBM.

---

## 4. Two Deployment Strategies: Combined vs Disaggregated

### 4.1 Combined Deployment — Traditional

```
┌──────────────────────────────────────┐
│      Single physical machine         │
│                                      │
│  GPU 0   GPU 1   GPU 2   GPU 3       │
│  ████    ████    ████    ████        │
│                                      │
│  Prefill and Decode share the same   │
│  GPU pool                            │
└──────────────────────────────────────┘

KV cache access:
  Prefill writes → Decode reads
  Same HBM, direct access
  Speed = HBM bandwidth (8 TB/s)
  Zero network transfer!

Problems:
  Prefill (compute-bound) and Decode (memory-bound)
  have completely different resource profiles.
  They interfere with each other:
    Incoming prefill preempts decode → decode stalls (jitter)
    Decode occupies GPU → prefill queues → TTFT spikes
  Average GPU utilization: 40–60%
```

#### Continuous Batching
logically its sequential for each batch, prefill then decode, but since its Tensor Parallelized, each batch passes through each GPU card sequetially in its own processing. We can do Continuos Batching:
```
Reality — Continuous Batching:

reqA prefill (1,d) and reqB's decode tokens (n,d) got combined and do the calculation together!

  Time  All 4 GPUs do together              reqA        reqB
──────────────────────────────────────────────────────────
t=1   prefill reqA                        processing  waiting
t=2   decode reqA step1 + prefill reqB    gen "on"    processing  ← BOTH at once! (interference, decode of reqA got compute power eaten by prefill reqB, delay in reqA output gen)
t=3   decode reqA step2 + decode reqB s1  gen "the"   gen "cat"
t=4   decode reqA step3 + decode reqB s2  gen "mat"   gen "sat"
...
```



### 4.2 Disaggregated Deployment — Production at Scale

```
Prefill Cluster                        Decode Cluster
┌──────────────────┐                   ┌──────────────────┐
│  Physical Node A  │                   │  Physical Node B  │
│                  │                   │                  │
│  High TFLOPS GPU │                   │  High BW GPU     │
│  (H100 SXM5)     │                   │  (H100 NVL)      │
│                  │                   │                  │
│  prompt → KV     │──InfiniBand────►  │  KV → tokens     │
│                  │  200–400 GB/s     │                  │
│  large batch OK  │  KV sent once!    │  KV in local HBM │
│  latency-tolerant│                   │  latency-critical│
└──────────────────┘                   └──────────────────┘

KV transfer size (Llama-3 70B, 32K context):
  = 32768 × 80 layers × 8 heads × 128 dim × 2 × 2 bytes = 5.4 GB
  Transfer time = 5.4 GB / 400 GB/s ≈ 13.5 ms  ← one-time cost only!

Important: the transfer uses InfiniBand / RoCE network bandwidth,
           NOT HBM bandwidth.
           Different physical machines → network RDMA, not HBM.
           Once received, decode reads KV from its own local HBM.
```

**Comparison:**

| Dimension | Combined | Disaggregated |
|---|---|---|
| GPU utilization | 40–60%, mutual interference | 80%+, **each optimized independently** |
| TTFT (time to first token) | Prefill delayed by decode | Prefill has dedicated GPUs, **lower TTFT** |
| TBT (inter-token latency) | Prefill preemptions cause jitter | Decode isolated, stable ~34ms |
| Scaling | Scale everything together, wasteful | Scale prefill/decode independently |
| KV access | HBM-local, extremely fast | **InfiniBand, 13.5ms one-time overhead** |
| Complexity | Simple, single process | **Complex, requires KV transfer protocol** |
---



## 5. KV Cache Sharding Under Tensor Parallelism

### Core design principle

**Model weights sharded by column (by head) ↔ KV cache sharded by column (by head)**

Both sharded along the same dimension means each GPU's attention computation is fully self-contained — no cross-GPU KV access needed.

### How model weights are sharded (column-parallel)

```
Wk shape = (d_model, n_KV_heads × head_dim)
         = (4096,    8 × 128)
         = (4096,    1024)

Column-parallel split across 4 GPUs (split by head):

  GPU0: Wk[:, head 0,1]  = (4096, 256)   ← owns KV heads 0 and 1
  GPU1: Wk[:, head 2,3]  = (4096, 256)   ← owns KV heads 2 and 3
  GPU2: Wk[:, head 4,5]  = (4096, 256)   ← owns KV heads 4 and 5
  GPU3: Wk[:, head 6,7]  = (4096, 256)   ← owns KV heads 6 and 7

Wq is split the same way across Q heads:
  32 Q heads / 4 GPUs = 8 Q heads per GPU
```

### How KV cache is sharded (column-parallel, by head)

```
K cache shape = (n_tokens, n_KV_heads, head_dim)
              = (1M,       8,          128)

Visualized as 2D (flatten head and dim):

         head0  head1  head2  head3  head4  head5  head6  head7
        [128   |128   |128   |128   |128   |128   |128   |128  ]
  tok0  [      |      |      |      |      |      |      |     ]
  tok1  [      |      |      |      |      |      |      |     ]
  ...
  tokN  [      |      |      |      |      |      |      |     ]

Split vertically (by head column):

  GPU0: (1M, 2, 128)  ← head 0,1 for ALL tokens
  GPU1: (1M, 2, 128)  ← head 2,3 for ALL tokens
  GPU2: (1M, 2, 128)  ← head 4,5 for ALL tokens
  GPU3: (1M, 2, 128)  ← head 6,7 for ALL tokens
```

The key insight: **KV cache is sharded vertically (by head) because the model is sharded horizontally (by head).** The sharding dimension is the same, so everything lines up.

### Why this symmetry makes each GPU self-contained

```
Decode step, GPU0's perspective:

Input: new token x, broadcast to all GPUs via NVLink
       (every GPU receives the full embedding)

GPU0 operates entirely independently:

  Step 1: project using its own weight shards:
    Q_heads_0_7  = x × Wq_0_7    → (1, 8×128)
    K_new_heads01 = x × Wk_01    → (1, 2×128)
    V_new_heads01 = x × Wv_01    → (1, 2×128)

  Step 2: append to its own KV cache shard:
    kv_cache_GPU0.append(K_new_01, V_new_01)
    ← writes to its own HBM only, no other GPUs involved!

  Step 3: attention using its own KV cache:
    K_all_01 = kv_cache_GPU0.K   ← reads from its own HBM!
    V_all_01 = kv_cache_GPU0.V   ← reads from its own HBM!
    att = softmax(Q × K_all_01ᵀ) × V_all_01

← Weight sharded by head column = KV cache sharded by head column
← Dimensions align = attention is fully local to each GPU
← No cross-GPU KV reads needed! NVLink communication minimized!
```

### When cross-GPU communication IS needed

```
No communication needed (each GPU independent):
  Attention computation      ← symmetric KV sharding, local HBM
  KV cache reads and writes  ← own HBM, no interference

NVLink communication required:
  Broadcast input x          ← once per step, leader → all GPUs
  AllGather attention output  ← concat each GPU's partial attention result
  AllReduce FFN output        ← sum partial FFN results across GPUs

Communication volume per layer:
  just a few KB of activation data
  NVLink at 600 GB/s transmits in < 1μs
← completely negligible compared to reading 271 GB of KV cache
```

### KV cache size distribution across GPUs

```
Per-GPU KV cache (Llama-3 70B, 1M context, 4-way TP):

  K cache per GPU = 1M tokens × 32 layers × 2 heads × 128 dim × 2 bytes
                  = 16.4 GB

  V cache per GPU = 16.4 GB  (same as K)

  KV cache per GPU = 32.8 GB

  4 GPUs total    = 32.8 × 4 = 131.2 GB  ← matches our earlier calculation!

H100 80GB VRAM allocation breakdown:
  Model weight shard:   35 GB   ← fixed at load time
  KV cache pool:       ~40 GB   ← managed dynamically by PagedAttention
  Activations:           3 GB   ← transient, reused across layers
  System overhead:       2 GB
  ─────────────────────────────
  Total:               80 GB
```

---

## Quick Reference Numbers

| Parameter | Value | Notes |
|---|---|---|
| Model | Llama-3 70B fp16 | 140 GB weights |
| GPU | B200 | 4500 TFLOPS, 8 TB/s HBM |
| Context | 1M tokens | |
| KV cache size | 131 GB | across 4 GPUs |
| Prefill time | ~31 seconds | compute-bound |
| Prefill throughput | ~32,000 tokens/sec | fully parallel |
| Decode time per token | ~34 ms | memory-bound |
| Decode throughput | ~29 tokens/sec | sequential generation |
| Speed gap | ~1,100× | prefill vs decode |
| KV cross-node transfer (32K ctx) | 13.5 ms | InfiniBand 400 GB/s, one-time |

---

## Further Reading

- [vLLM — Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Splitwise — Prefill-Decode Disaggregation (Microsoft Research)](https://arxiv.org/abs/2311.18677)
- [DistServe — Disaggregating Prefill and Decoding for Goodput Optimization (PKU)](https://arxiv.org/abs/2401.09670)
- [Megatron-LM — Efficient Large-Scale Language Model Training with Tensor Parallelism](https://arxiv.org/abs/1909.08053)
