# Chapter 2: Dataset

---

## 1. Dataset Composition

A dataset is split into two parts:

| Split | Purpose |
|---|---|
| **Training set** | Model learns from these — weights are updated based on this data |
| **Test set** (verification set) | Verify how well the model learned — never seen during training |

**Using MNIST as example:** https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data

```
total 107344
drwxr-xr-x  3 root  staff        96 Mar 17 13:58 t10k-images-idx3-ubyte
-rw-r--r--  1 root  staff   7840016 Mar 17 13:58 t10k-images.idx3-ubyte
drwxr-xr-x  3 root  staff        96 Mar 17 13:58 t10k-labels-idx1-ubyte
-rw-r--r--  1 root  staff     10008 Mar 17 13:58 t10k-labels.idx1-ubyte
drwxr-xr-x  3 root  staff        96 Mar 17 13:58 train-images-idx3-ubyte
-rw-r--r--  1 root  staff  47040016 Mar 17 13:58 train-images.idx3-ubyte
drwxr-xr-x  3 root  staff        96 Mar 17 13:58 train-labels-idx1-ubyte
-rw-r--r--  1 root  staff     60008 Mar 17 13:58 train-labels.idx1-ubyte

train-images.idx3-ubyte  ← TRAINING images (47MB)
train-labels.idx1-ubyte  ← TRAINING labels (60KB)

t10k-images.idx3-ubyte   ← TEST images (7.8MB)
t10k-labels.idx1-ubyte   ← TEST labels (10KB)
```

> `t10k` = "test 10,000" — 10,000 test images

| | Count | Share |
|---|---|---|
| Training | 60,000 | 85% |
| Test | 10,000 | 15% |
| **Total** | **70,000** | |

---

## 2. Binary File Format (IDX)

These files use **IDX format** — a simple binary format invented for MNIST.

### Images file (`train-images.idx3-ubyte`)

```
File size: 47,040,016 bytes

HEADER (16 bytes):
  bytes 0-3:   magic number  = 2051    (means "this is images")
  bytes 4-7:   num images    = 60,000
  bytes 8-11:  num rows      = 28
  bytes 12-15: num cols      = 28

DATA (after header):
  60,000 × 28 × 28 = 47,040,000 bytes
  47,040,000 + 16 header = 47,040,016 ✓ matches file size!
```

Each byte = one pixel:
- `0` = black
- `255` = white
- values in between = grey shades

> This is why **Parquet** (or other unstructured datalake format) can be used to house unstructured datasets like images and video — it stores raw bytes per row just as well as structured columns.

### Labels file (`train-labels.idx1-ubyte`)

```
File size: 60,008 bytes

HEADER (8 bytes):
  bytes 0-3:  magic number = 2049  (means "this is labels")
  bytes 4-7:  num labels   = 60,000

DATA:
  60,000 × 1 byte = 60,000 bytes
  60,000 + 8 header = 60,008 ✓ matches file size!
```

Each byte = one label value:

| Value | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| ... | ... |
| 9 | Ankle boot |

---

## 3. Tensor Representation

When loaded into an ML framework, raw bytes become tensors:

- **Images tensor** — shape `(60000, 28, 28)`: 60,000 images, each 28×28 pixels, each pixel a value 0–255
- **Labels tensor** — shape `(60000,)`: a flat list of 60,000 label values

```
tensor([9, 0, 0, 3, 0, 2, 7, 2, 5, 5, 0, 9, ...])
         ↑ each value is the class label for that image
```

---

## 4. Data Ingestion Flow

> Refer to [load_dataset_sample_flow_explained.py](https://github.com/lucyge2022/distributed-ml-examples/blob/main/ddp-testrun/load_dataset_sample_flow_explained.py) for showing a simple breakdown of dataset loading process.


```
Binary file (on SSD)
    ↓ read bytes
In-memory array (NumPy)
  shape: (60000, 28, 28)  dtype: uint8
    ↓ convert + normalize
tf.Tensor / torch.Tensor
  shape: (60000, 28, 28)  dtype: float32
  divide by 255 → values become 0.0 to 1.0
    ↓ batch
Mini-batch tensor
  shape: (32, 28, 28)  ← 32 images at a time
    ↓ flatten for simple model
  shape: (32, 784)     ← 28×28 = 784 pixels per image
    ↓ feed into model
Forward pass!
```

```
Raw format → CPU RAM → GPU VRAM → Forward pass
```

**Why normalize (divide by 255)?**
These values flow through matrix multiplications. If they stay at 0–255, gradients explode. Keeping them at 0.0–1.0 keeps the math stable.

**Labels** are only used during loss calculation — not fed into the forward pass.

### Optional: Data Ingestion via GDS (GPU Direct Storage)

```
Raw format → GPU VRAM → Forward pass
```

With NVIDIA GDS support (`cuFile` API), data can be loaded directly from NVMe storage into GPU VRAM, bypassing CPU RAM entirely — eliminates a full copy in the pipeline.

---
[TODO] add link to ddp-testrun for dataset breakdown + feeding example

## 5. NumPy vs Pure Python

NumPy is a Python library for fast mathematical operations on arrays and matrices.

| | Pure Python | NumPy |
|---|---|---|
| Execution | Interpreted line by line | Written in C under the hood |
| Loops | Slow, not optimized for math | Operates on entire arrays at once |
| CPU instructions | General | Uses SIMD instructions |
| Parallelism | None | Same concept as GPU parallelism, but on CPU |

---

## 6. Why NumPy → Tensor (not just NumPy)?

> **Most important:** NumPy has no autograd. It doesn't remember intermediate computation results, so it cannot do backprop.
>
> A **Tensor** remembers every operation applied to it (the computation graph), which is what makes gradient computation possible.

```
Raw format → NumPy (CPU RAM) → Tensor (GPU VRAM) → Forward pass
                ↑                     ↑
         fast math, C-backed    autograd-capable, runs on GPU
```
