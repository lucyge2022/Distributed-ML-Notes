# Chapter 2: Batching, Sharding & Caching

> Reference implementation: [prepare_dataset.py](https://github.com/lucyge2022/distributed-ml-examples/blob/main/ddp-testrun/prepare_dataset.py)

---

## Are dataset formats framework-agnostic?

**Yes — the stored format and the consuming framework are separate concerns.**

The files on disk (`.parquet`, `.tar`, `.mds`) are just files. What ties them to a framework is the **loader library** you use to read them. Model-specific preprocessing (tokenization, image augmentation, feature engineering) is applied at load time as a transform/map step — it is never baked into the stored format. The same `.parquet` files could in principle be read by Ray, Spark, pandas, or DuckDB.

```
Stored format (files on disk)
        ↓  loader library reads + streams
Batches in memory (raw samples)
        ↓  transform / map function applied
Model-ready tensors (tokenized, normalized, etc.)
        ↓
Training loop
```

---

## Comparison at a Glance

| | Ray Dataset | WebDataset | MosaicML (MDS) |
|---|---|---|---|
| File format | `.parquet` | `.tar` shards | `.mds` shards + `index.json` |
| Sharding unit | Row count (configurable) | Sample count per tar | Byte size limit |
| Batching direction | Horizontal (rows) | Sequential within shard | Horizontal, with random access |
| Random access | Yes (columnar index) | No (sequential only) | Yes (offset index per shard) |
| Remote streaming | Yes (S3, GCS, HDFS) | Yes (S3, GCS, HTTP, local) | Yes (S3, GCS, Azure, local) |
| Shuffle | Distributed shuffle, sort-based | Shuffle buffer (in-memory window) | Epoch-consistent, multiple algos |
| Primary consumer | Ray Train / Ray Data | PyTorch DataLoader | PyTorch DataLoader |
| Resume from checkpoint | Via Ray lineage | Manual (shard position) | Built-in (tracks seen samples) |

---

## 1. Ray Dataset + Parquet

### Format

Parquet is a **columnar binary format**:
- Stores data column-by-column rather than row-by-row
- Built-in compression (Snappy, Gzip, Zstd)
- Schema-enforced — every file knows its column names and types
- Supports **predicate pushdown** — you can filter and project columns before reading any data into memory

```
shard-00000-of-00004.parquet
shard-00001-of-00004.parquet
...
```

Each shard is an independent Parquet file. Ray points at the whole directory.

### Sharding pattern

Sharding is **horizontal — by row count**:

```
Full dataset: 200,000 rows
  ↓ split into 4 shards of 50,000 rows each
shard-00000: rows 0–49,999
shard-00001: rows 50,000–99,999
...
```

Vertical selection (loading only specific columns) is also possible via column projection — Parquet's columnar layout means unused columns are never read from disk.

### Consuming with Ray

```python
import ray

ds = ray.data.read_parquet("/path/to/parquet/")
# __id__ column present for shuffle tracking

# apply preprocessing as a map (lazy — not executed yet)
ds = ds.map(lambda row: tokenize(row["text"]))

# batch for training
for batch in ds.iter_torch_batches(batch_size=32):
    ...
```

### Features

- **Lazy materialization** — `read_parquet()` doesn't load data; it builds a logical plan. Data only moves when you iterate.
- **Parallel I/O** — Ray reads multiple shards across workers simultaneously, overlapping I/O with compute
- **Prefetching** — `prefetch_batches=N` keeps N batches ready ahead of the training loop
- **Built-in caching** — `.materialize()` pins a dataset in Ray object store memory so it's not re-read from disk on each epoch
- **Distributed shuffle** — Ray can globally shuffle across all shards using a sort-based algorithm; no single-node bottleneck

---

## 2. WebDataset + `.tar` Shards

### Format

WebDataset packs samples into standard `.tar` archives. Each sample is a **group of files sharing a key prefix**:

```
shard-000000.tar
  └── 000000/000000.txt    ← text content of sample 0
  └── 000000/000000.json   ← metadata of sample 0
  └── 000000/000001.txt    ← sample 1
  └── 000000/000001.json
  ...
```

The `.tar` format is universal — any tool that reads tar files can inspect the data. There is no custom binary encoding; samples are just files inside an archive.

### Sharding pattern

Sharding is **by sample count** — write N samples per tar, then start a new one:

```python
SHARD_SIZE = 5_000  # samples per shard

shard-000000.tar  ← samples 0–4,999
shard-000001.tar  ← samples 5,000–9,999
...
```

There is **no random access** within a shard — reading is sequential. To shuffle, WebDataset uses a **shuffle buffer**: load K samples into memory, pick one at random, replace it with the next sample from the stream.

```python
.shuffle(1000)  # keep 1000 samples in buffer, randomly pop from it
```

### Consuming with PyTorch

```python
import webdataset as wds

dataset = (
    wds.WebDataset("data/webdataset/shard-{000000..000004}.tar")
    .shuffle(1000)               # in-memory shuffle buffer
    .decode()                    # decode bytes to Python types
    .to_tuple("__key__", "txt", "json")
    .map(lambda key, txt, meta: tokenize(txt))  # model-specific preprocessing
)

loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

WebDataset implements PyTorch's `IterableDataset` — it plugs directly into `DataLoader`.

### Features

- **Remote streaming** — can read directly from `s3://`, `gs://`, or HTTP URLs without downloading first; shards are fetched on demand
- **No local storage required** — useful when dataset is larger than local disk
- **Simple format** — `.tar` is readable by any OS tool; no special library needed to inspect raw files
- **Multi-worker friendly** — each DataLoader worker takes a different shard, no coordination needed
- **Limitation** — no random access, no epoch-consistent shuffle; shuffle quality depends on buffer size

---

## 3. MosaicML StreamingDataset + `.mds` Shards

### Format

MDS is a **custom binary format** designed specifically for large-scale ML streaming:

```
mds/
  index.json          ← manifest: shard list, column schemas, sample counts
  shard.00000.mds     ← binary shard file
  shard.00001.mds
  ...
```

Each `.mds` shard has an internal **offset table** — a lookup that maps sample index → byte offset within the file. This enables **true random access** within a shard, unlike `.tar` which must be read sequentially.

### Sharding pattern

Sharding is **byte-size based** — `MDSWriter` keeps writing rows until the shard hits `size_limit` bytes, then starts a new shard:

```python
# from prepare_dataset.py:
MDSWriter(out=str(out), columns=columns, size_limit=shard_size * 2048)
# size_limit = 5000 samples × 2048 bytes ≈ 10MB per shard → 111 shards for ~1.1GB
```

This means shards have roughly equal byte sizes but potentially different sample counts (longer texts → fewer samples per shard).

### Consuming with PyTorch

```python
from streaming import StreamingDataset
from torch.utils.data import DataLoader

dataset = StreamingDataset(
    local="data/mds",           # local cache dir
    remote="s3://my-bucket/mds", # optional: stream from remote
    shuffle=True,
    shuffle_algo="py1s",         # epoch-consistent shuffle algorithm
)

loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### Shuffle algorithms

MDS ships multiple shuffle algorithms as a first-class feature — unlike WebDataset where shuffle is an afterthought:

| Algorithm | Tradeoff |
|---|---|
| `py1s` | Downloads 1 shard at a time, low memory, weaker shuffle |
| `py2s` | Downloads 2 shards at a time, better shuffle |
| `naive` | Full in-memory shuffle, best quality, high memory |

### Features

- **Random access** — offset index allows seeking to any sample without reading the whole shard; enables proper epoch-consistent shuffling
- **Download cache management** — automatically downloads shards from remote storage to local cache as needed; LRU eviction when cache is full
- **Epoch-consistent shuffle** — every epoch sees all samples exactly once, in a different order; no duplicates or skips
- **Resume from checkpoint** — tracks exactly which samples were consumed; training can resume mid-epoch after a crash
- **Schema-enforced** — `index.json` records column names and types; reader validates on load
- **Tightly integrated with LLM training stacks** — used in production at MosaicML/Databricks for training models like MPT

---

## Choosing Between Them

| If you… | Use |
|---|---|
| Already use Ray for orchestration / distributed training | **Ray + Parquet** |
| Have a giant unstructured dataset (images, audio, video, text) and want simplest format | **WebDataset** |
| Need epoch-consistent shuffle, checkpoint resume, and remote streaming for LLM training | **MosaicML MDS** |
| Need to run SQL-style queries or column projections on the dataset | **Parquet** (only columnar format) |
| Want zero infrastructure — just read from S3 with no local disk | **WebDataset** (lowest overhead) |
