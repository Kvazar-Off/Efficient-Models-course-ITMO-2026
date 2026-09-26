# Homework 1 - Analytical performance model of a small CNN

## Student - Trifonov Sergei

## Hardware / software

| item | value |
|---|---|
| GPU | NVIDIA H100 (96 GB), UUID in `results/metadata.json` |
| memory budget | `torch.cuda.set_per_process_memory_fraction(2e9/total)` → 2 GB budget per process (`--mem-limit-gb`, default 2), so the large-S/large-B corner genuinely OOMs (see the OOM section) |
| driver / torch CUDA | 595.84 / 12.8 |
| PyTorch | 2.8.0+cu128, Python 3.12.3, NumPy 2.3.3, SciPy 1.16.2, matplotlib 3.11, nvidia-ml-py 12.535 |
| protocol flags | `cudnn.benchmark=False`, `cudnn.allow_tf32=False`, `cuda.matmul.allow_tf32=False`; allocator cache flushed (`empty_cache`) between configs |

## Files

```
hw1/
├── README.md            # GPU / software versions, how to reproduce, results summary, 1-page discussion
├── derivations.md       # typed derivation draft (to be rewritten by hand)
├── models.py            # the network
├── equations.py         # flops(), memory(), latency(), energy()
├── measure.py           # full measurement grid -> results/measurements.csv
├── calibrate.py         # fits theta on the base grid -> results/theta.json
├── make_figures.py      # all plots -> results/figures/
├── hw1_colab.ipynb      # one-click full pipeline for Google Colab (T4)
└── results/
    ├── measurements.csv # S, B, latency, memory, energy, is_validation, status(ok/OOM)
    ├── theta.json       # fitted parameters
    └── figures/*.png
```

## How to reproduce

```bash
python3 measure.py --resume    # GPU; ~45 min; 2 GB budget (--mem-limit-gb), resumable
python3 calibrate.py           # CPU only
python3 make_figures.py        # CPU only
```

## The four closed forms

Derived for the given net (18 CUDA kernels per forward, W = 1 040 324 params, all activations are const · B · S²):

1. **FLOPs** (1 MAC = 2 FLOPs; ReLU/MaxPool = 0 FLOPs):
   `flops(S,B) = B · (17 714 · S² + 313 700)`
2. **Memory** - peak `torch.cuda.max_memory_allocated()` inside one forward.
   The peak is at MaxPool, because PyTorch always materialises the pool's
   int64 argmax indices (2·S² elements × 8 bytes): input 3S² + pool input
   8S² + pool output 2S² + indices (4S² fp32-units) = 17·S² per batch element:
   `memory(S,B) = 4 · (1 040 324 + 17 · B · S²)` bytes
3. **Latency** - every kernel i takes max(launch, compute, memory) time:
   `latency(S,B,θ) = Σ_{i=1..18} max(t_launch, FLOPs_i / P, Bytes_i / BW)`,
   θ = (t_launch, BW, P)
4. **Energy** - three physical channels, linear model:
   `energy(S,B,θe) = 18·e_kernel + e_flop·FLOPs + e_byte·Bytes`,
   θe = (e_kernel, e_flop, e_byte)

Full derivation for the handwritten sheet: `hw1_handwritten.pdf`.

## OOM: equation vs reality

By the equation a config does not fit a budget L when
`memory(S,B) > L ⇔ B·S² > (L/4 − 1 040 324)/17`.
Uncapped, the largest grid point needs ≈ **4.57 GB** — so even a 16 GB T4
would see *zero* OOMs and this step of the
assignment would be vacuous. Therefore `measure.py` runs every config under a
**2 GB per-process budget** (`--mem-limit-gb`, implemented as
`set_per_process_memory_fraction(2e9/total_memory)` — portable: 2 GB means 2 GB
both on this H100 and on a Colab T4).

Predicted OOM set (`B·S² > 2.94·10⁷`): **(384,256), (448,256),
(496,{126,128,141,256}), (512,{115,126,128,141,256})** — 11 configs.
Observed in `measurements.csv`: **exactly the same 11 rows, `status=OOM`,
11/11 with zero classification errors** (`results/figures/oom_boundary.png`).
The largest surviving point (448,141) measured 1.96 GB — passed with a 40 MB margin.

Two ways the equation still breaks around this boundary:

* **Reserved vs live bytes.** The fraction limit is enforced on allocator
  *segments*, not on `memory(S,B)`-style live tensors. In the first capped
  attempts (448,115) OOMed although the equation gives only 1.57 GB — the
  cached blocks of the previous config ate the budget. Protocol fix:
  `empty_cache()` after every config; without it the observed boundary sits
  systematically *below* the predicted one.
* **Workspaces move the true boundary** by the constant + cuDNN jumps
  (below), so a point the equation puts 1–3 % under the budget can still fail.

## Calibrated θ

Fit on the 60 measurable base-grid points (3 corner configs were OOM),
validated on 61 unseen points.

```
latency: t_launch = 3.0e-5 s   BW = 1.94e11 B/s   P = 3.08e13 FLOP/s
energy:  e_kernel = 0          e_flop  = 4.6e-12 J/FLOP   e_byte ≈ 0
```

| quantity | MAPE (calib. grid) | MAPE (unseen points) |
|---|---|---|
| latency | 12.0 % | 11.9 % |
| energy (above the noise gate only: 17 / 24 pts) | 53.6 % | 35.6 % |
| memory | median relative error 5.2 % for >500 MB points (raw MAPE is a small-denominator artefact) | |

## What the equations get right

* `latency_vs_B.png`, `latency_vs_S.png`, `latency_pred_vs_measured.png`,
  `latency_map.png`: the max-term model tracks both grids within ~12 % on
  points never used for calibration. Regime switching appears exactly where the
  model puts it (`regimes.png`): a flat 18·t_launch ≈ 0.55 ms plateau at small
  S·B (launch-bound) → ∝B·S² slope (memory-bound). Compute-bound is *not
  reached inside this grid on this GPU*: the net does ~47 FLOP per byte
  (17 714/380) while the fitted crossover BW/P sits at ~158 FLOP per byte —
  FP32 CUDA-core convs with heuristic algorithms cannot saturate 96 GB of HBM.
  On a T4 the hardware crossover is 300 GB/s ÷ 8.1 TF/s ≈ 27 FLOP/B < 47 —
  so the same equations predict a compute-bound corner: the regime story is GPU-dependent exactly as intended.
* `oom_boundary.png`: 11/11 agreement of the OOM set with the memory equation
  (given a clean allocator state between configs).
* Memory: verified exact at the largest point in a reference uncapped run —
  predicted 4 567 MB vs measured 4 601 MB at (S=512, B=256); the int64 MaxPool
  indices are the dominant term nobody expects.

## Where and why they break

1. **Memory: +33…40 MB constant** (median residual 38.5 MB) - cuDNN v9 engine
   workspaces + 2 MB allocator rounding, invisible to a tensor-lifetime
   model. It dominates relative error at small sizes.
2. **Memory: deterministic cuDNN heuristic flips.** Under the 2 GB budget:
   up to +772 MB over the equation at (256,115); in the uncapped reference run
   a +1.7 GB cliff at (256, B≥115) appeared *byte-exact reproducible* across
   two independent runs. Algorithm choice (and hence workspace size) depends
   on shape *and available memory* — not expressible via (S,B) algebra.
3. **Latency: ~30–40 % error for B=1** - 18 eager kernels pay host dispatch
   between launches; the single fitted t_launch (30 µs, ~4x the launch cost
   on an idle card) absorbs it.
4. **Energy: the sensor floor.** Whole-GPU delta power of launch-bound
   configs is ~1 W against ±3–4 W background on a 330 W idle card: 47 of
   132 configs record 0, another 33 fail the burst-consistency gate (median
   of 2 isolated bursts, spread < 50 %). Only the memory-bound corner is
   measurable at all.
5. **Energy: 35–55 % scatter.** (a) FLOPs and Bytes are collinear on this grid
   (both ∝ B·S²), the (e_flop, e_byte) split is not identifiable - two
   independent measurement runs picked opposite terms yet similar MAPE;
   (b) DVFS: under the shared power/thermal budget clocks droop inside long
   bursts, energy grows superlinearly with B·S² (a power law E = α·t^1.31
   explains the gated data with ~35 % MAPE), which no linear model can do.

## Data honesty (shared GPU, memory budget)

* This H100 also serves other processes. Mitigations: CUDA-event medians
  for latency; energy = median of isolated bursts with a util == 0 % gate
  before each burst, baseline subtracted from the NVML energy integral
  counter; `empty_cache()` between configs; measurements scheduled when
  neighbours were idle.
* The 2 GB per-process budget (`--mem-limit-gb`) is an intentional emulation
  of a low-grade GPU so the OOM step of the assignment is non-vacuous; on an
  actual 16 GB T4 nothing in this grid OOMs (max 4.57 GB by the equation —
  the no-budget OOM boundary is B·S² > 2.35·10⁸, far outside the grid).
* Energy of launch-bound points is below the measurable floor *of this
  hardware*; on a 70 W T4 the same protocol would resolve them.
