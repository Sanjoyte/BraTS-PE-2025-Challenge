# HOW_IT_WORKS.md — what this project currently does, and exactly how

A detailed technical walkthrough of the BraTS-PED 2025 baseline as it stands
today. This is the *explanatory* document:

| Document | Answers |
|---|---|
| `CLAUDE.md` | What the project is, attribution, cleanup log, status — kept short |
| `run_commands.md` | **How to run it** — copy-pasteable commands in dependency order |
| **`HOW_IT_WORKS.md`** (this file) | **What each part does and how it works internally** |

Every claim below was read off the code in this repo. Where something is
implemented but has never been executed, it is marked **⚠ UNVERIFIED** rather
than described as working.

---

## Table of contents

1. [What the project is](#1-what-the-project-is)
2. [The data](#2-the-data)
3. [The label and region scheme — the central contract](#3-the-label-and-region-scheme--the-central-contract)
4. [The configuration layer](#4-the-configuration-layer)
5. [Stage 1 — nnU-Net dataset conversion](#5-stage-1--nnu-net-dataset-conversion)
6. [Stage 2 — nnU-Net preprocessing](#6-stage-2--nnu-net-preprocessing)
7. [Stage 3 — HFF data staging and frequency decomposition](#7-stage-3--hff-data-staging-and-frequency-decomposition)
8. [The three ensemble members](#8-the-three-ensemble-members)
9. [Stage 4 — Training](#9-stage-4--training)
10. [Stage 5 — Inference](#10-stage-5--inference)
11. [Stage 6 — Ensembling](#11-stage-6--ensembling)
12. [Stage 7 — Evaluation](#12-stage-7--evaluation)
13. [The orchestrator](#13-the-orchestrator)
14. [Memory safety](#14-memory-safety)
15. [Skull stripping (inactive by default)](#15-skull-stripping-inactive-by-default)
16. [What has actually been run so far](#16-what-has-actually-been-run-so-far)
17. [Known gaps and correctness notes](#17-known-gaps-and-correctness-notes)

---

## 1. What the project is

The project segments paediatric brain tumours from four co-registered MRI
modalities, producing a voxel-wise label map per case. It is a **thesis
baseline**: it reproduces a published 1st-place BraTS 2025 Pediatric solution,
keeping that method intact, so that a novel contribution can be added on top and
measured against it.

The published method is an **ensemble of three segmentation networks**, combined
in probability space rather than by voting:

| Member | Network | What makes it different |
|---|---|---|
| A | nnU-Net (3D full-res U-Net) | Adjustable weight-initialization scale (`-gamma`) |
| B | Swin UNETR | Transformer backbone + BraTS 2021 transfer learning |
| C | HFF-Net | Frequency-domain dual-branch (DTCWT low / NSCT high) |

**What this repo adds over the published code:** the published repo could not
run as shipped. Two of the three inference command builders were never defined,
the ensemble driver called the nnU-Net builder three times into one directory,
and the ensembler used `argmax`, which is wrong for this training scheme
(§11). The dataset conversion, configuration layer, ensemble glue, HFF inference,
and run book in this repo are all additions. The networks, losses, and training
schemes are untouched.

---

## 2. The data

### 2.1 Layout as delivered

```
raw_dataset/BraTS-PEDs-v1/
├── Training/     257 cases
│   └── BraTS-PED-00001-000/
│       ├── BraTS-PED-00001-000-t1n.nii.gz    native T1
│       ├── BraTS-PED-00001-000-t1c.nii.gz    contrast-enhanced T1
│       ├── BraTS-PED-00001-000-t2w.nii.gz    T2-weighted
│       ├── BraTS-PED-00001-000-t2f.nii.gz    T2-FLAIR
│       └── BraTS-PED-00001-000-seg.nii.gz    ground-truth labels
└── Validation/    91 cases — same, minus the -seg file (labels not public)
```

Every volume is **240 × 240 × 155 at 1 mm isotropic**, already co-registered
across modalities.

### 2.2 The data is already skull-stripped

This was measured, not assumed, and it matters because the pipeline contains a
skull-stripping stage that would otherwise be mandatory.

A naïve `voxel > 0` test is misleading here: the volumes have a low but nonzero
noise floor left by interpolation, so the four modalities appear to have
*different* masks. Thresholding at 2 % of each volume's maximum instead, the four
modalities agree on a single brain mask with **Dice 0.92–0.98**, occupying ~30 %
of the volume, with a sharp air→tissue transition and no bright scalp-fat ring.
That is a skull-stripped brain with interpolation blur.

**Consequence:** skull stripping is **off by default** (§15).

### 2.3 Three incompatible naming conventions

This is the single most common source of silent failure in this pipeline,
because a wrong name yields an empty dataset rather than an error.

| Consumer | Expects | Example |
|---|---|---|
| Raw data | hyphens | `BraTS-PED-00001-000-t1c.nii.gz` |
| nnU-Net v2 | numeric channel suffix | `BraTS-PED-00001-000_0001.nii.gz` |
| nnU-Net v1 (skull strip) | numeric, **different order** | `..._0002.nii.gz` for t1c |
| HFF-Net | underscores | `BraTS-PED-00001-000_t1c.nii.gz` |

The nnU-Net v2 channel order is fixed by `peds/labels.py::CHANNEL_ORDER`:

```
0000 = t1n    0001 = t1c    0002 = t2w    0003 = t2f
```

nnU-Net v1's skull-stripping model uses a *different* order
(`0000=t2f, 0001=t1n, 0002=t1c, 0003=t2w`), hardcoded separately in
`skull_stripping/skull_stripping.py`. The two must not be conflated.

---

## 3. The label and region scheme — the central contract

This section is the most important one to understand; everything downstream
depends on it.

### 3.1 Raw voxel labels

| Value | Name | Structure |
|---|---|---|
| 0 | — | background |
| 1 | ET | enhancing tumour |
| 2 | NET | non-enhancing tumour |
| 3 | CC | cystic component |
| 4 | ED | peritumoral oedema |

### 3.2 Why the network is not trained on these labels

BraTS does not score the four disjoint labels. It scores **overlapping
structures** — whole tumour, tumour core, enhancing tumour — because those are
what is clinically meaningful and what the label boundaries are most reliable
about. So the network is trained on *regions*: six binary channels, each with an
independent sigmoid, rather than a 5-way softmax.

This is nnU-Net's built-in "region-based training" mode, and it is what the
original authors' dataset name (`..._6regions`) referred to.

### 3.3 The six regions

Defined once in `peds/labels.py`. **Declaration order is part of the contract.**

```python
REGIONS = {
    "WT":  [1, 2, 3, 4],   # whole tumour
    "TC":  [1, 2, 3],      # tumour core
    "ED":  [4],            # oedema
    "NET": [2],
    "CC":  [3],
    "ET":  [1],            # enhancing tumour
}
REGIONS_CLASS_ORDER = [4, 2, 4, 2, 3, 1]
```

### 3.4 How a label map is reconstructed

nnU-Net decodes region predictions by walking the regions **in declaration
order** and painting `REGIONS_CLASS_ORDER[i]` wherever region *i* exceeds 0.5.
Later regions overwrite earlier ones, so ordering coarse → fine reconstructs the
original label space:

| Step | Region fires | Paints | Reasoning |
|---|---|---|---|
| 1 | WT | 4 (ED) | everything tumour-ish starts as oedema |
| 2 | TC | 2 (NET) | the core defaults to non-enhancing |
| 3 | ED | 4 | oedema keeps its own value |
| 4 | NET | 2 | |
| 5 | CC | 3 | |
| 6 | ET | 1 | most specific, painted last, wins |

**This was verified, not assumed.** Encoding real ground truth into the six
region channels and decoding it back reproduced the original label map exactly on
**20/20 tested training cases**. So predictions come out directly as
`{0,1,2,3,4}` — valid BraTS-PED submissions, no post-hoc remapping.

### 3.5 Why it lives in one module

`peds/labels.py` is imported by the dataset conversion, the HFF predictor, and
the inference runner. Training writes this ordering into `dataset.json`;
inference decodes with it. A second, drifting copy would not crash — it would
silently produce wrong labels. Centralising it makes that impossible.

`peds/labels.py` also exposes `decode_regions()`, a standalone reimplementation
of the same walk, used by the HFF predictor which has no nnU-Net `LabelManager`
available.

---

## 4. The configuration layer

### 4.1 Files

| File | Role |
|---|---|
| `.env.example` | Committed template, documents every setting |
| `.env` | Your machine's actual values — **gitignored** |
| `peds/config.py` | Parses `.env`, exposes the typed `CFG` object |
| `scripts/peds_env.sh` | `source` this to export `.env` into a shell |

### 4.2 How resolution works

`peds/config.py` implements a dependency-free `.env` parser (no
`python-dotenv` needed) supporting `KEY=value`, `#` comments, and `${OTHER}`
interpolation resolved against earlier lines then the live environment.

Importing `peds.config` also pushes `nnUNet_raw`, `nnUNet_preprocessed`, and
`nnUNet_results` into `os.environ`, so any subprocess spawned from Python gets a
correctly configured nnU-Net CLI without the caller exporting anything.

### 4.3 Precedence, and why it is deliberately asymmetric

**`.env` overrides pre-existing environment variables.** This is the opposite of
the usual dotenv default, and it is intentional: a stale
`export nnUNet_raw=/old/path` left in a teammate's shell profile must not
silently redirect the whole pipeline somewhere else. This exact failure happened
during development — stale exports pointed at a deleted directory.

The one exception is `SHELL_WINS = {"CUDA_VISIBLE_DEVICES"}`, so the ordinary
`CUDA_VISIBLE_DEVICES=1 python ...` idiom keeps working for picking a GPU.

Full precedence, lowest to highest:

```
ambient shell environment  <  .env  <  CLI flags
                              (except CUDA_VISIBLE_DEVICES, where the shell wins)
```

CLI flags beating config is why `runner_ped2025.py` exposes `--trainer`,
`--checkpoint`, and `--fold`.

### 4.4 Quoting

All values in `.env` are quoted. This repo lives under a path containing a
space (`mr adit`), and `set -a; . ./.env` mis-parses unquoted values with
spaces — it silently assigns a truncated value and then tries to execute the
remainder as a command.

---

## 5. Stage 1 — nnU-Net dataset conversion

**Script:** `scripts/prepare_nnunet_dataset.py`
**Input:** `raw_dataset/BraTS-PEDs-v1/`
**Output:** `$nnUNet_raw/Dataset501_BraTSPED/`

### 5.1 What it does

Turns the per-case folder layout into the flat layout nnU-Net requires:

```
$nnUNet_raw/Dataset501_BraTSPED/
├── imagesTr/   BraTS-PED-00001-000_0000.nii.gz … _0003.nii.gz   (1028 files)
├── labelsTr/   BraTS-PED-00001-000.nii.gz                       (257 files)
├── imagesTs/   validation split, unlabelled                     (364 files)
└── dataset.json
```

### 5.2 How

1. Iterate case directories in each split.
2. Skip, with a warning, any case missing a modality (or a segmentation, for the
   training split) — a partial case is never silently half-converted.
3. For each modality, create a **relative symlink** named by `CHANNEL_ORDER`.
4. Write `dataset.json` containing `channel_names`, the region `labels` block,
   `regions_class_order`, `numTraining`, and `file_ending`.

### 5.3 Why symlinks

The raw dataset is 33 GB. Copying it would double that for no benefit, since
nnU-Net only ever reads these files. Symlinking makes `$nnUNet_raw` **6.6 MB**
instead of 33 GB. Relative symlinks keep the working directory relocatable.

`--copy` is available for filesystems that cannot follow symlinks.

**Consequence:** `$nnUNet_raw` is meaningless on another machine and should
*not* be uploaded to shared storage — regenerate it instead (it takes seconds).

### 5.4 Options

| Flag | Effect |
|---|---|
| `--limit N` | Convert only the first N cases per split (use ≥6; nnU-Net builds a 5-fold split) |
| `--copy` | Copy instead of symlink |
| `--no-validation` | Skip the unlabelled split |

---

## 6. Stage 2 — nnU-Net preprocessing

**Command:** `nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -c 3d_fullres -np 2`
**Output:** `$nnUNet_preprocessed/Dataset501_BraTSPED/` (**17 GB** for 257 cases)

Three phases, all stock nnU-Net:

1. **Integrity verification** — every case has all channels, geometry is
   consistent, labels are within the declared set.
2. **Fingerprint extraction** — per-case shapes, spacings, and intensity
   statistics collected into `dataset_fingerprint.json`.
3. **Planning + preprocessing** — nnU-Net derives the network topology and
   training configuration into `nnUNetPlans.json`, then normalizes and
   resamples every case.

### What it derived for this dataset

| Property | Value |
|---|---|
| Patch size | 96 × 160 × 160 |
| Batch size | 2 |
| Network stages | 6 |
| Features per stage | 32, 64, 128, 256, 320, 320 |
| Normalization | Z-score per channel |
| Spacing | 1 × 1 × 1 mm (no resampling — data is already isotropic) |

The 3d_lowres configuration is dropped automatically: the images are small
enough that a low-resolution stage would be nearly identical to full-res.

**File format note:** preprocessed volumes are **blosc2 `.b2nd`** files
(nnU-Net ≥ 2.6), not `.npz` — 514 `.b2nd` (data + seg per case) plus 257
`.pkl` property files. Prediction *probability* maps are a different thing and
*are* `.npz` (§10).

---

## 7. Stage 3 — HFF data staging and frequency decomposition

HFF-Net does not consume MRI volumes directly. It consumes each modality
**decomposed into frequency bands**: one low-frequency volume and four
directional high-frequency volumes, giving 4 + 16 = 20 input channels.

### 7.1 Staging — `scripts/prepare_hff_dataset.py`

Mirrors every case into `$PEDS_HFF_DATA/<split>/<case>/` with **underscore**
naming (`<case>_t1c.nii.gz`), via symlinks. This is the naming gap the upstream
HFF README warns about; getting it wrong yields an empty dataset, not an error.

It also writes `train.txt` / `val.txt` — newline-delimited absolute paths to
case folders, the format HFF's loader reads — using a seeded shuffle with a
configurable validation fraction (default 0.2).

### 7.2 Low frequency — DTCWT (Python)

Run by the same script, via HFF's own `DTCWT_LF.py`.

**How:** the Dual-Tree Complex Wavelet Transform is applied **slice by slice**
in 2D across the axial axis. At `nlevels=1` the lowpass output is the same
in-plane size as the input, so the volume shape is preserved
(240 × 240 × 155). Each slice's lowpass is min-max normalized to [0, 255] and
cast to `uint8`, then slices are stacked back into a volume saved as
`<case>_<modality>_L.nii.gz`.

DTCWT is chosen over an ordinary wavelet transform because it is approximately
**shift-invariant** — a small translation of the input does not scramble the
coefficients, which matters for a segmentation task.

### 7.3 High frequency — NSCT (MATLAB)

**Script:** `src/HFF/NSCT_BTS/nsct_hf.m`, driven by `scripts/run_nsct.sh`.

The Nonsubsampled Contourlet Transform decomposes each axial slice into
**4 directional high-frequency sub-bands**, capturing oriented texture/edge
detail that a separable wavelet cannot represent well.

Parameters (unchanged from the original authors):

```matlab
nlevels = [2, 2];    % first-level high-frequency decomposition, 4 directions
pfilt   = 'pyrexc';  % pyramid filter
dfilt   = 'cd';      % directional filter
```

Per slice it calls `nsctdec`, takes the four first-level directional sub-bands,
applies `imadjust` with `stretchlim(·, [0.01 0.99])` for contrast
normalization, and stacks the results into
`<case>_<modality>_H1..H4.nii.gz`.

**What this repo changed:** the file was a script with the original author's
personal absolute paths and a hardcoded `BraTS20_Training_*` folder glob. It is
now a **function** `nsct_hf(baseDir, nsct_tbx_dir, casePattern)`. The
decomposition body is byte-identical; only the header and the glob changed, plus
a guard excluding `.`/`..` from the directory listing.

**Hard dependency:** MATLAB with the Image Processing Toolbox (`imadjust`,
`stretchlim`). The bundled NSCT toolbox ships prebuilt `.mexa64` binaries, so no
compilation is needed on x86-64 Linux. `nsct_hf.m` uses `parfor`, which
degrades gracefully to a serial loop without the Parallel Computing Toolbox.

`scripts/run_nsct.sh` checks for MATLAB and fails with a clear message rather
than a MATLAB stack trace if it is missing.

### 7.4 Checking coverage

```bash
python scripts/prepare_hff_dataset.py --split Training --check
```

Reports how many cases have all 16 high-frequency volumes. Must read `257/257`
before HFF training will work.

---

## 8. The three ensemble members

### 8.1 Member A — nnU-Net with adjustable initialization scale

This is the original authors' first contribution, and it is a **weight
initialization** change, not an architecture change.

**Where:** `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`,
`nnunetv2/run/run_training.py`.

**What it does.** After the network is built, every convolutional and linear
layer is re-initialized:

```python
def init_weights(m):
    gamma = self.plans_dict["std_gamma"]
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d,
                      nn.ConvTranspose3d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0, std=m.weight.size(1) ** (-gamma))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
```

so weights are drawn from `N(0, σ)` with

```
σ = weight.size(1) ^ (-gamma)
```

**A precision note worth stating exactly.** `weight.size(1)` is the **input
channel count**, which is *not* the same as fan-in for a convolution:

| Layer | Weight shape | `size(1)` | True fan-in |
|---|---|---|---|
| `Conv3d(16, 32, k=3)` | (32, 16, 3, 3, 3) | **16** | 16 × 27 = 432 |
| `Linear(64, 128)` | (128, 64) | **64** | 64 |

So for linear layers this is fan-in scaling, but for convolutions it scales by
input channels only, ignoring the kernel volume. Larger `gamma` ⇒ smaller
initial weights. The submission used **`gamma = 0.7`**.

**How the value travels.** This is worth understanding because it has a side
effect:

1. `nnUNetv2_train ... -gamma 0.7` calls `modify_std_gamma()`, which
   **writes `"std_gamma": 0.7` into `nnUNetPlans.json` on disk**.
2. The trainer reads `plans_dict["std_gamma"]` at network build time.
3. If it is a number (not `""`), the re-initialization runs and the trainer logs
   `The distribution of model weights have been changed with std rate = 0.7!!!`
4. The output folder gets a `__std_gamma0.7` suffix, so different gammas do not
   collide.

**At prediction time, `-gamma` does something different.** It does *not*
re-initialize anything — the weights come from the checkpoint. It only appends
`__std_gamma0.7` to the model folder path so the right results directory is
found. Omitting it at prediction time makes nnU-Net look in a folder that does
not exist.

⚠ **Caveat (see §17):** because step 1 rewrites the shared plans file, two
concurrent trainings with different gammas on the same dataset would race.

### 8.2 Member B — Swin UNETR

**Where:** `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_Swinunetr_Xepoch.py`.

The trainer keeps all of nnU-Net's data pipeline, loss, and schedule, and swaps
only the network by overriding `build_network_architecture()` to return MONAI's
`SwinUNETR` (from the pip `monai` package):

```python
SwinUNETR(img_size=(96, 160, 160), in_channels=4,
          out_channels=6, feature_size=48, use_checkpoint=True)
```

Two supporting details:

* `enable_deep_supervision = False` — SwinUNETR has no deep-supervision heads.
* `set_deep_supervision_enabled()` is **overridden to a no-op**, because the
  base trainer would otherwise try to set `network.decoder.deep_supervision`
  and crash on a backbone that has no such attribute.

**The `_1005epochs` variant** is the submission configuration:

* `img_size=(128, 128, 128)` — needs the matching plans file, hence
  `-p nnUNetPlansPatch` (§9.2). The trainer prints a reminder.
* `initial_lr = 1e-3`
* Warm-starts from a **BraTS 2021** checkpoint: weights are loaded with
  `module.` prefixes stripped and shape-mismatched keys filtered out, then
  applied with `strict=False`, so only the compatible backbone transfers.

**What this repo changed:** the checkpoint path was hardcoded to the original
authors' machine (`/mlcube_project/...`) inside a `try/except`. On any other
machine the load failed and the model **silently fell back to random
initialization** — quietly dropping the transfer learning the trainer exists
for. It now reads `PEDS_SWIN_PRETRAINED` from the environment and prints which
mode it is in. Leaving the variable empty keeps random init deliberately, and
says so.

### 8.3 Member C — HFF-Net

**Where:** `src/HFF/` (vendored third-party, IEEE TMI 2025).

A **dual-branch 3D network**: one branch over the 4 low-frequency channels, one
over the 16 high-frequency channels, fused at depth.

```
model(low[B,4,128,128,128], high[B,16,128,128,128])
  -> (out_LF, out_HF, side_LF, side_HF)
```

Four components:

**Adaptive Laplacian Convolution (ALC).** The high-frequency branch's input
layer (`HighFreqConv`) is a residual 3×3×3 conv, `x + conv(x)`, whose weights
are **initialized to a 3D Laplacian kernel** (centre −6, six face-neighbours
+1). During training an L2 term pulls it back toward that Laplacian target:

```python
reg_loss = torch.sum((model.input_ed.conv.weight
                      - model.laplacian_target) ** 2) * reg_weight
```

so the layer can adapt to the data while retaining its edge-detecting
behaviour. Periodically, **Fisher information** is computed for this layer
(accumulated squared gradients over the training set) and the most important
weights are reset toward the Laplacian target — elastic weight consolidation
applied to keep the filter from drifting away from its purpose.

**Frequency-Domain Cross-Attention (FDCA).** Applied at the two deepest stages
of both branches. It transforms features into the Fourier domain, attends
there, and transforms back:

```python
freq = torch.fft.fftn(x, dim=(2,3,4), norm='ortho')
# three attentions computed on the real part, applied to BOTH real and imaginary:
#   semantic   — channel attention
#   positional — spatial attention
#   slice      — per-slice attention (low-rank, with uncertainty)
x = torch.fft.ifftn(torch.complex(real, imag), dim=(2,3,4), norm='ortho').real
```

Attending in the frequency domain gives each attention a global receptive field
by construction, and the slice attention specifically targets the anisotropy of
MRI volumes.

**Frequency-Domain Decomposition (FDD).** The DTCWT/NSCT split of §7 — done
offline, not in the network.

**Training objective** (`src/HFF/train.py`):

```
loss = sup(out_LF) + sup(out_HF)          # main heads vs ground truth
     + sup(side_LF) + sup(side_HF)        # side heads vs ground truth
     + unsup_weight · FF(out_LF, out_HF)  # inter-branch consistency
     + reg_loss                           # ALC Laplacian anchor
```

where `sup` is Dice loss and `FF` is a **Focal Frequency Loss** comparing the
two branches' outputs in the frequency domain. `unsup_weight` ramps linearly
from 0 to its maximum (default 15) across training, so the branches specialise
before being forced to agree.

---

## 9. Stage 4 — Training

### 9.1 Invocation

```bash
bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_1000epochs -gamma 0.7
```

Positional arguments are dataset ID, configuration, fold. nnU-Net builds a
**5-fold cross-validation split** with a fixed seed (12345); folds are
independent and can be trained on different machines simultaneously.

### 9.2 The results folder name is a composite key

```
nnUNetTrainer_1000epochs__nnUNetPlans__3d_fullres__std_gamma0.7
└── trainer class      └── plans   └── config    └── gamma suffix
```

Every one of those four components must be reproduced at prediction time
(`-tr`, `-p`, `-c`, `-gamma`) or nnU-Net will look in a directory that does not
exist. This is the most common operational error with this pipeline.

Member B additionally needs `-p nnUNetPlansPatch`, a plans variant created by
copying `nnUNetPlans.json`, setting `patch_size = [128,128,128]` and a distinct
`data_identifier`, then running `nnUNetv2_preprocess -plans_name nnUNetPlansPatch`.

### 9.3 What a run produces

```
fold_0/
├── checkpoint_best.pth     best EMA pseudo-Dice
├── checkpoint_final.pth    last epoch
├── training_log_*.txt
├── progress.png
└── validation/             predictions + summary.json on the held-out fold
```

Training logs a per-region pseudo-Dice each epoch, in `REGIONS` order —
`[WT, TC, ED, NET, CC, ET]`.

---

## 10. Stage 5 — Inference

### 10.1 nnU-Net members

```bash
nnUNetv2_predict -i <images> -o <out> -d 501 -c 3d_fullres \
  -tr <trainer> -gamma 0.7 --save_probabilities
```

Sliding-window inference with Gaussian patch weighting and test-time mirroring,
then resampling back to the original geometry. Per case it writes:

| File | Contents |
|---|---|
| `<case>.nii.gz` | Label map, native BraTS labels |
| `<case>.npz` | `probabilities`, float32 `(6, 155, 240, 240)` |
| `<case>.pkl` | Geometry/properties |

**`--save_probabilities` is mandatory for ensembling** — the ensembler averages
probabilities, not labels.

The six channels are **independent sigmoids**, one per region. They do **not**
sum to 1. This is the fact that makes `argmax` wrong (§11).

### 10.2 HFF member — `scripts/hff_predict.py` ⚠ UNVERIFIED

Upstream HFF ships `eval.py`, which **cannot** feed an ensemble: it requires
ground truth, computes only metrics, writes no predictions, and concatenates
patch tensors without reassembling volumes. (It also references an
`args.output_dir` that it never defines.) This script is the missing inference
half, written for this project.

It reconciles two mismatches:

**Geometry.** It reproduces HFF's training-time preprocessing exactly — drop the
first 3 axial slices (`ThrowFirstZ`), per-volume min-max normalize to
[−1, 1], then take a **brain-centred deterministic 128³ crop** (the bounding-box
centre of non-background voxels, clamped inside the volume). The model runs
once on that crop, and the result is pasted back onto the full grid with
background elsewhere, so every member shares one array shape.

**Label space.** HFF emits a per-class softmax over the 5 BraTS-PED labels,
while the nnU-Net members emit 6 region channels. Class probabilities are summed
into region space — a region's probability is the total mass of its constituent
labels:

```
P(WT) = p1+p2+p3+p4      P(NET) = p2
P(TC) = p1+p2+p3         P(CC)  = p3
P(ED) = p4               P(ET)  = p1
```

The two branch heads are averaged after softmax
(`(softmax(out_LF) + softmax(out_HF)) / 2`), since the training objective drives
both toward the same target.

**Status:** the crop/paste-back geometry and the class→region mapping have been
unit-tested, but the script has **never been run against a real checkpoint**,
because that requires a trained HFF model, which requires the MATLAB NSCT step.

---

## 11. Stage 6 — Ensembling

**Module:** `ensemble/ensemble.py`

### 11.1 What it does

```bash
python ensemble/ensemble.py \
  --members $PEDS_PREDICTIONS/{nnunet,swin,hff} \
  --model-dir <a trained results folder> \
  -o $PEDS_PREDICTIONS/ensemble
```

1. Intersect the case IDs present in every member directory — only cases *all*
   members predicted are ensembled.
2. For each case, load each member's `probabilities` array and compute a
   (optionally weighted) mean. Shapes are checked; a mismatch raises rather
   than broadcasting silently. A missing `.npz` raises `FileNotFoundError`
   rather than quietly dropping that member.
3. Convert the averaged probabilities to a label map.
4. Transpose `(Z,Y,X) → (X,Y,Z)` and save with the affine and header taken from
   a member's own `.nii.gz`, preserving geometry.

### 11.2 The correctness point: why not `argmax`

The published `ensemble.py` did `np.argmax(prob, axis=0)`. That is only valid
for a softmax over mutually exclusive classes. Here the six channels are
**independent sigmoids over overlapping regions** that do not sum to 1, and
`argmax` over them is meaningless — it would return a region index, not a label,
and would systematically lose the nesting structure.

The rewrite delegates to nnU-Net's own `LabelManager`:

```python
label_manager = PlansManager(plans).get_label_manager(dataset_json)
seg = label_manager.convert_probabilities_to_segmentation(prob)
```

which applies `regions_class_order` correctly — and still does the right thing
if the label scheme is ever changed to a plain softmax setup. This is why
`--model-dir` is required: it supplies `plans.json` and `dataset.json`, i.e. the
label scheme the probabilities were produced under.

### 11.3 How it was verified

Ensembling a member **with itself** reproduced that member's own segmentation
**bit-for-bit on 8/8 cases**, including the affine. Separately, the weighted
mean was unit-tested against a hand-computed expectation. Output labels were
confirmed to be a subset of `{0,1,2,3,4}`.

---

## 12. Stage 7 — Evaluation

```bash
nnUNetv2_evaluate_folder <labelsTr> <predictions> -djfile <model>/dataset.json -pfile <model>/plans.json
```

Writes `summary.json` with per-region Dice and IoU, per case and averaged.
Because the dataset is region-based, the region keys are tuples of raw labels:

| Key | Region |
|---|---|
| `(1,)` | ET enhancing tumour |
| `(2,)` | NET non-enhancing |
| `(3,)` | CC cystic |
| `(4,)` | ED oedema |
| `(1, 2, 3)` | TC tumour core |
| `(1, 2, 3, 4)` | WT whole tumour |

---

## 13. The orchestrator

**Script:** `runner_ped2025.py` — Stages 5 and 6 in one command.

Originally a challenge-container entry point hardcoded to `/input`,
`/tmp/inputs`, `/tmp/outputs`, `/output`, executing on import with no `main()`.
Now a configurable CLI.

Four phases:

1. **Prepare** — flatten per-case folders into nnU-Net's `_0000.._0003` layout.
   The rename map is *derived* from `peds.labels.CHANNEL_ORDER`, so it cannot
   drift from the conversion used at training time.
2. **Skull-strip** — skipped unless `--skull-strip` (§15).
3. **Predict** — `runner/runner.py::model_runner()` runs each requested member
   into **its own** directory.
4. **Ensemble** — average and write final label maps.

```bash
# preview commands without executing
python runner_ped2025.py --input raw_dataset/BraTS-PEDs-v1/Validation --dry-run

# one member, pinning trainer/checkpoint/fold
python runner_ped2025.py --input <dir> --members nnunet \
  --trainer nnUNetTrainer_1000epochs --checkpoint best --fold 0
```

### What `runner/runner.py` fixes

The published version was broken in three ways:

| Published | Now |
|---|---|
| `model_runner()` called `nnunet_runner()` **three times** into one folder | Each member has its own builder and output directory |
| `swin_runner` / `hff_runner` **never defined** — `fast_model_runner` was a guaranteed `NameError` | Both defined |
| Everything hardcoded to `/tmp` and `/output` | Derived from `CFG`, overridable per call |

The builders encode each member's requirements: the nnU-Net member passes
`-gamma`; the Swin member selects `nnUNetPlansPatch` when the trainer name
contains `1005` and passes no gamma (its initialization is its own); the HFF
member shells out to `scripts/hff_predict.py` and reads from the **staged HFF
tree**, not the flat nnU-Net layout.

`fold="ensemble"` (the default) omits `-f` entirely, which makes nnU-Net average
every trained fold.

---

## 14. Memory safety

The workstation this runs on has **16 GB of RAM** and was hard-crashing during
training and preprocessing. Three layers now prevent that.

### 14.1 Hard ceiling — `scripts/capped.sh`

Runs any command inside a transient systemd scope with a cgroup memory limit:

```bash
systemd-run --user --scope -p MemoryMax=11G -p MemorySwapMax=2G -- "$@"
```

An overrun is killed with **exit code 137** instead of freezing the desktop.
`MemorySwapMax` stops the job from thrashing swap instead of failing fast. If
systemd user scopes are unavailable it warns and runs uncapped rather than
refusing.

**Verified** by deliberately allocating 3 GB under a 1 GB cap: the process was
killed with 137, the machine was unaffected.

### 14.2 Worker limits

nnU-Net sizes its pools from the CPU count — on this 12-core machine that is 12
dataloader workers and 8 general workers, each holding full 3D patches. Pinned
in `.env`:

```
nnUNet_n_proc_DA=2      nnUNet_def_n_proc=2
PEDS_NP_PREPROCESS=2    PEDS_NP_PREDICT=1
```

### 14.3 Single-threaded BLAS

`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` — nnU-Net does its own
parallelism, and nested thread pools waste memory as well as CPU.

### 14.4 Measured peaks

| Stage | Peak RAM (of 15.5 GB) |
|---|---|
| Preprocessing, 257 cases | 9.7 GB |
| Training, 1 epoch, 257 cases | 8.6 GB |
| Prediction | 9.0 GB |

All of these should be raised on a larger machine.

---

## 15. Skull stripping (inactive by default)

**Module:** `skull_stripping/skull_stripping.py`. Uses **nnU-Net v1**
(`src/nnUNet-1.7.1/`) with a `Task070_autosegm` brain-mask model.

How it works: per case, copy modalities into a temp dir renamed to nnU-Net
**v1's** channel order, run `nnUNet_predict` to get a brain mask, multiply each
modality by the mask, and write the result out under nnU-Net **v2**'s naming.
Cases are sharded evenly across available GPUs via a multiprocessing `Pool`.

**Why it is off:** BraTS-PED ships already skull-stripped (§2.2). It is also
unusable as-is — the `Task070_autosegm` model is not in this repo. Enable with
`--skull-strip` only for data that still has a skull, and supply that model.

This repo replaced a large commented-out `argparse` block with a real working
CLI, and made `cuda_ids` a parameter instead of always claiming every GPU.

---

## 16. What has actually been run so far

Verified end-to-end on this machine, memory-capped throughout:

| Stage | Status | Evidence |
|---|---|---|
| Authors' nnU-Net installed editable, `-gamma` live | ✅ | `-gamma` in `--help`; `nnunetv2` resolves inside the repo |
| Memory cap kills overruns | ✅ | exit 137 on a deliberate 1 GB overrun |
| Dataset conversion | ✅ | 257 train + 91 val; `$nnUNet_raw` = 6.6 MB |
| Region scheme round-trip | ✅ | exact on 20/20 real ground-truth cases |
| Preprocessing, **all 257 cases** | ✅ | exit 0, integrity verified, 17 GB |
| Training, `gamma 0.7` active | ✅ smoke only | 1 epoch, 205 train / 52 val |
| Prediction with `--save_probabilities` | ✅ | 8 cases, `(6,155,240,240)` float32 |
| Evaluation | ✅ | per-region Dice in `summary.json` |
| Ensemble | ✅ | self-ensemble reproduces a member 8/8, bit-for-bit |
| `runner_ped2025.py` end to end | ✅ | 3 cases, prepare → predict → ensemble |
| HFF staging (underscore naming + splits) | ✅ | 257 cases staged |
| HFF DTCWT low-frequency | ⚠ partial | **2 of 257** cases |
| HFF NSCT high-frequency | ❌ | **0 of 257** — MATLAB unavailable |
| HFF training / inference | ❌ | blocked on the above |
| Swin UNETR training | ❌ | commands written, not run here |

### Current artifacts on disk

`$PEDS_WORK` defaults to `${PEDS_ROOT}/peds_work` — inside the repo, gitignored.

```
$PEDS_WORK/
├── nnUNet_raw/            6.6 MB   symlinks, 257 + 91 cases
├── nnUNet_preprocessed/   17 GB    all 257 cases + nnUNetPlans{,Patch}.json
├── nnUNet_results/        472 MB   ONE smoke model (nnUNetTrainer_1epoch, gamma 0.7)
├── hff_data/              33 MB    257 staged; 2 with _L; 0 with _H1..H4
├── hff_results/           —        does not exist
└── predictions/           524 MB   nnunet/, ensemble/, eval_tr/; hff/ and swin/ empty
```

### On the numbers produced so far

The only trained model is a **deliberately tiny 1-epoch smoke run**. Its scores
exist to prove the plumbing works, and are **not** meaningful results:

| Region | Dice (52 held-out cases, 1 epoch) |
|---|---|
| WT whole tumour | 0.647 |
| TC tumour core | 0.621 |
| NET non-enhancing | 0.567 |
| ET / CC / ED | 0.000 |
| **foreground mean** | **0.306** |

The pattern — bulk structures partly learned, the three small/rare ones not at
all — is exactly what one epoch produces, and confirms the label plumbing is
correct. Real numbers require the 1000-epoch runs (`run_commands.md` Steps 6–8),
which take days per member on one GPU.

---

## 17. Known gaps and correctness notes

Ordered by how likely each is to bite.

**1. HFF-Net is blocked on MATLAB.** `matlab` is not on this machine's `PATH`.
Without it there are no NSCT high-frequency volumes, so HFF cannot be trained or
run, and the three-member ensemble cannot be produced. Set `PEDS_MATLAB` in
`.env` if MATLAB lives elsewhere.

**2. HFF's class count is wrong for BraTS-PED.** `src/HFF/train.py` computes
`classnum = 4 if class_type == 'all' else 2`, but BraTS-PED has **five** label
values (background + ET/NET/CC/ED). This must be raised to 5 before a checkpoint
will match `scripts/hff_predict.py --num-classes 5`. Left unmodified because it
is method code — flagged rather than silently edited.

**3. `scripts/hff_predict.py` has never executed against a real model.** Its
geometry and label mapping are unit-tested; the end-to-end path is not.

**4. Swin UNETR falls back to random init** unless `PEDS_SWIN_PRETRAINED` points
at a BraTS 2021 checkpoint. It now says so loudly, but the transfer learning
that distinguishes this member is absent until you supply that file.

**5. `-gamma` rewrites the shared plans file.** Every `nnUNetv2_train` launch
calls `modify_std_gamma()`, which writes `std_gamma` into `nnUNetPlans.json`.
Two concurrent trainings with *different* gamma values on the same dataset would
race on that file. Training different **folds** with the *same* gamma is safe.

**6. DTCWT output is quantized to `uint8`.** `DTCWT_LF.py` min-max normalizes
each slice to [0, 255] and casts to `uint8`, discarding precision, and does so
**per slice** rather than per volume, so inter-slice intensity relationships are
not preserved. This is upstream behaviour, left unmodified.

**7. `--limit` in `prepare_hff_dataset.py` applies after staging.** All cases are
staged (symlinked) regardless; the limit only truncates the DTCWT list. This is
why 257 cases are staged but only 2 have `_L` volumes.

**8. `nnUNetTrainer_1000epochs` is defined twice** in
`variants/training_length/nnUNetTrainer_Xepochs.py` (~lines 78 and 115); the
second shadows the first. Both set `num_epochs = 1000`, so behaviour is
unaffected. Left as-is — it is the authors' file.

**9. `Task070_autosegm` is not in the repo**, so skull stripping cannot run.
Not a blocker, since the data is already skull-stripped.

**10. `numpy` is pinned to 1.26.4** by `torchio`. torch 2.11.0+cu128 and nnU-Net
were verified working on it, but it blocks a numpy 2.x upgrade.

---

## Appendix — file map

| Path | Role |
|---|---|
| `peds/config.py` | `.env` parsing, path resolution, `CFG` |
| `peds/labels.py` | Label/region scheme + `decode_regions()` |
| `scripts/peds_env.sh` | Source to export `.env` into a shell |
| `scripts/capped.sh` | Hard memory cap wrapper |
| `scripts/prepare_nnunet_dataset.py` | Raw → nnU-Net layout |
| `scripts/prepare_hff_dataset.py` | Raw → HFF tree + DTCWT + splits |
| `scripts/run_nsct.sh` | MATLAB NSCT driver |
| `scripts/hff_predict.py` | HFF whole-volume inference ⚠ unverified |
| `runner/runner.py` | Per-member command builders |
| `ensemble/ensemble.py` | Region-aware probability averaging |
| `runner_ped2025.py` | Prepare → predict → ensemble CLI |
| `skull_stripping/` | nnU-Net v1 brain masking (inactive) |
| `src/nnUNet-official-new/` | nnU-Net v2, modified (`-gamma`, Swin trainer) |
| `src/nnUNet-1.7.1/` | nnU-Net v1, used only by skull stripping |
| `src/HFF/` | Vendored HFF-Net |
| `environment.yml` | Pinned dependencies (cu128 torch) |
