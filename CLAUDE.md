# CLAUDE.md — Project Memory

> Concise living document. Update it as the project moves; don't let it bloat.

## What this project is

Thesis baseline for **BraTS 2025 Pediatric (BraTS-PED)** brain tumour
segmentation. It starts from a public 1st-place challenge solution (attribution
below) and keeps that method — the nnU-Net / Swin UNETR / HFF-Net ensemble —
intact. What was removed is the original authors' identity/documentation
artifacts and dead code; what was added is the wiring the published repo lacked:
a dataset conversion, a central config, the missing ensemble glue, and a working
run book. My own novel contribution gets added on top of this baseline.

## Original work — attribution & citation

**Baseline repo (the code this project starts from)**
- Name: *Frequency-Aware Ensemble Learning for BraTS 2025 Pediatric Brain Tumor Segmentation*
- Paper: https://arxiv.org/pdf/2509.19353
- Challenge: BraTS 2025 Pediatric (BraTS-PED), https://www.synapse.org/Synapse:syn64153130/wiki/630130
- Result: 1st place, BraTS 2025 Pediatric Brain Tumor Segmentation Challenge;
  oral presentation at the MICCAI BraTS 2025 Challenge Workshop, 23 Sept 2025.
- License: MIT, `Copyright (c) 2025 Seauagain` (retained as `LICENSE`).

**Third-party components (each keeps its own upstream license)**

| Component | Upstream | License | Cite as |
|---|---|---|---|
| nnU-Net v2 (modified, vendored) | https://github.com/MIC-DKFZ/nnUNet | `src/nnUNet-official-new/LICENSE` (Apache-2.0) | Isensee et al., *Nat. Methods* 2021 |
| nnU-Net v1 1.7.1 (vendored) | https://github.com/MIC-DKFZ/nnUNet | `src/nnUNet-1.7.1/LICENSE` (Apache-2.0) | Isensee et al., *Nat. Methods* 2021 |
| Swin UNETR | MONAI `research-contributions`, `SwinUNETR/BRATS21` | Apache-2.0, ships with the pip `monai` package | Hatamizadeh et al., BrainLes 2021 |
| HFF-Net (vendored) | https://github.com/VinyehShaw/HFF | `src/HFF/LICENSE` | Shao et al., *IEEE TMI* 2025, doi:10.1109/TMI.2025.3579213 |
| NSCT toolbox (inside HFF) | `src/HFF/NSCT_BTS/nsct_toolbox` | see `NSCT_BTS/README` | Ganasala & Kumar, *J. Digit. Imaging* 2014 |

HFF-Net BibTeX (preserved from the upstream HFF README before it was removed):

```bibtex
@ARTICLE{11032150,
  author={Shao, Minye and Wang, Zeyu and Duan, Haoran and Huang, Yawen and Zhai, Bing
          and Wang, Shizheng and Long, Yang and Zheng, Yefeng},
  journal={IEEE Transactions on Medical Imaging},
  title={Rethinking Brain Tumor Segmentation from the Frequency Domain Perspective},
  year={2025}, pages={1-1}, doi={10.1109/TMI.2025.3579213}}
```

HFF-Net itself builds on XNet (Zhou et al., ICCV 2023) and Ganasala et al. (2014).

## Codebase map

```
.
├── runner_ped2025.py       End-to-end inference CLI: prepare -> members -> ensemble.
├── environment.yml         Pinned deps incl. the CUDA-matched (cu128) torch build.
├── peds/
│   ├── config.py           Single source of truth for paths/settings; reads .env.
│   └── labels.py           The BraTS-PED label/region scheme, defined once.
├── scripts/
│   ├── peds_env.sh             `source` this to load .env into a shell.
│   ├── capped.sh               Run any command under a hard RAM cap (systemd cgroup).
│   ├── prepare_nnunet_dataset.py  raw_dataset -> $nnUNet_raw/Dataset501_BraTSPED.
│   ├── prepare_hff_dataset.py     raw_dataset -> HFF tree + DTCWT low-freq + splits.
│   ├── run_nsct.sh                MATLAB high-frequency (NSCT) decomposition.
│   └── hff_predict.py             Whole-volume HFF inference (written here; see below).
├── runner/runner.py        Command builders for all three ensemble members.
├── ensemble/ensemble.py    Region-aware probability averaging -> final label maps.
├── skull_stripping/        nnU-Net v1 brain-mask inference (optional; see Data).
├── src/
│   ├── nnUNet-official-new/  nnU-Net v2, MODIFIED by the original authors (-gamma).
│   ├── nnUNet-1.7.1/         nnU-Net v1, used only by skull_stripping.
│   └── HFF/                  Vendored HFF-Net (frequency-domain member).
├── .env / .env.example     Machine-specific config (.env is gitignored).
└── raw_dataset/            BraTS-PED data (gitignored; pull from Drive).
```

## The method (kept, not modified)

Three segmentation models ensembled in probability space:

1. **nnU-Net with adjustable initialization scale.** The authors added a `-gamma`
   flag, persisted into the plans JSON as `std_gamma` and applied at network
   build time to re-initialize every conv/linear layer as
   `N(0, weight.size(1)^(-gamma))`, biases zeroed (`nnUNetTrainer.init_weights`).
   Note `size(1)` is the **input-channel count**, which equals fan-in only for
   `Linear`; for convolutions it ignores the kernel volume. Their submission used
   `gamma = 0.7`. The value is baked into the results folder name and **must be
   passed again at prediction time** — where it only selects the folder, it does
   not re-initialize anything.
2. **Swin UNETR with BraTS 2021 transfer learning.** `nnUNetTrainer_Swinunetr`
   swaps nnU-Net's backbone for MONAI `SwinUNETR` (feature_size 48, deep
   supervision off). The `_1005epochs` variant uses 128³ patches (needs
   `-p nnUNetPlansPatch`), `lr 1e-3`, and warm-starts from a BraTS 2021 checkpoint.
3. **HFF-Net.** Each modality is split into a shift-invariant low-frequency
   volume (DTCWT) and four directional high-frequency volumes (NSCT), fed to a
   dual-branch network with adaptive Laplacian convolution and frequency-domain
   cross-attention, trained with supervised + inter-branch consistency losses.

Everything above is out of scope for cleanup — it is the baseline being built on.

## Pipeline stages

| Stage | Entry point |
|---|---|
| Dataset conversion | `scripts/prepare_nnunet_dataset.py` |
| HFF data + frequency decomposition | `scripts/prepare_hff_dataset.py`, `scripts/run_nsct.sh` |
| Preprocess | `nnUNetv2_plan_and_preprocess -d 501` |
| Train | `nnUNetv2_train 501 3d_fullres <fold> -tr <trainer> -gamma <g>` |
| Evaluate | `nnUNetv2_evaluate_folder` |
| Infer | `nnUNetv2_predict ... --save_probabilities`, `scripts/hff_predict.py` |
| Ensemble | `ensemble/ensemble.py` |
| All of the above, one command | `runner_ped2025.py` |

Full copy-pasteable commands with dependencies live in **`run_commands.md`**.
A detailed walkthrough of what each stage does internally — the region contract,
the `-gamma` mechanics, the HFF frequency pipeline, the ensembling maths — lives
in **`HOW_IT_WORKS.md`**.

## Data

**`raw_dataset/` as provided (BraTS-PED v1):**

```
raw_dataset/BraTS-PEDs-v1/
├── Training/    257 cases: BraTS-PED-00001-000/
│                  BraTS-PED-00001-000-{t1n,t1c,t2w,t2f,seg}.nii.gz
└── Validation/   91 cases: same minus -seg.nii.gz (no public labels)
```

Volumes are 240×240×155 at 1 mm isotropic. **The data is already
skull-stripped** — verified by thresholding above the interpolation noise floor,
where the four modalities agree on one brain mask (Dice 0.92–0.98). Skull
stripping is therefore *off by default*; it is only needed for data that still
has a skull, and it requires an `Task070_autosegm` model that is not in this repo.

Naming conventions differ by consumer, which is a common source of silent
failure:
- Raw data uses **hyphens**: `<case>-t1n.nii.gz`.
- nnU-Net v2 wants `<case>_0000..0003.nii.gz`, order **0000=t1n, 0001=t1c,
  0002=t2w, 0003=t2f**.
- nnU-Net v1 skull stripping wants a *different* order: 0000=t2f, 0001=t1n,
  0002=t1c, 0003=t2w.
- HFF wants **underscores**: `<case>_<modality>.nii.gz`.

**Label scheme.** Voxel labels are `1`=ET, `2`=NET, `3`=CC, `4`=ED. Training uses
nnU-Net *regions* (the "6 regions" the authors' dataset name referred to):
WT, TC, ED, NET, CC, ET with `regions_class_order = [4, 2, 4, 2, 3, 1]`. Walking
the regions coarse→fine reconstructs the native label space exactly — verified
to round-trip on real ground truth — so predictions come out directly
submittable as `{0,1,2,3,4}`.

**Working directories** (all gitignored; a teammate needs to pull these from the
shared Drive rather than recompute them):

| Path | What goes in it | Drive folder |
|---|---|---|
| `raw_dataset/` | the challenge data | `brats-ped/raw_dataset/` |
| `$nnUNet_preprocessed` | fingerprint, plans, preprocessed volumes | `brats-ped/nnUNet_preprocessed/` |
| `$nnUNet_results` | trained checkpoints | `brats-ped/nnUNet_results/` |
| `$PEDS_HFF_DATA` | DTCWT/NSCT frequency volumes + splits | `brats-ped/hff_data/` |
| `$PEDS_HFF_RESULTS` | HFF checkpoints | `brats-ped/hff_results/` |
| `$PEDS_PREDICTIONS` | per-member probability maps, ensemble output | `brats-ped/predictions/` |

`$nnUNet_raw` is **not** worth uploading — it is symlinks into local
`raw_dataset/`, so it costs ~7 MB instead of 33 GB but is meaningless elsewhere.
Regenerate it with `scripts/prepare_nnunet_dataset.py` (seconds).

## Cleanup log

**Identity / documentation artifacts removed** (attribution extracted first,
into the section above):

| Removed | Why | Preserved where |
|---|---|---|
| `README.md` | Authors' title, paper link, leaderboard "News", acknowledgements | Attribution section above |
| `assets/` (`brats2025_logo.png`) | Banner for that README only | — |
| `src/HFF/README.md` | Third-party README: badges, author bio, citation | Citation above; DTCWT/NSCT steps, flags and the underscore-naming warning folded into `run_commands.md` |
| `src/HFF/figs/` (4 images) | Figures for the HFF README only | — |
| `src/nnUNet-official-new/readme.md`, `src/nnUNet-1.7.1/readme.md` | Upstream project READMEs | License table above |
| `src/nnUNet-official-new/官方代码` | Author's scratch note with their personal absolute path | The one fact it recorded is in this file |
| `src/nnUNet-official-new/nnunet_md540801.txt` | Author's md5 snapshot of their own tree | — |

**Dead code removed:**

| Removed | Evidence |
|---|---|
| `src/research-contributions/` (13 MB, 303 files) | Entire MONAI research-contributions repo, **imported by nothing**. Swin UNETR comes from the pip `monai` package. Held 12 unrelated projects (auto3dseg, DiNTS, DAE, SwinMM, UNETR, prostate-mri, SkullRec, coplenet, LAMP…). |
| `skull_stripping.py` commented `argparse` block + trailing usage string | Dead comment block; replaced with a real working CLI. The usage it documented is now in `run_commands.md`. |
| `nnUNetTrainer_Swinunetr_Xepoch.py` lines 53–116 | ~60-line commented-out `load_pretrained_weight`, superseded by the inline copy in `_1005epochs`. All 7 trainer classes verified intact afterwards. |
| `src/nnUNet-official-new/install0515.sh` | Two-line dated scratch installer; superseded by `run_commands.md` Step 0. |
| 2 stale notebooks | `prostate-mri-lesion-seg/build_and_run.ipynb`, `nnUNet-1.7.1/.../MIC-DKFZ.ipynb` — unrelated tasks. |
| `nsct_toolbox/decdemo.asv` | MATLAB editor autosave file. |

Repo shrank from 33 MB / 917 files to ~9 MB / ~600 files. **All `LICENSE` files
kept** — required by MIT/Apache-2.0 redistribution terms.

**Fixed / added (the published repo could not run as shipped):**

- `runner/runner.py` — `model_runner()` called `nnunet_runner()` three times into
  one folder, and `swin_runner`/`hff_runner` were never defined (`fast_model_runner`
  was a guaranteed `NameError`). All three builders now exist and write to
  separate directories.
- `ensemble/ensemble.py` — rewritten. It used `argmax`, which is wrong for
  region-based sigmoid outputs; it now uses nnU-Net's `LabelManager`, which is
  correct for both regions and plain softmax. Verified: self-ensembling a member
  reproduces that member's own segmentation bit-for-bit.
- `runner_ped2025.py` — was a container script hardcoded to `/input`, `/tmp/*`,
  `/output`; now a configurable CLI with `--trainer/--checkpoint/--fold/--dry-run`.
- `nsct_hf.m` — hardcoded personal paths and a BraTS-2020 folder glob became
  function arguments. The decomposition itself is unchanged.
- Added `peds/config.py`, `.env`/`.env.example`, `scripts/peds_env.sh`,
  `scripts/capped.sh`, and the dataset-preparation and HFF-inference scripts.

## Open questions / things flagged for review

1. **HFF-Net is wired but unverified end-to-end.** Two real gaps remain:
   - `src/HFF/train.py` computes its class count as `4 if class_type=='all'`, but
     BraTS-PED has **five** label values (background + ET/NET/CC/ED). That count
     has to be raised to 5 before an HFF checkpoint will match
     `scripts/hff_predict.py --num-classes 5`. Left untouched because it is
     method code — flagging rather than silently editing.
   - Upstream `src/HFF/eval.py` only prints metrics: it requires ground truth,
     writes no predictions, and never reassembles volumes, so it cannot feed the
     ensemble. `scripts/hff_predict.py` was written to fill that gap (brain-centred
     128³ window, class→region probability mapping). **It has never been executed**,
     because it needs a trained HFF checkpoint, which needs the MATLAB NSCT step.
2. **MATLAB is not installed on this machine** (`matlab` not on `PATH`), so
   Step 3.2 of `run_commands.md` and everything downstream of it (HFF training and
   the three-member ensemble) are written but unrun here.
3. **Swin UNETR transfer learning needs a checkpoint that is not in the repo.**
   `nnUNetTrainer_Swinunetr_1005epochs` had a hardcoded `/mlcube_project/...`
   path wrapped in `try/except`, so on any machine but the authors' it failed
   and silently fell back to random init — quietly dropping the transfer
   learning the trainer exists for. It now reads `PEDS_SWIN_PRETRAINED` from the
   environment and says out loud which mode it is in. To reproduce the authors'
   result, download the BraTS 2021 SwinUNETR checkpoint and set that variable;
   leaving it empty keeps random init deliberately.
4. **`nnUNetTrainer_1000epochs` is defined twice** in
   `variants/training_length/nnUNetTrainer_Xepochs.py` (lines ~78 and ~115); the
   second shadows the first. Both set `num_epochs = 1000`, so behaviour is
   unaffected. Left as-is — it is the authors' file.
5. **No trained weights ship with the repo**, including the `Task070_autosegm`
   skull-stripping model. Not a blocker: BraTS-PED is already skull-stripped.

## Conventions

- **Config:** one gitignored `.env` (template `.env.example`), read by
  `peds/config.py`. `.env` deliberately **overrides** stale shell exports for
  path variables — a leftover `export nnUNet_raw=...` must not silently redirect
  the pipeline — but `CUDA_VISIBLE_DEVICES` still lets the shell win, so
  `CUDA_VISIBLE_DEVICES=1 python ...` works. CLI flags beat both.
- **Paths:** `pathlib` throughout; no hardcoded absolute paths. Quote values in
  `.env` — this repo lives under a path containing a space.
- **Label scheme:** defined once in `peds/labels.py` and imported by the dataset
  conversion, the HFF predictor and the inference runner. The region ordering and
  `REGIONS_CLASS_ORDER` are a contract — training writes them into
  `dataset.json` and inference decodes with them, so a drifting second copy
  would silently corrupt predictions rather than fail loudly.
- **Dependencies:** pinned in `environment.yml`. `nnunetv2` is deliberately
  *not* listed there — it must be installed editable from
  `src/nnUNet-official-new` to get the authors' `-gamma` contribution.
- **Scripts:** bash + Python only, identical on native Linux and WSL2. No
  PowerShell/`.bat`. `.gitattributes` forces LF on `*.sh`/`*.py`/`*.m` and marks
  the prebuilt `.mexa64` NSCT binaries as binary.
- **WSL2:** the NVIDIA driver lives on the **Windows host** — teammates should
  install only the CUDA toolkit/libraries inside WSL, never a separate Linux
  NVIDIA driver. Keep the dataset and `PEDS_WORK` on the native Linux
  filesystem (`~/...`), not `/mnt/c/...`, where cross-filesystem I/O is much
  slower and bottlenecks preprocessing.

## This machine's constraints

- **16 GB RAM — this was crashing the desktop during training.** Mitigated in
  three layers:
  1. `scripts/capped.sh` runs any command in a systemd cgroup with
     `MemoryMax=$PEDS_MEM_MAX` (default 11G) and `MemorySwapMax=2G`. An overrun
     is killed with exit code **137** instead of freezing the machine —
     verified by deliberately overrunning a 1G cap.
  2. Worker pools pinned in `.env`: `nnUNet_n_proc_DA=2`, `nnUNet_def_n_proc=2`
     (nnU-Net otherwise derives 12 and 8 from the CPU count, and each worker
     holds full 3D patches). Stage worker counts: `PEDS_NP_PREPROCESS=2`,
     `PEDS_NP_PREDICT=1`.
  3. BLAS pinned single-threaded (`OMP/MKL/OPENBLAS_NUM_THREADS=1`).

  Measured peaks with these settings: **9.0 GB** during prediction, **9.7 GB**
  during full-dataset preprocessing, out of 15.5 GB. Raise all of these on a
  bigger machine.
- **GPU: RTX 5060 Ti 16 GB (Blackwell, `sm_120`)** — needs CUDA 12.8+ builds.
  The env runs `torch 2.11.0+cu128`. HFF's `requirements.txt` pin
  (`torch==2.1.2+cu118`) will **not** run on this card; install per
  `run_commands.md` Step 0.3 instead.
- Disk: 145 GB free after full preprocessing. `raw_dataset/` is 33 GB,
  `$nnUNet_preprocessed` is 17 GB; `$nnUNet_raw` is symlinked (6.6 MB) rather
  than duplicating the raw data.
- Preprocessed volumes are blosc2 `.b2nd` files (nnU-Net 2.6+), not `.npz`.
  Prediction probability maps *are* `.npz` — that is what the ensemble reads.
- `numpy` is pinned to 1.26.4 by `torchio`; torch/nnU-Net verified working on it.

## Status

**Verified working end-to-end on this machine** (memory-capped throughout):

| Stage | Status |
|---|---|
| Env: authors' nnU-Net installed editable, `-gamma` flag live | ✅ |
| Hard 11 GB memory cap (`scripts/capped.sh`) kills overruns | ✅ verified (exit 137) |
| Dataset conversion, 257 train + 91 val | ✅ |
| Region label scheme round-trips to native BraTS labels | ✅ verified on real GT |
| Preprocessing, full 257-case run (17 GB, exit 0, integrity verified) | ✅ |
| Training (`-gamma 0.7` confirmed active in the log) | ✅ smoke: 1 epoch |
| Prediction with `--save_probabilities` | ✅ |
| Evaluation (`nnUNetv2_evaluate_folder`, per-region Dice) | ✅ |
| Ensemble (region-aware; reproduces a member exactly when self-ensembled) | ✅ |
| `runner_ped2025.py` end to end | ✅ |
| Swin UNETR member | ⚠️ commands written, not trained here |
| HFF-Net member | ⚠️ blocked on MATLAB; see Open questions 1–2 |

**Not done:** full-length training. The 1000-epoch runs take days on one GPU and
were out of scope for this pass; the commands are in `run_commands.md` Steps 6–8.
Reported numbers so far come from a deliberately tiny smoke run (1 epoch,
8 cases) and are **not** meaningful scores.

## Next phase (not started yet)

This baseline keeps the original method intact. My own novel contribution and my
own README/contribution doc get added next, on top of this.
