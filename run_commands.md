# run_commands.md — BraTS-PED 2025 baseline, start to finish

Follow this file top to bottom. Every command has a one-line description above
it, and every step states what it needs, what it produces, and what it can run
alongside. No prior knowledge of this repo is assumed.

**Machine assumptions:** Linux or Windows + WSL2, NVIDIA GPU, conda installed.
All commands are plain bash + Python and are identical on native Linux and WSL2.

---

## Dependency / parallelization overview

| Step | What it does | Depends on | Can run in parallel with |
|---|---|---|---|
| 0 | Environment setup | — | everyone runs this individually |
| 1 | Configure `.env` | 0 | — |
| 2 | Convert raw data to nnU-Net layout | 1 | 3 |
| 3 | Stage HFF data + frequency decomposition | 1 | 2 |
| 4 | nnU-Net preprocessing (fingerprint + plans) | 2 | 3 |
| 5 | Custom 128³ plans for Swin UNETR | 4 | 6 |
| 6 | **Train member A** — nnU-Net (`-gamma`) | 4 | 7, 8 |
| 7 | **Train member B** — Swin UNETR | 5 | 6, 8 |
| 8 | **Train member C** — HFF-Net | 3 | 6, 7 |
| 9 | Evaluate a trained member | its own training step | other evaluations |
| 10 | Predict with each member | 6 / 7 / 8 | the three predictions are independent |
| 11 | Ensemble the members | 10 (all members you intend to ensemble) | — |

In prose:

- **Step 0 and 1** are per-machine setup; everyone does them.
- **Steps 2 and 3** are independent — one teammate can convert the nnU-Net
  dataset while another stages HFF's frequency decomposition.
- **Step 4** needs Step 2. **Step 5** needs Step 4.
- **Steps 6, 7 and 8 are the three ensemble members and have no dependency on
  each other.** Give them to three different teammates/machines to train at the
  same time. Step 6 needs Step 4, Step 7 needs Step 5, Step 8 needs Step 3.
- **Step 10** needs whichever member you are predicting with. The three member
  predictions are independent of each other.
- **Step 11 needs every member's predictions from Step 10** — pull them from the
  shared Drive if they were produced on someone else's machine. It cannot be
  parallelized further.

> ⚠️ **This machine has 16 GB of RAM.** Training and preprocessing had been
> crashing the desktop. Every heavy command below is wrapped in
> `bash scripts/capped.sh`, which runs it under a hard memory ceiling
> (`PEDS_MEM_MAX`, default 11G) so a runaway job is killed on its own instead of
> freezing the machine. The worker counts in `.env` are set low for the same
> reason. On a machine with more RAM, raise `nnUNet_n_proc_DA`,
> `nnUNet_def_n_proc` and `PEDS_MEM_MAX`.

---

## Step 0 — Environment setup

*Prerequisite:* None.
*Produces:* A conda environment named `peds_nnunet`.
*Can run in parallel with:* Nothing — everything needs it, but every teammate
runs it once on their own machine.

### 0.1 Create the environment from the pinned file

Create the environment with every dependency pinned to the versions this
baseline was verified on, including the CUDA-matched PyTorch build.

```bash
conda env create -f environment.yml
```

### 0.2 Activate it

**Do this in every new terminal** before running anything else in this file.

```bash
conda activate peds_nnunet
```

### 0.3 Confirm the GPU is visible

Check that PyTorch sees the card and the build covers its architecture.

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0)); print('arch list:', torch.cuda.get_arch_list())"
```

`environment.yml` pins the **cu128** build, which is required for Blackwell
(RTX 50-series, `sm_120`) and backward compatible with older supported cards.

> **WSL2 note:** the NVIDIA driver lives on the **Windows host**. Do *not*
> install a Linux NVIDIA driver inside WSL — only the CUDA toolkit/libraries the
> framework needs. `nvidia-smi` should already work inside WSL if the Windows
> driver is current.

### 0.4 Install the authors' modified nnU-Net

Install the in-repo nnU-Net **editable**. This copy carries the authors' `-gamma`
initialization-scale contribution; a stock `pip install nnunetv2` does **not**
have it and the training commands below would fail.

```bash
pip install -e src/nnUNet-official-new
```

### 0.5 Verify the toolchain

Check that the authors' `-gamma` flag is present and nnU-Net resolves to the
in-repo copy. Both lines must succeed.

```bash
nnUNetv2_train --help | grep -A2 -- "-gamma"
python -c "import nnunetv2, os; print('nnunetv2 from:', os.path.dirname(nnunetv2.__file__))"
```

The second command must print a path **inside this repo**
(`.../src/nnUNet-official-new/nnunetv2`). If it points into `site-packages`,
redo Step 0.4.

---

## Step 1 — Configure paths for your machine

*Prerequisite:* Step 0.
*Produces:* `.env` in the repo root (gitignored; never commit it).
*Can run in parallel with:* Nothing.

### 1.1 Create your `.env` from the template

Copy the checked-in template. Everything machine-specific lives here.

```bash
cp .env.example .env
```

### 1.2 Edit the paths

Open `.env` and set **`PEDS_ROOT` to this repo's absolute path** — that is the
only line most people need to change. `PEDS_WORK` defaults to
`${PEDS_ROOT}/peds_work`, so all generated data lands inside the repo (and is
gitignored, so it is never committed).

Budget roughly **60–100 GB** there for the full 257-case dataset. If the disk
holding the repo is tight, point `PEDS_WORK` at another disk instead — it is the
only setting that has to move.

```bash
${EDITOR:-nano} .env
```

> **WSL2 note:** keep `PEDS_WORK` and the dataset inside the native Linux
> filesystem (e.g. `/home/you/peds_work`), **not** under `/mnt/c/...`.
> Cross-filesystem I/O on WSL2 is several times slower and will bottleneck
> preprocessing and training.

### 1.3 Load the configuration into your shell

Source the helper. It exports everything in `.env`, including the three
variables the nnU-Net CLI requires. **Do this in every new terminal.**

```bash
source scripts/peds_env.sh
```

### 1.4 Confirm the configuration resolved correctly

Print the resolved configuration and sanity-check the paths.

```bash
python -m peds.config
```

### 1.5 Put the dataset in place

The raw data is **not** in git — pull it from the shared Drive. It must end up
laid out exactly like this, with `Training/` and `Validation/` folders of
per-case directories:

```
raw_dataset/BraTS-PEDs-v1/
├── Training/     257 cases: BraTS-PED-00001-000/BraTS-PED-00001-000-{t1n,t1c,t2w,t2f,seg}.nii.gz
└── Validation/    91 cases: same, minus the -seg file
```

Verify the counts before continuing (expect `257` then `91`).

```bash
ls raw_dataset/BraTS-PEDs-v1/Training | wc -l && ls raw_dataset/BraTS-PEDs-v1/Validation | wc -l
```

---

## Step 2 — Convert the raw data to the nnU-Net layout

*Prerequisite:* Step 1.
*Produces:* `$nnUNet_raw/Dataset501_BraTSPED/` — `imagesTr/`, `labelsTr/`,
`imagesTs/`, `dataset.json`. Images are **symlinked**, so this costs ~7 MB
rather than duplicating 33 GB.
⚠️ Do **not** upload this to Drive — it is symlinks into your local
`raw_dataset/` and is meaningless on another machine. Regenerate it instead; it
takes seconds.
*Can run in parallel with:* Step 3.

### 2.1 Convert both splits

Rename BraTS's hyphenated modalities into nnU-Net's numeric channel order
(`0000=t1n, 0001=t1c, 0002=t2w, 0003=t2f`) and write `dataset.json` with the
six-region label scheme.

```bash
python scripts/prepare_nnunet_dataset.py
```

For a quick smoke test on a handful of cases instead, use `--limit` (use at
least 6, since nnU-Net builds a 5-fold split):

```bash
python scripts/prepare_nnunet_dataset.py --limit 8
```

### 2.2 Confirm the conversion

Check the file counts: 1028 training images (257 cases × 4 modalities), 257
labels, 364 test images (91 × 4).

```bash
ls $nnUNet_raw/Dataset501_BraTSPED/imagesTr | wc -l
ls $nnUNet_raw/Dataset501_BraTSPED/labelsTr | wc -l
ls $nnUNet_raw/Dataset501_BraTSPED/imagesTs | wc -l
```

---

## Step 3 — Stage HFF data and run the frequency decomposition

*Prerequisite:* Step 1.
*Produces:* `$PEDS_HFF_DATA/Training/` — per-case folders with underscore
naming plus `_L` (low-frequency) and `_H1.._H4` (high-frequency) volumes, and
`train.txt` / `val.txt` split lists.
⚠️ **Upload `$PEDS_HFF_DATA` to the shared Drive at `brats-ped/hff_data/` once
this finishes successfully** — the NSCT step is slow and MATLAB-gated, so
teammates should not have to re-run it.
*Can run in parallel with:* Step 2.

> HFF's dataloader addresses files with **underscores** (`<case>_t1c.nii.gz`)
> while BraTS ships **hyphens** (`<case>-t1c.nii.gz`). Step 3.1 handles this;
> it is the most common cause of a silently empty HFF dataset.

### 3.1 Stage the cases and run the low-frequency (DTCWT) transform

Mirror each case under underscore naming and compute the shift-invariant
low-frequency volume for all four modalities. Pure Python; no MATLAB needed.

```bash
python scripts/prepare_hff_dataset.py --split Training
```

### 3.2 Run the high-frequency (NSCT) transform

Compute the four directional high-frequency sub-bands per modality.

> **Requires MATLAB with the Image Processing Toolbox.** Set `PEDS_MATLAB` in
> `.env` to the matlab executable if it is not on your `PATH`. This step is slow
> — budget hours for the full 257 cases.

```bash
bash scripts/run_nsct.sh Training
```

### 3.3 Verify every case has its frequency volumes

Report any case still missing its 16 high-frequency volumes. This must show
`257/257` before HFF training will work.

```bash
python scripts/prepare_hff_dataset.py --split Training --check
```

### 3.4 Stage the validation split too

Same treatment for the unlabelled validation cases, needed only at inference.

```bash
python scripts/prepare_hff_dataset.py --split Validation
bash scripts/run_nsct.sh Validation
```

---

## Step 4 — nnU-Net preprocessing

*Prerequisite:* Step 2.
*Produces:* `$nnUNet_preprocessed/Dataset501_BraTSPED/` — dataset fingerprint,
`nnUNetPlans.json`, and the preprocessed/normalized volumes (~18 GB for 257
cases).
⚠️ **Upload this to the shared Drive at `brats-ped/nnUNet_preprocessed/` once it
finishes successfully, so teammates don't have to re-run it.**
*Can run in parallel with:* Step 3.

### 4.1 Fingerprint, plan and preprocess

Analyse the dataset, derive the training plans, and write the preprocessed
volumes. `-np 2` keeps worker memory within this machine's 16 GB.

```bash
bash scripts/capped.sh nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -c 3d_fullres -np $PEDS_NP_PREPROCESS
```

### 4.2 Confirm the plans were written

The plans file must exist before any training starts.

```bash
ls -la $nnUNet_preprocessed/Dataset501_BraTSPED/nnUNetPlans.json
```

---

## Step 5 — Custom 128³ plans for Swin UNETR

*Prerequisite:* Step 4.
*Produces:* `$nnUNet_preprocessed/Dataset501_BraTSPED/nnUNetPlansPatch.json`
and its preprocessed data.
⚠️ **Upload to the shared Drive at `brats-ped/nnUNet_preprocessed/` with the
Step 4 output.**
*Can run in parallel with:* Step 6 (which uses the default plans).

> The `nnUNetTrainer_Swinunetr_1005epochs` variant builds `SwinUNETR` with
> `img_size=(128,128,128)`, so it needs a plans file whose patch size is 128³
> rather than the default 96×160×160.

### 5.1 Create the 128³ plans variant

Copy the default plans and override the 3d_fullres patch size.

```bash
python - <<'PY'
import json, os
from pathlib import Path
src = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_BraTSPED" / "nnUNetPlans.json"
dst = src.with_name("nnUNetPlansPatch.json")
plans = json.loads(src.read_text())
plans["plans_name"] = "nnUNetPlansPatch"
cfg = plans["configurations"]["3d_fullres"]
cfg["patch_size"] = [128, 128, 128]
cfg["data_identifier"] = "nnUNetPlansPatch_3d_fullres"
dst.write_text(json.dumps(plans, indent=4))
print("wrote", dst)
PY
```

### 5.2 Preprocess using the new plans

Generate the preprocessed data for that plans identifier.

```bash
bash scripts/capped.sh nnUNetv2_preprocess -d 501 -plans_name nnUNetPlansPatch -c 3d_fullres -np $PEDS_NP_PREPROCESS
```

---

## Step 6 — Train member A: nnU-Net with adjustable initialization

*Prerequisite:* Step 4.
*Produces:* `$nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_1000epochs__nnUNetPlans__3d_fullres__std_gamma0.7/fold_<N>/checkpoint_best.pth`
⚠️ **Upload the whole results folder to the shared Drive at
`brats-ped/nnUNet_results/` once training finishes, so teammates don't have to
re-train it.**
*Can run in parallel with:* Steps 7 and 8 (different machines — a single GPU
cannot hold two of these at once).

> `-gamma 0.7` is the authors' contribution: every conv/linear layer is
> re-initialized as `N(0, fan_in^(-0.7))`. The value is written into the plans
> file and reflected in the output folder name, and **the same value must be
> passed at prediction time**.

### 6.1 Smoke-test with a single epoch first

Confirm the whole training path works before committing to a long run. Takes a
couple of minutes.

```bash
bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_1epoch -gamma 0.7
```

Look for `The distribution of model weights have been changed with std rate = 0.7!!!`
in the output — that confirms the initialization contribution is active.

### 6.2 Train fold 0 for real

The full 1000-epoch run. This takes **days** on a single GPU.

```bash
bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_1000epochs -gamma 0.7
```

### 6.3 (Optional) Train the remaining folds

Folds are independent — hand them to different machines and run them at the same
time. Predicting with `fold=ensemble` later averages whichever folds exist.

```bash
bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 1 -tr nnUNetTrainer_1000epochs -gamma 0.7
bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 2 -tr nnUNetTrainer_1000epochs -gamma 0.7
bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 3 -tr nnUNetTrainer_1000epochs -gamma 0.7
bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 4 -tr nnUNetTrainer_1000epochs -gamma 0.7
```

### 6.4 Resume an interrupted run

Add `--c` to continue from the last checkpoint rather than starting over.

```bash
bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_1000epochs -gamma 0.7 --c
```

---

## Step 7 — Train member B: Swin UNETR with BraTS 2021 transfer learning

*Prerequisite:* Step 5.
*Produces:* `$nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_Swinunetr_1005epochs__nnUNetPlansPatch__3d_fullres/fold_<N>/checkpoint_best.pth`
⚠️ **Upload the results folder to the shared Drive at `brats-ped/nnUNet_results/`
once training finishes.**
*Can run in parallel with:* Steps 6 and 8 (different machines).

### 7.1 (Optional) Fetch the BraTS 2021 pre-trained checkpoint

The authors warm-start Swin UNETR from a BraTS 2021 model. Download the
SwinUNETR BRATS21 checkpoint from the MONAI model zoo
(<https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21>)
and point `.env` at it:

```bash
${EDITOR:-nano} .env    # set PEDS_SWIN_PRETRAINED=/absolute/path/to/fold0_f48_ep300_4gpu_dice0_8854_model.pt
```

Leave `PEDS_SWIN_PRETRAINED` empty to train from random initialization — the
trainer falls back to that automatically and prints a warning.

### 7.2 Smoke-test the Swin trainer

Verify the backbone swap and the 128³ plans line up before the long run.

```bash
bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_Swinunetr_500epochs -p nnUNetPlansPatch
```

### 7.3 Train the submission variant

The 1005-epoch variant used in the authors' submission (`lr 1e-3`, 128³ patches,
transfer-learning warm start). Takes **days**.

```bash
bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_Swinunetr_1005epochs -p nnUNetPlansPatch
```

---

## Step 8 — Train member C: HFF-Net

*Prerequisite:* Step 3 (including the MATLAB NSCT step — `--check` must report
every case complete).
*Produces:* `$PEDS_HFF_RESULTS/checkpoints/` — the trained HFF checkpoint.
⚠️ **Upload to the shared Drive at `brats-ped/hff_results/` once training
finishes.**
*Can run in parallel with:* Steps 6 and 7 (different machines).

### 8.1 Train HFF-Net

Train the dual-branch frequency network. `--selected_modal` must list the four
low-frequency channels followed by the sixteen high-frequency ones, using
BraTS-PED modality names (`t1n/t1c/t2w/t2f`) — **not** the BraTS-19/20 default
(`flair/t1/t1ce/t2`) baked into `train.py`.

```bash
cd src/HFF && bash -c '
source ../../scripts/peds_env.sh >/dev/null
python train.py \
  --train_list "$PEDS_HFF_DATA/Training/train.txt" \
  --val_list   "$PEDS_HFF_DATA/Training/val.txt" \
  --path_trained_models "$PEDS_HFF_RESULTS/checkpoints" \
  --dataset_name brats23men \
  --class_type all \
  --selected_modal t1n_L t1c_L t2w_L t2f_L \
      t1n_H1 t1n_H2 t1n_H3 t1n_H4 \
      t1c_H1 t1c_H2 t1c_H3 t1c_H4 \
      t2w_H1 t2w_H2 t2w_H3 t2w_H4 \
      t2f_H1 t2f_H2 t2f_H3 t2f_H4 \
  --num_epochs 450 -b 1
'; cd ../..
```

> **Known gap:** `train.py` derives its class count as `4 if class_type=='all'`,
> but BraTS-PED has **five** label values (background + ET/NET/CC/ED). Training
> HFF for PED needs that count raised to 5 before the checkpoint will be
> compatible with `scripts/hff_predict.py --num-classes 5`. See the HFF entry
> under "Open questions" in `CLAUDE.md`.

---

## Step 9 — Evaluate a trained member

*Prerequisite:* the training step for whichever member you are evaluating.
*Produces:* `summary.json` next to the predictions, with per-region Dice.
*Can run in parallel with:* other evaluations.

### 9.1 Predict on the labelled training images

Generate segmentations for cases that have ground truth to score against.

```bash
bash scripts/capped.sh nnUNetv2_predict \
  -i $nnUNet_raw/Dataset501_BraTSPED/imagesTr \
  -o $PEDS_PREDICTIONS/eval_tr \
  -d 501 -c 3d_fullres -tr nnUNetTrainer_1000epochs -f 0 \
  -gamma 0.7 -npp $PEDS_NP_PREDICT -nps $PEDS_NP_PREDICT
```

### 9.2 Score against the ground truth

Compute Dice per region. Point `-djfile`/`-pfile` at the trained model's folder.

```bash
MODEL=$nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_1000epochs__nnUNetPlans__3d_fullres__std_gamma0.7
nnUNetv2_evaluate_folder \
  $nnUNet_raw/Dataset501_BraTSPED/labelsTr \
  $PEDS_PREDICTIONS/eval_tr \
  -djfile $MODEL/dataset.json -pfile $MODEL/plans.json
```

### 9.3 Read the per-region scores

Print the Dice for each BraTS-PED region.

```bash
python -c "
import json; s = json.load(open('$PEDS_PREDICTIONS/eval_tr/summary.json'))
print('foreground mean Dice:', round(s['foreground_mean']['Dice'], 4))
for k, v in s['mean'].items(): print(f'  region {k}: Dice={v[\"Dice\"]:.4f}')"
```

Region keys map as: `(1,)`=ET, `(2,)`=NET, `(3,)`=CC, `(4,)`=ED,
`(1,2,3)`=tumour core, `(1,2,3,4)`=whole tumour.

---

## Step 10 — Predict with each member

*Prerequisite:* the corresponding training step.
*Produces:* `$PEDS_PREDICTIONS/<member>/` — one `<case>.nii.gz` plus a
`<case>.npz` probability map per case.
⚠️ Probability maps are large. **Upload to the shared Drive at
`brats-ped/predictions/<member>/` if a teammate needs to ensemble them on a
different machine.**
*Can run in parallel with:* the other members' predictions.

> `--save_probabilities` is required — the ensemble averages probability maps,
> not label maps. And `-gamma` must match the value used during training.

### 10.1 Predict with member A (nnU-Net)

```bash
bash scripts/capped.sh nnUNetv2_predict \
  -i $nnUNet_raw/Dataset501_BraTSPED/imagesTs \
  -o $PEDS_PREDICTIONS/nnunet \
  -d 501 -c 3d_fullres -tr nnUNetTrainer_1000epochs \
  -gamma 0.7 --save_probabilities -npp $PEDS_NP_PREDICT -nps $PEDS_NP_PREDICT
```

### 10.2 Predict with member B (Swin UNETR)

```bash
bash scripts/capped.sh nnUNetv2_predict \
  -i $nnUNet_raw/Dataset501_BraTSPED/imagesTs \
  -o $PEDS_PREDICTIONS/swin \
  -d 501 -c 3d_fullres -tr nnUNetTrainer_Swinunetr_1005epochs -p nnUNetPlansPatch \
  --save_probabilities -npp $PEDS_NP_PREDICT -nps $PEDS_NP_PREDICT
```

### 10.3 Predict with member C (HFF-Net)

HFF is not an nnU-Net trainer, so it uses this project's own whole-volume
inference script, which writes the same `.npz` + `.nii.gz` layout.

```bash
bash scripts/capped.sh python scripts/hff_predict.py \
  --input $PEDS_HFF_DATA/Validation \
  --output $PEDS_PREDICTIONS/hff \
  --checkpoint $PEDS_HFF_RESULTS/checkpoints/hff_best.pth \
  --num-classes 5 --save-probabilities
```

---

## Step 11 — Ensemble the members

*Prerequisite:* Step 10 for every member you want to include. If a member was
predicted on another machine, pull its folder from the Drive first.
*Produces:* `$PEDS_PREDICTIONS/ensemble/` — the final segmentations, as native
BraTS-PED labels (`1`=ET, `2`=NET, `3`=CC, `4`=ED). These are the submission
files.
⚠️ **Upload to the shared Drive at `brats-ped/predictions/ensemble/`.**
*Can run in parallel with:* Nothing — this is the last step.

### 11.1 Average the probability maps

Average the members' probabilities and convert to a label map. Only cases
present in *every* member directory are ensembled.

```bash
MODEL=$nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_1000epochs__nnUNetPlans__3d_fullres__std_gamma0.7
bash scripts/capped.sh python ensemble/ensemble.py \
  --members $PEDS_PREDICTIONS/nnunet $PEDS_PREDICTIONS/swin $PEDS_PREDICTIONS/hff \
  --model-dir $MODEL \
  -o $PEDS_PREDICTIONS/ensemble
```

To weight members unequally, add `--weights 1.0 1.0 0.5` in the same order as
`--members`.

### 11.2 Sanity-check the output

Confirm the label values are a subset of `{0,1,2,3,4}`.

```bash
python -c "
import nibabel as nib, numpy as np, glob, os
labs = set()
for f in glob.glob(os.path.join('$PEDS_PREDICTIONS/ensemble', '*.nii.gz')):
    labs |= set(np.unique(np.asarray(nib.load(f).dataobj)).astype(int).tolist())
print('labels:', sorted(labs)); print('valid:', labs <= {0,1,2,3,4})"
```

---

## One-shot alternative: the full inference pipeline

Once the members are trained, `runner_ped2025.py` performs Steps 10 and 11 in a
single command: it prepares the inputs, runs each member, and ensembles.

*Prerequisite:* Steps 6, 7 and 8.
*Produces:* the same output as Step 11.

Preview the commands it would run without executing anything:

```bash
python runner_ped2025.py --input raw_dataset/BraTS-PEDs-v1/Validation --dry-run
```

Run it for real:

```bash
bash scripts/capped.sh python runner_ped2025.py --input raw_dataset/BraTS-PEDs-v1/Validation
```

Run a single member (useful before all three are trained), pinning the trainer,
checkpoint and fold:

```bash
bash scripts/capped.sh python runner_ped2025.py \
  --input raw_dataset/BraTS-PEDs-v1/Validation \
  --members nnunet --trainer nnUNetTrainer_1000epochs --checkpoint best --fold 0
```

Add `--skull-strip` only for data that is not already skull-stripped. BraTS-PED
ships skull-stripped, so it is off by default; enabling it also requires the
nnU-Net v1 `Task070_autosegm` model, which is **not** included in this repo.

---

## Troubleshooting

**The machine freezes or a job is killed with exit code 137.**
137 means the memory cap did its job. Lower `PEDS_MEM_MAX` is not the fix —
lower the worker counts in `.env` (`nnUNet_n_proc_DA`, `nnUNet_def_n_proc`) or
raise `PEDS_MEM_MAX` if the machine actually has spare RAM.

**`nnUNetv2_train: error: unrecognized arguments: -gamma`.**
Stock nnU-Net is installed instead of the in-repo copy. Redo Step 0.4 and
re-check with Step 0.5.

**`nnUNet_raw is not defined`, or paths point somewhere unexpected.**
You did not `source scripts/peds_env.sh` in this terminal, or a stale
`export nnUNet_raw=...` is in your shell profile. `.env` wins over stale
exports; re-source and confirm with `python -m peds.config`.

**HFF loads zero volumes.**
Underscore vs hyphen naming. Re-run Step 3.1 and check with Step 3.3.

**`FileNotFoundError: ... checkpoint_best.pth` naming a fold you never trained.**
Prediction defaults to averaging every fold. Pass `-f 0` (or
`--fold 0` to `runner_ped2025.py`) when only one fold exists.
