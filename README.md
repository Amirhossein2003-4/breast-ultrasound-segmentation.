# breast-ultrasound-segmentation.
# Lightweight Breast Ultrasound Lesion Segmentation on BUS-UCLM

CPU-friendly transfer learning with patient-level evaluation, controlled ablations and an honest failure analysis.

> **Research project, not a clinical tool.** The model was developed on a public research dataset and has no external validation. It must not be used for diagnosis or any clinical decision.

---

## Overview

This project trains a lightweight lesion-segmentation model on the **BUS-UCLM** breast ultrasound dataset using an ImageNet-pretrained **MobileNetV3-Small** encoder and a small U-Net-style decoder. Everything runs on a laptop **CPU** (Intel Core i5-1334U, no discrete GPU).

The goal was not only to maximise Dice. The project is organised around three questions:

1. How far does a small pretrained CNN go on a small, patient-grouped medical dataset?
2. Which design changes actually help when they are tested one at a time against a fixed baseline?
3. Where and why does the model fail, and what can (and cannot) be concluded from the evidence?

Negative results are kept on purpose and reported as such.

## Key results

| Model | Validation pooled Dice | Test pooled Dice |
|---|---:|---:|
| **Baseline** (final model) | **0.6802** (epoch 27) | **0.6917** |

Main findings:

- **The simple baseline was the strongest configuration tested.** Dropout, freezing one more encoder stage and a generic augmentation policy did not improve on it.
- **Failures are dominated by complete misses and by false positives on normal images**, not by a lack of confidence at the decision threshold. Lowering the threshold barely changes Dice.
- **Validation errors cluster in a few patients**, and the validation and test sets contain only six patient groups each, so all subgroup observations are exploratory.

## Dataset

**BUS-UCLM** (breast ultrasound lesion segmentation, Mendeley Data, CC BY 4.0): 683 image/mask pairs from 38 patient groups.

| | Normal | Benign | Malignant | Total |
|---|---:|---:|---:|---:|
| Images | 419 | 174 | 90 | 683 |

- `INFO.csv` provides `Image`, `Resolution`, `Label`, `Doppler`, `Marks` and `Combined`.
- All images are reported as 856 × 606 pixels.
- Masks are colour-coded: black background, green benign lesion, red malignant lesion. This project uses a **binary** task (lesion vs background): any non-black pixel is lesion.
- Normal images have empty masks.

### Patient-level split

The filename prefix (e.g. `ALWI`, `ANAT`) identifies the patient group (38 groups). Images from one patient never appear in more than one split.

The split uses `GroupShuffleSplit` twice with seed 42 (70 % train, then the remaining 30 % split 50/50 into validation and test):

| Split | Images | Patient groups | Normal | Benign | Malignant |
|---|---:|---:|---:|---:|---:|
| Train | 431 | 26 | 267 | 94 | 70 |
| Validation | 144 | 6 | 88 | 45 | 11 |
| Test | 108 | 6 | 64 | 35 | 9 |

No patient-group overlap exists between splits (checked programmatically). The sizes `431 / 144 / 108` are asserted in the code so the split cannot drift silently.

> The split is group-based but **not stratified**. Validation has only 11 malignant images and test only 9, so metrics are sensitive to individual patients.

## Method

### Preprocessing

- Image and mask are resized to **384 × 256** (images bilinear, masks nearest-neighbour). This is a direct resize without padding, so the aspect ratio changes by about 6 % relative to the original 856 × 606.
- Grayscale ultrasound is stored as 3 identical channels to satisfy the pretrained network's input format.
- Inputs are normalised with ImageNet mean and standard deviation.
- No other preprocessing (CLAHE, denoising, cropping) is used in the final model; see the error analysis for why none was justified by the evidence.

### Architecture

- **Encoder:** MobileNetV3-Small with ImageNet-pretrained weights, split into stages `s2`, `s4`, `s8`, `s16`, `s32`.
- **Decoder:** four upsampling blocks with skip connections (`Up(576,48,128)`, `Up(128,24,64)`, `Up(64,16,32)`, `Up(32,16,24)`) and a 1×1 convolution head.
- **Frozen:** `s2`, `s4`, `s8`. Frozen stages are kept in evaluation mode so their batch-normalisation statistics do not change.
- **Trainable:** `s16`, `s32`, decoder and head.

### Training

| Setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rates | encoder 3e-4, decoder 1e-3 |
| Schedule | OneCycleLR, `pct_start = 0.2` |
| Epochs / batch size | 40 / 8 |
| Loss | cross-entropy + soft Dice (background excluded from the Dice term) |
| Augmentation | light: horizontal flip, brightness/contrast jitter, small random crop-and-rescale (train only) |
| Seed | 42 |
| Hardware | CPU only |

Checkpoint selection uses the best **validation** pooled Dice.

### Metrics

| Metric | Definition |
|---|---|
| `dice_pool` (primary) | Dice pooled over all foreground pixels of the evaluated set |
| `iou_pool` | pooled IoU |
| `dice` | mean per-image Dice over images that contain a lesion |
| `fp_normal` | fraction of normal images with more than 50 predicted lesion pixels |

Pooled Dice and mean per-image Dice aggregate differently (pooled Dice is dominated by large lesions) and are **not interchangeable**.

## Experiments

All controlled runs use the same split, seed and scheduler as the baseline, changing one factor at a time.

| Run | Change vs baseline | Best val Dice (epoch) | Val IoU | Test Dice | Test IoU | Outcome |
|---|---|---:|---:|---:|---:|---|
| **Baseline** | `s2/s4/s8` frozen, no added dropout | **0.6802** (27) | 0.5153 | **0.6917** | 0.5287 | Final model |
| Dropout 0.15 | decoder dropout 0.15 | 0.6605 (21) | 0.4931 | 0.6759 | 0.5104 | Rejected |
| Freeze `s16` | also freeze `s16` | 0.5793 (24) | 0.4078 | 0.5889 | 0.4174 | Rejected |
| Generic augmentation | stronger augmentation policy (below) | 0.6613 (26) | 0.4940 | not evaluated | – | Did not beat baseline |
| Earlier augmentation | augmentation **plus** new split (482/100/101) and `pct_start=0.3` | 0.6215 (30) | – | – | – | **Confounded**, not a valid comparison |
| Activation function | – | – | – | – | – | **Not run** |

**Dropout 0.15.** Lower on both validation (−0.0197) and test (−0.0158). This does not show that all regularisation is ineffective.

**Freeze `s16`.** Large drop (−0.1009 validation, −0.1028 test) while gradients were healthy, so the loss of performance is not explained by a gradient failure. `s16` should stay trainable in this setup.

**Generic augmentation (controlled).** Training-only policy: horizontal flip (p=0.5); affine transform (p=0.7) with rotation ±10°, zoom 0.9–1.2 and translation up to ±10 %; brightness/contrast jitter (p=0.8); gamma (p=0.3); noise (p=0.3). Best validation pooled Dice was 0.0189 below the baseline, and performance degraded late in training:

| | Best checkpoint (epoch 26) | Epoch 40 |
|---|---:|---:|
| Pooled Dice | 0.6613 | 0.6044 |
| Validation loss | 0.5048 | 0.5362 |
| Training loss | 0.1679 | 0.1454 |

Training loss kept falling while validation loss rose, consistent with late-stage overfitting. This conclusion applies to this particular policy only. Checkpoints were not saved and no test evaluation was run for this experiment.

**Earlier augmentation run.** It changed the split, the scheduler and the augmentation at the same time (best validation Dice 0.6215 at epoch 30, 0.4742 at epoch 40). It is kept in the log but supports no conclusion about augmentation.

**Activation functions.** Gradient and activation statistics were inspected and showed no dead-neuron or unhealthy-gradient pathology. A ReLU vs SiLU/GELU comparison was discussed but never completed, so no activation-function claim is made.

## Validation error analysis

The diagnostics below were computed on the **validation split only**; the test set was not used. Validation contains 56 lesion images (45 benign, 11 malignant) and 88 normal images from six patient groups, and images from the same patient are not independent, so everything here is descriptive.

### Image statistics vs per-image Dice

Spearman correlation with per-image Dice on lesion images:

| Feature | ρ |
|---|---:|
| Mean intensity | +0.22 |
| Intensity std | −0.04 |
| Dynamic range (p99 − p1) | +0.23 |
| Black-pixel fraction | −0.16 |
| Colour spread | +0.22 |
| Lesion area fraction | +0.14 |
| Signed contrast (lesion mean − surrounding-ring mean) | −0.20 |
| Border contact | undefined (constant in this set) |

All correlations are weak. Note that contrast is **signed**: more negative means a darker lesion relative to its surroundings. No single simple brightness, contrast, black-border or size feature explains the failures, so no generic preprocessing (CLAHE, denoising, intensity normalisation) was justified by this evidence.

### Subgroups (mean per-image Dice, lesion images)

| Group | Images | Dice | | Group | Images | Dice |
|---|---:|---:|---|---|---:|---:|
| Benign | 45 | 0.603 | | Doppler = No | 52 | 0.553 |
| Malignant | 11 | 0.420 | | Doppler = Yes | 4 | 0.757 |
| Marks = No | 23 | 0.388 | | Area: low / mid / high | 19 / 18 / 19 | 0.460 / 0.688 / 0.562 |
| Marks = Yes | 33 | 0.693 | | Contrast: low / mid / high | 19 / 18 / 19 | 0.620 / 0.613 / 0.472 |

False-positive fraction on normal images: 0.10 for `Marks = No` (n = 86) and 0.50 for `Marks = Yes` (n = 2, not reliable).

Cautions:

- Malignant lesions score lower, but there are only 11 such validation images.
- The **`Marks` gap is large but unexplained.** It could reflect an annotation-related cue, easier-to-see lesions in annotated images, patient or acquisition effects, or a mixture. No causal claim is made; this was not resolved.
- Lesion size alone does not explain performance (the relationship is non-monotonic).

### Complete misses

Nine validation lesion images received an **empty prediction** at the default threshold. Pixel counts are at the 384 × 256 model resolution; `Marks` is from `INFO.csv`.

| Image | Label | Marks | GT pixels | Predicted pixels |
|---|---|---|---:|---:|
| `DAPA_025` | Benign | No | 1,180 | 0 |
| `VITR_009` | Benign | No | 918 | 0 |
| `VITR_010` | Benign | No | 1,457 | 0 |
| `FLBA_017` | Benign | Yes | 1,137 | 0 |
| `DAPA_038` | Benign | No | 823 | 0 |
| `PAGY_004` | Malignant | No | 13,526 | 0 |
| `PAGY_011` | Malignant | No | 6,993 | 0 |
| `FLBA_004` | Malignant | No | 5,808 | 0 |
| `FLBA_005` | Malignant | No | 8,522 | 0 |

Eight of the nine come from only four patient groups (`DAPA`, `VITR`, `PAGY`, `FLBA`), which suggests patient-level variation rather than isolated image noise, although six validation patients are too few to be sure. Large malignant lesions are missed as well as small benign ones.

![Eight validation lesion images with the lowest Dice (rows: image, ground truth, prediction)](figures/worst8_val.png)

*The eight lowest-Dice validation lesion images (image / ground truth / prediction). All have empty predictions.*

Spurious predictions on normal images also occur, for example `CHSP_004` (2,129 predicted pixels), `DAPA_018` (2,391), `FLBA_001` (3,377), `ORPE_025` (12,309) and `ORPE_026` (3,214). Successful cases exist too, e.g. `DAPA_002` (Dice 0.912) and `VITR_003` (Dice 0.795). These examples show two failure patterns (misses and false positives) but do not identify a root cause.

![24 random training/validation images with label, Marks (M) and Doppler (D) flags](figures/random24_train_val.png)

*A random sample of training/validation images (`M:` = Marks, `D:` = Doppler). Appearance varies considerably between patients, including dark shadowing regions in normal images.*

### Threshold sweep

| Lesion-probability threshold | Mean lesion Dice | Empty predictions (lesion images) | Normal-image FP fraction |
|---:|---:|---:|---:|
| 0.5 | 0.567 | 9 / 56 | 0.11 |
| 0.4 | 0.569 | 9 / 56 | 0.12 |
| 0.3 | 0.570 | 8 / 56 | 0.15 |
| 0.2 | 0.571 | 8 / 56 | 0.16 |
| 0.1 | 0.573 | 5 / 56 | 0.17 |

The maximum predicted lesion probability inside the ground-truth mask has quantiles 0.06 / 0.91 / 1.00 / 1.00 / 1.00 (10th / 25th / 50th / 75th / 90th percentile). The model is therefore usually confident, and its failures are usually confident misses rather than under-confident detections. The per-image maximum does not prove correct localisation (a single confident pixel suffices), and lower thresholds trade a negligible Dice gain for more normal-image false positives, so **0.5 was kept**. The fixed protocol was not changed on the basis of this sweep.

## What the evidence supports, and what it does not

**Supported**

- The baseline is the strongest configuration among the completed, controlled comparisons.
- Dropout 0.15 and freezing `s16` hurt; the generic augmentation policy tested did not help on pooled validation Dice.
- Threshold tuning is not an effective fix.
- Errors include confident complete misses and false positives on normal images, and are concentrated in a few patients.

**Not supported (not claimed)**

- Any single root cause (brightness, lesion size, border artefacts, insufficient augmentation).
- That `Marks` causes better segmentation, or that it is a shortcut.
- That malignant lesions are intrinsically harder (n = 11 validation images).
- Any activation-function conclusion.
- Any statement about clinical performance or generalisation to other scanners, sites or populations.

## Limitations

- **Very few patients in evaluation:** six patient groups each in validation and test; 11 malignant validation images and 9 malignant test images. Estimates are noisy and sensitive to individual patients.
- **Single split, single seed.** No patient-level cross-validation and no confidence intervals.
- **Test-set exposure.** The test split was evaluated for the baseline, dropout and freeze-`s16` runs. Model choice was based on validation, but the test score is not a pristine unbiased estimate and should be read as indicative.
- **Metric mixing.** The primary metric is pooled Dice. Per-image diagnostics (e.g. 0.567 mean lesion Dice at threshold 0.5) and the per-image Dice logged for the generic-augmentation run (0.6303) come from different scripts and checkpoints; they are not presented as a controlled comparison. The statement that augmentation "did not beat the baseline" refers to the primary pooled metric.
- **Binary task only.** Benign and malignant lesions are merged; no lesion-type classification is attempted.
- **Distortion from resizing.** Images are resized without preserving aspect ratio; its effect was not tested.
- **No external validation.**

## Reproducibility

- **Environment:** Windows, Python (virtual environment), PyTorch on CPU with `torchvision`, `scikit-learn`, `pandas`, `numpy`, `Pillow`, `matplotlib`. Exact package versions were not recorded in the experiment log; pin them with `pip freeze` before publishing.
- **Hardware:** Intel Core i5-1334U, integrated Iris Xe graphics only; training ran on CPU.
- **Seeds:** training seed 42, split seed 42.
- **Split:** `GroupShuffleSplit` as described above; expected sizes `431 / 144 / 108` are asserted.
- **Pretrained weights:** torchvision-format ImageNet MobileNetV3-Small weights, loaded with all keys matching.
- **To run:** download BUS-UCLM, edit the `CONFIG` block at the top of `bus_seg.py` (`ROOT`, `WEIGHTS`, output directory), then run `python bus_seg.py`. The best validation checkpoint is saved as `best.pth`.

### Suggested repository layout

```
.
├── README.md
├── bus_seg.py              # data, model, training, evaluation
├── diagnose.py             # validation error analysis (no test data)
├── threshold_sweep.py      # lesion-probability threshold sweep (validation only)
├── figures/
│   ├── worst8_val.png
│   └── random24_train_val.png
└── requirements.txt
```

## Possible future work (not done)

These are recorded as ideas only; none was run and none is part of the results above.

- Patient-level cross-validation for more reliable estimates.
- Extra training data from other sources (for example BUSI), added to training only, with BUS-UCLM kept for validation and test. This would be a new experiment, not a continuation of the baseline.
- A clean activation-function comparison.
- A targeted investigation of the `Marks` gap.

## Data, license and citation

- **Dataset:** BUS-UCLM, Mendeley Data, CC BY 4.0. Please cite the dataset authors as given on the official dataset record: `<add official citation here>`.
- **Pretrained weights:** ImageNet-pretrained MobileNetV3-Small (torchvision format).
- **Code license:** `<choose a license>`.

<img width="2160" height="630" alt="worst8_val" src="https://github.com/user-attachments/assets/2bbd5a75-acd2-43b1-95d5-59be7fd1e980" />
<img width="1440" height="720" alt="random24_train_val" src="https://github.com/user-attachments/assets/914c9a29-fd7e-485b-ac4d-03280eaafe9d" />



## Author

Amirhossein Gadimi · [GitHub](https://github.com/Amirhossein2003-4) · [LinkedIn](https://www.linkedin.com/in/amirhossein-gadimi-57861a369/)
