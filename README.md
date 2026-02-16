# Calibrated Semi-Supervised Models for Disaster Response based on Training Dynamics

This repository contains the code and experiments for the paper:

> **Calibrated Semi-Supervised Models for Disaster Response based on Training Dynamics**
> Khushboo Gupta, Nikita Gautam, Tiberiu Sosea, Doina Caragea, and Cornelia Caragea
> *Proceedings of the 22nd International ISCRAM Conference, Halifax, Canada, May 2025*
> DOI: [10.59297/5xkjq067](https://doi.org/10.59297/5xkjq067)

The project implements and evaluates semi-supervised learning (SSL) methods — including **Self-Training (ST)**, **Uncertainty-aware Self-Training (UST)**, and **AUM-ST** — for few-shot text classification of disaster-related tweets into humanitarian categories. A key focus is on **model calibration**: ensuring that predicted probabilities reflect actual correctness, which is critical for trustworthy decision-making in disaster response.

## Table of Contents

- [Motivation](#motivation)
- [Algorithm](#algorithm)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiment Notebook](#experiment-notebook)
- [Key Hyperparameters](#key-hyperparameters)
- [Results Format](#results-format)
- [References](#references)

---

## Motivation

During natural disasters, social media platforms like Twitter become critical channels for real-time situational awareness. Tweets must be classified into humanitarian categories (e.g., infrastructure damage, injured people, rescue efforts) to help responders prioritize actions.

The challenge is that **labeled data is extremely scarce** in disaster scenarios — new events happen quickly and manual annotation is slow. Semi-supervised learning addresses this by leveraging a small set of labeled examples together with a large pool of unlabeled tweets. However, many SSL approaches suffer from **miscalibration** — they may achieve competitive F1 scores but produce poorly calibrated confidence estimates, limiting their reliability for real-world deployment.

This work comprehensively evaluates both **classification performance (macro-F1)** and **calibration (Expected Calibration Error / ECE)** across multiple SSL methods, and proposes approaches that improve calibration while maintaining strong F1 scores. This is the first such comprehensive evaluation in the disaster response domain.

## Algorithm

### Overview

The SSL pipeline has three phases:

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────┐
│  Phase 1:       │     │  Phase 2:            │     │  Phase 3:      │
│  Base Model     │────▶│  Self-Training Loop  │────▶│  Evaluation    │
│  Selection      │     │  (Pseudo-labeling)   │     │  (F1 + ECE)    │
└─────────────────┘     └──────────────────────┘     └────────────────┘
```

### Phase 1: Base Model Selection

To reduce sensitivity to random initialization (which matters a lot with few-shot data), the system:

1. Fine-tunes `N_base` (default: 3) independent copies of BERTweet on the small labeled set
2. Each copy trains for up to `sup_epochs` (default: 18) epochs with early stopping (patience = 3) on validation macro-F1
3. The best-performing model (by validation F1) is selected as the starting checkpoint

### Phase 2: Self-Training Loop

For each of `unsup_epochs` (default: 12) iterations:

1. **Sample unlabeled data:** Draw `sample_size` (default: 1,800) instances from the unlabeled pool
2. **Generate pseudo-labels:** The model predicts labels for the sampled unlabeled data. The method for selecting which pseudo-labels to keep depends on the `sample_scheme`:
   - **`uniform`** — Standard self-training (ST). Randomly selects `unsup_size` (default: 1,000) pseudo-labeled instances without any uncertainty filtering. This is the simplest baseline.
   - **`easy_bald_class_conf`** — Uncertainty-aware self-training (UST). Runs `T` stochastic forward passes with MC Dropout, computes BALD acquisition scores, and preferentially selects "easy" (low-uncertainty) examples per class with confidence-weighted loss.
   - Other schemes: `easy_bald`, `bald_difficulty`, `easy_bald_class` (see [Sampling Schemes](#sampling-schemes) below)
3. **Re-train:** Combine the original labeled data with the selected pseudo-labeled data and re-train the model with a mixed loss:
   - `loss = 0.5 × supervised_loss + 0.5 × unsupervised_loss`
   - The unsupervised loss is optionally weighted by model confidence (controlled by `alpha`)
4. **Update checkpoint:** If validation F1 improves, save the new model as the best checkpoint

### Phase 3: Evaluation

The best checkpoint is evaluated on the held-out test set, reporting:
- **Macro-F1:** Primary classification metric, averaged across all classes
- **ECE (Expected Calibration Error):** Measures how well the model's predicted probabilities match actual correctness, using `n_bins=10`. Lower ECE means better-calibrated predictions — essential for trustworthy deployment in disaster response.

### Sampling Schemes

The `sample_scheme` parameter controls how pseudo-labeled instances are selected from the unlabeled pool:

| Scheme | Description | Uncertainty? | Per-class? | Confidence weighting? |
|--------|-------------|:---:|:---:|:---:|
| `uniform` | Random selection (standard ST baseline) | No | No | No |
| `easy_bald` | Prefer low-BALD (easy) examples | Yes | No | No |
| `easy_bald_class_conf` | Per-class easy BALD with confidence loss | Yes | Yes | Yes |
| `bald_difficulty` | Prefer high-BALD (difficult) examples | Yes | No | No |

### BALD (Bayesian Active Learning by Disagreement)

BALD measures the mutual information between the model's predictions and its parameters. It is estimated using MC Dropout — running `T` stochastic forward passes with dropout enabled at inference time. High BALD means the model is uncertain because different dropout masks give different predictions. Low BALD means the model is confident regardless of dropout.

```
BALD = H[y | x] - E_θ[H[y | x, θ]]
     = (entropy of mean prediction) - (mean entropy of individual predictions)
```

### Confidence-Weighted Loss

When `conf` is part of the sampling scheme name (e.g., `easy_bald_class_conf`), pseudo-labeled examples are weighted by their prediction confidence during training. The weight is computed as `-log(confidence) × alpha`, where `alpha` controls the strength of the reweighting. This allows the model to down-weight uncertain pseudo-labels in the loss function.

## Dataset

### Source

This project uses a subset of the [**HumAID** (Human-Annotated Disaster Incidents Data)](https://crisisnlp.qcri.org/humaid_dataset) dataset (Alam et al., ICWSM 2021). HumAID contains ~77K human-labeled tweets sampled from ~24 million tweets across 19 major natural disaster events (earthquakes, hurricanes, wildfires, floods) that occurred between 2016 and 2019.

We use **10 of the 19 events** from HumAID:

| Disaster Event | Year | Type | Classes |
|---|:---:|---|:---:|
| California Wildfires | 2018 | Wildfire | 10 |
| Cyclone Idai | 2019 | Cyclone | 10 |
| Canada Wildfires | 2016 | Wildfire | 8 |
| Hurricane Dorian | 2019 | Hurricane | 9 |
| Hurricane Florence | 2018 | Hurricane | 9 |
| Hurricane Harvey | 2017 | Hurricane | 9 |
| Hurricane Irma | 2017 | Hurricane | 9 |
| Hurricane Maria | 2017 | Hurricane | 9 |
| Kaikoura Earthquake | 2016 | Earthquake | 9 |
| Kerala Floods | 2018 | Flood | 9 |

### Humanitarian Categories

Each tweet is classified into one of up to 10 humanitarian categories. **Not every disaster has all 10 classes** — some events have only 7, 8, or 9 classes depending on the types of tweets observed during that event:

| ID | Category | Description |
|---|---|---|
| 0 | `caution_and_advice` | Warnings, safety tips, preparedness advice |
| 1 | `displaced_people_and_evacuations` | Reports of evacuations, shelters, displacement |
| 2 | `infrastructure_and_utility_damage` | Damage to buildings, roads, power, water |
| 3 | `injured_or_dead_people` | Reports of casualties or injuries |
| 4 | `missing_or_found_people` | Missing persons reports or reunifications |
| 5 | `not_humanitarian` | Tweets not related to humanitarian response |
| 6 | `other_relevant_information` | Relevant but not fitting other categories |
| 7 | `requests_or_urgent_needs` | Requests for help, supplies, resources |
| 8 | `rescue_volunteering_or_donation_effort` | Rescue operations, volunteer coordination, donations |
| 9 | `sympathy_and_support` | Expressions of solidarity, prayers, emotional support |

For example, `canada_wildfires_2016` has only 8 classes (missing `injured_or_dead_people` and `missing_or_found_people`), while most other events have 9 classes (missing `missing_or_found_people`). The code dynamically detects the actual classes per disaster at runtime, ensuring the model only predicts classes that are present.

### Few-Shot Data Splits

To simulate the realistic scenario of limited labeled data during an emerging disaster, the full training pool for each event is partitioned into small **labeled** subsets and large **unlabeled** pools:

```
data/{disaster}/
├── {disaster}_train.tsv           # Full training pool (~5,000 tweets)
├── {disaster}_dev.tsv             # Validation set (fixed across all splits)
├── {disaster}_test.tsv            # Test set (fixed across all splits)
├── labeled_{k}_set{s}.tsv         # Few-shot labeled subset
└── unlabeled_{k}_set{s}.tsv       # Corresponding unlabeled pool (remainder)
```

**Few-shot labeled splits** are created by sampling `k` examples per class from the full training pool:
- **Sizes (`k`):** 5, 10, 25, 50 examples per class
- **Sets (`s`):** 1, 2, 3 (three independent random samples for each size)

This gives **12 labeled splits per disaster** (4 sizes × 3 sets = 12). The three sets at each size allow measuring variance due to the specific labeled examples chosen, which is important in few-shot settings where performance can vary significantly depending on which examples are selected.

For example, `labeled_5_set1.tsv` contains 5 examples per class (50 total for a 10-class disaster), and `unlabeled_5_set1.tsv` contains the remaining ~5,100 training tweets not included in that labeled split. The dev and test sets are **shared across all splits** for a given disaster — only the labeled/unlabeled partition changes.

**TSV format** (tab-separated, with header):

```
tweet_id	tweet_text	class_label
1061291825879691264	Be in Ventura County later today...	caution_and_advice
```

**Example sizes** (california_wildfires_2018):

| Split | Size |
|---|---|
| `labeled_5_set1.tsv` | 50 tweets (5 per class × 10 classes) |
| `labeled_50_set1.tsv` | 500 tweets (50 per class × 10 classes) |
| `unlabeled_5_set1.tsv` | ~5,100 tweets |
| `{disaster}_dev.tsv` | ~750 tweets |
| `{disaster}_test.tsv` | ~1,460 tweets |

## Project Structure

```
├── run_ust.py              # CLI entry point — parses arguments, loads data, calls train_model()
├── ust.py                  # Core training logic — BertModel, train_model(), mc_dropout_evaluate()
├── sampler.py              # Uncertainty-based sampling strategies for pseudo-label selection
├── custom_dataset.py       # PyTorch Dataset classes for tokenized tweet data
├── notebooks/
│   └── st-experiments.ipynb  # Jupyter notebook for running batch ST experiments
├── data/
│   ├── california_wildfires_2018/
│   ├── canada_wildfires_2016/
│   ├── cyclone_idai_2019/
│   ├── hurricane_dorian_2019/
│   ├── hurricane_florence_2018/
│   ├── hurricane_harvey_2017/
│   ├── hurricane_irma_2017/
│   ├── hurricane_maria_2017/
│   ├── kaikoura_earthquake_2016/
│   └── kerala_floods_2018/
├── results/                # Experiment output (JSON files with F1, ECE metrics)
├── requirements.txt
├── .env                    # Environment variables (PYTHONHASHSEED, CUDA_VISIBLE_DEVICES)
└── .gitignore
```

### Module Details

**`run_ust.py`** — Command-line interface. Parses all hyperparameters as arguments, loads the TSV data files into `CustomDataset_tracked` objects, and calls `train_model()`. Contains the default 10-class `label_to_id` mapping. Use this for individual runs from the terminal.

**`ust.py`** — Core training engine.
- `BertModel`: Wraps HuggingFace's `AutoModelForSequenceClassification` with an optional learned temperature scaling parameter `T` for post-hoc calibration.
- `train_model()`: Orchestrates the full pipeline — base model selection, self-training loop, and test evaluation. Saves model checkpoints as `data/{disaster}/pytorch_model.bin` and results as JSON to `results/{disaster}/`.
- `mc_dropout_evaluate()`: Runs `T` stochastic forward passes with dropout enabled at inference time, collecting prediction distributions for BALD-based uncertainty estimation.
- `evaluate()`: Computes macro-F1 and ECE on a given dataset split.

**`sampler.py`** — Implements the uncertainty-based sampling strategies:
- `get_BALD_acquisition()`: Computes BALD scores from the `T` stochastic forward passes.
- `sample_by_bald_class_easiness()`: The primary UST sampling function — selects pseudo-labels per class, preferring low-uncertainty (easy) examples, weighted by BALD-based easiness scores.
- Other variants: `sample_by_bald_easiness()`, `sample_by_bald_difficulty()`, `sample_by_bald_class_difficulty()`.

**`custom_dataset.py`** — PyTorch Dataset subclasses:
- `CustomDataset`: Basic text/label pairs with tokenization. Supports `get_subset_dataset()` for sampling subsets.
- `CustomDataset_tracked`: Extends `CustomDataset` with tweet ID tracking (`idxes`), used throughout the pseudo-labeling pipeline to maintain provenance of unlabeled instances.

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended; CPU works but is very slow)
- PyTorch with CUDA support

```bash
pip install -r requirements.txt
```

**Note:** The default `torch` package from PyPI is CPU-only. For GPU support, install PyTorch with the appropriate CUDA version:

```bash
# Example for CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check your CUDA driver version with `nvidia-smi` and install the matching PyTorch build. CUDA drivers are backward-compatible (e.g., driver CUDA 12.6 supports PyTorch built for CUDA 12.4).

### Environment Variables

Create a `.env` file (or set these in your shell):

```bash
PYTHONHASHSEED=42              # Global seed for reproducibility
CUDA_VISIBLE_DEVICES=0         # GPU device index (0, 1, etc.)
```

`PYTHONHASHSEED` is required — the code reads it at startup as the global random seed.

## Usage

### Command Line (Single Experiment)

```bash
PYTHONHASHSEED=42 python run_ust.py \
  --disaster california_wildfires_2018 \
  --train_file 5_set1 \
  --sample_scheme uniform \
  --sup_epochs 18 \
  --unsup_epochs 12 \
  --T 7 \
  --alpha 0.1 \
  --N_base 3
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--disaster` | (required) | Disaster directory name under `data/` |
| `--train_file` | `S1T_5` | Labeled split name (e.g., `5_set1`, `10_set2`, `50_set3`) |
| `--sample_scheme` | `easy_bald_class_conf` | Sampling strategy (`uniform` for standard ST) |
| `--sup_epochs` | 18 | Max supervised epochs per training phase |
| `--unsup_epochs` | 12 | Number of self-training iterations |
| `--N_base` | 3 | Number of random initializations for base model |
| `--T` | 7 | MC Dropout forward passes (unused for `uniform`) |
| `--alpha` | 0.1 | Confidence loss weighting factor |
| `--sample_size` | 1800 | Unlabeled instances sampled per iteration |
| `--unsup_size` | 1000 | Pseudo-labeled instances kept per iteration |
| `--pt_teacher_checkpoint` | `vinai/bertweet-base` | Pre-trained model checkpoint |

### Batch Experiments (Notebook)

For running experiments across all disasters and splits, use the Jupyter notebook:

```bash
cd notebooks
jupyter notebook st-experiments.ipynb
```

See [Experiment Notebook](#experiment-notebook) below for details.

## Experiment Notebook

The notebook `notebooks/st-experiments.ipynb` provides a structured workflow for running Self-Training experiments at scale across all 10 disasters and all 12 labeled splits (120 total experiments).

### Cells

1. **Setup** — Imports, seeds, GPU detection, logging suppression
2. **Helpers** — `detect_classes()` for dynamic class detection per disaster, `get_dataset()` and `load_disaster_datasets()` for data loading
3. **Configuration** — Model checkpoint, hyperparameters, and all 12 train file combinations (`LABEL_SIZES × SETS`)
4. **Part 1: Sanity Check** — Runs a single experiment (`california_wildfires_2018`, `5_set1`) to verify the pipeline works end-to-end
5. **Part 2: Pre-scan** — Scans all 120 experiments (10 disasters × 12 splits), shows which are done vs. pending
6. **Part 2: Full Experiment** — Runs all pending experiments with progress tracking and resume support
7. **Summary** — Pivot tables showing macro-F1 and ECE by disaster × label size, averaged over the 3 sets

### Resume Support

Results are saved to disk after each experiment completes (`results/{disaster}/st_uniform_{size}_set{s}.txt`). If the notebook crashes or is interrupted, simply re-run the experiment cell — it will skip any experiments that already have results files and continue from where it left off.

### Progress Tracking

The experiment loop reports:
- Per-experiment: disaster name, split, class count, F1 result, and elapsed time
- Periodic summaries: percentage complete (by entries processed), elapsed time, and ETA

## Key Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `N_base` | 3 | More initializations → better base model, but slower startup |
| `sup_epochs` | 18 | Max epochs per training phase (early stopping at patience=3) |
| `unsup_epochs` | 12 | More ST iterations → more pseudo-label refinement |
| `sample_size` | 1800 | Larger pool → more candidates for pseudo-labeling |
| `unsup_size` | 1000 | More pseudo-labels per iteration → faster but noisier expansion |
| `alpha` | 0.1 | Higher → pseudo-labels weighted more by confidence |
| `T` | 7 | More MC passes → better uncertainty estimates (only for BALD schemes) |
| `sample_scheme` | varies | `uniform` = baseline ST; `easy_bald_class_conf` = full UST |

## Results Format

Each experiment produces a JSON file at `results/{disaster}/st_uniform_{size}_set{s}.txt`:

```json
{
    "Temperature Scaling": false,
    "Label Smoothing": 0.0,
    "Best ST model": {
        "F1 before temp scaling": "0.383",
        "ECE before temp scaling": "tensor(0.4798, device='cuda:0')",
        "T before temp scaling": "1.0"
    }
}
```

- **F1 before temp scaling**: Macro-F1 on the test set
- **ECE before temp scaling**: Expected Calibration Error (lower is better)
- **T before temp scaling**: Temperature parameter value (1.0 = no scaling)

When temperature scaling is enabled (`temp_scaling=True`), additional post-hoc calibrated metrics (`F1 after temp scaling`, `ECE after temp scaling`) are also recorded.

Model checkpoints are saved as `data/{disaster}/pytorch_model.bin` (overwritten each run).

## References

- Gupta, K., Gautam, N., Sosea, T., Caragea, D., & Caragea, C. (2025). [Calibrated Semi-Supervised Models for Disaster Response based on Training Dynamics](https://doi.org/10.59297/5xkjq067). *Proceedings of the 22nd International ISCRAM Conference*, Halifax, Canada.
- Sosea, T., & Caragea, C. (2022). [Leveraging Training Dynamics and Self-Training for Text Classification](https://aclanthology.org/2022.findings-emnlp.350/). *Findings of the Association for Computational Linguistics: EMNLP 2022*.
- Li, H., Caragea, D., & Caragea, C. (2021). [Combining Self-training with Deep Learning for Disaster Tweet Classification](https://idl.iscram.org/files/hongminli/2021/2367_HongminLi_etal2021.pdf). *Proceedings of the 18th International ISCRAM Conference*.
- Mukherjee, S., & Awadallah, A. (2020). [Uncertainty-aware Self-training for Few-shot Text Classification](https://arxiv.org/abs/2006.15315). *NeurIPS 2020*.
- Alam, F., Qazi, U., Imran, M., & Ofli, F. (2021). [HumAID: Human-Annotated Disaster Incidents Data from Twitter with Deep Learning Benchmarks](https://ojs.aaai.org/index.php/ICWSM/article/view/18116). *ICWSM 2021*.
- Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). [BERTweet: A Pre-trained Language Model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2/). *EMNLP 2020*.
