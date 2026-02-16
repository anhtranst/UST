# Uncertainty-aware Self-Training (UST) for Few-Shot Disaster Tweet Classification

This project implements **Uncertainty-aware Self-Training (UST)** for few-shot text classification of disaster-related tweets. Given only a handful of labeled examples per class (as few as 5), the system leverages a large pool of unlabeled tweets through iterative pseudo-labeling to improve classification performance.

The approach is based on [Mukherjee & Awadallah (2020)](https://arxiv.org/abs/2006.15315) and uses **BERTweet** (`vinai/bertweet-base`) as the backbone encoder, with **MC Dropout** for uncertainty estimation during pseudo-label selection.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Algorithm](#algorithm)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Experiment Notebook](#experiment-notebook)
- [Key Hyperparameters](#key-hyperparameters)
- [Results Format](#results-format)

---

## Problem Statement

During natural disasters, social media platforms like Twitter become critical channels for real-time situational awareness. Tweets must be classified into humanitarian categories (e.g., infrastructure damage, injured people, rescue efforts) to help responders prioritize actions.

The challenge is that **labeled data is extremely scarce** in disaster scenarios — new events happen quickly and manual annotation is slow. This project addresses the problem by:

1. Starting with very few labeled examples (5, 10, 25, or 50 per class)
2. Using a large pool of unlabeled tweets from the same event
3. Applying self-training to iteratively expand the training set with high-confidence pseudo-labels

## Algorithm

### Overview

The UST pipeline has three phases:

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────┐
│  Phase 1:       │     │  Phase 2:            │     │  Phase 3:      │
│  Base Model     │────▶│  Self-Training Loop  │────▶│  Evaluation    │
│  Selection      │     │  (Pseudo-labeling)   │     │                │
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
   - **`uniform`** — Standard self-training. Randomly selects `unsup_size` (default: 1,000) pseudo-labeled instances without any uncertainty filtering. This is the simplest baseline.
   - **`easy_bald_class_conf`** — Uncertainty-aware selection. Runs `T` stochastic forward passes with MC Dropout, computes BALD acquisition scores, and preferentially selects "easy" (low-uncertainty) examples per class with confidence-weighted loss.
   - Other schemes: `easy_bald`, `bald_difficulty`, `easy_bald_class` (see [Sampling Schemes](#sampling-schemes) below)
3. **Re-train:** Combine the original labeled data with the selected pseudo-labeled data and re-train the model with a mixed loss:
   - `loss = 0.5 × supervised_loss + 0.5 × unsupervised_loss`
   - The unsupervised loss is optionally weighted by model confidence (controlled by `alpha`)
4. **Update checkpoint:** If validation F1 improves, save the new model as the best checkpoint

### Phase 3: Evaluation

The best checkpoint is evaluated on the held-out test set, reporting:
- **Macro-F1:** Primary metric, averaged across all classes
- **ECE (Expected Calibration Error):** Measures how well the model's predicted probabilities match actual correctness, using `n_bins=10`

### Sampling Schemes

The `sample_scheme` parameter controls how pseudo-labeled instances are selected from the unlabeled pool:

| Scheme | Description | Uncertainty? | Per-class? | Confidence weighting? |
|--------|-------------|:---:|:---:|:---:|
| `uniform` | Random selection (standard ST baseline) | No | No | No |
| `easy_bald` | Prefer low-BALD (easy) examples | Yes | No | No |
| `easy_bald_class_conf` | Per-class easy BALD with confidence loss | Yes | Yes | Yes |
| `bald_difficulty` | Prefer high-BALD (difficult) examples | Yes | No | No |

**BALD (Bayesian Active Learning by Disagreement):** Measures the mutual information between the model's predictions and its parameters. High BALD = the model is uncertain because different dropout masks give different predictions. Low BALD = the model is confident regardless of dropout.

```
BALD = H[y | x] - E_θ[H[y | x, θ]]
     = (entropy of mean prediction) - (mean entropy of individual predictions)
```

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
- `BertModel`: Wraps HuggingFace's `AutoModelForSequenceClassification` with an optional learned temperature scaling parameter.
- `train_model()`: Orchestrates the full pipeline — base model selection, self-training loop, and test evaluation. Saves model checkpoints as `data/{disaster}/pytorch_model.bin` and results as JSON to `results/{disaster}/`.
- `mc_dropout_evaluate()`: Runs `T` stochastic forward passes with dropout enabled at inference time, collecting prediction distributions for uncertainty estimation.
- `evaluate()`: Computes macro-F1 and ECE on a given dataset split.

**`sampler.py`** — Implements the uncertainty-based sampling strategies:
- `get_BALD_acquisition()`: Computes BALD scores from the `T` stochastic forward passes.
- `sample_by_bald_class_easiness()`: The primary UST sampling function — selects pseudo-labels per class, preferring low-uncertainty (easy) examples, weighted by BALD-based easiness scores.
- Other variants: `sample_by_bald_easiness()`, `sample_by_bald_difficulty()`, `sample_by_bald_class_difficulty()`.

**`custom_dataset.py`** — PyTorch Dataset subclasses:
- `CustomDataset`: Basic text/label pairs with tokenization. Supports `get_subset_dataset()` for sampling subsets.
- `CustomDataset_tracked`: Extends `CustomDataset` with tweet ID tracking (`idxes`), used throughout the pseudo-labeling pipeline to maintain provenance of unlabeled instances.

## Dataset

### Source

The datasets contain tweets from **10 real-world disaster events** spanning 2016–2019, annotated with humanitarian categories. Each tweet is classified into one of up to 10 categories describing the type of information it conveys.

### Humanitarian Categories

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

**Important:** Not every disaster has all 10 classes. Some events have only 7, 8, or 9 classes depending on the types of tweets observed:

| Disaster | Classes |
|---|:---:|
| california_wildfires_2018 | 10 |
| cyclone_idai_2019 | 10 |
| canada_wildfires_2016 | 8 (missing: `injured_or_dead_people`, `missing_or_found_people`) |
| All others | 9 (missing: `missing_or_found_people`) |

The code dynamically detects the actual classes per disaster at runtime.

### Data Splits

Each disaster directory contains the following TSV files:

```
data/{disaster}/
├── {disaster}_train.tsv           # Full training pool (used to create labeled/unlabeled splits)
├── {disaster}_dev.tsv             # Validation set (fixed, same across all splits)
├── {disaster}_test.tsv            # Test set (fixed, same across all splits)
├── labeled_{k}_set{s}.tsv         # Few-shot labeled subset
└── unlabeled_{k}_set{s}.tsv       # Corresponding unlabeled pool
```

**Few-shot labeled splits** are created by sampling `k` examples per class from the full training pool:
- **Sizes (`k`):** 5, 10, 25, 50 examples per class
- **Sets (`s`):** 1, 2, 3 (three independent random samples for each size)

This gives **12 labeled splits per disaster** (4 sizes × 3 sets). The three sets at each size allow measuring variance due to the specific labeled examples chosen.

For example, `labeled_5_set1.tsv` contains 5 examples per class (50 total for a 10-class disaster), and `unlabeled_5_set1.tsv` contains the remaining training tweets not included in that labeled split.

**TSV format** (tab-separated, with header):

```
tweet_id    tweet_text    class_label
1061291825879691264    Be in Ventura County later today...    caution_and_advice
```

| Split | Example Size (california_wildfires_2018) |
|---|---|
| `labeled_5_set1.tsv` | 50 tweets (5 per class × 10 classes) |
| `labeled_50_set1.tsv` | 500 tweets (50 per class × 10 classes) |
| `unlabeled_5_set1.tsv` | ~5,100 tweets |
| `{disaster}_dev.tsv` | ~750 tweets |
| `{disaster}_test.tsv` | ~1,460 tweets |

The dev and test sets are **shared across all splits** for a given disaster — only the labeled/unlabeled partition changes.

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

The notebook `notebooks/st-experiments.ipynb` provides a structured workflow for running Self-Training experiments at scale.

### Cells

1. **Setup** — Imports, seeds, GPU detection, logging suppression
2. **Helpers** — `detect_classes()` for dynamic class detection, `get_dataset()` and `load_disaster_datasets()` for data loading
3. **Configuration** — Model checkpoint, hyperparameters, and all 12 train file combinations (`LABEL_SIZES × SETS`)
4. **Part 1: Sanity Check** — Runs a single experiment (`california_wildfires_2018`, `5_set1`) to verify the pipeline works end-to-end
5. **Part 2: Pre-scan** — Scans all 120 experiments (10 disasters × 12 splits), shows which are done vs. pending
6. **Part 2: Full Experiment** — Runs all pending experiments with progress tracking and resume support
7. **Summary** — Pivot tables showing macro-F1 and ECE by disaster × label size

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

Model checkpoints are saved as `data/{disaster}/pytorch_model.bin` (overwritten each run).

## References

- Mukherjee, S., & Awadallah, A. (2020). [Uncertainty-aware Self-training for Few-shot Text Classification](https://arxiv.org/abs/2006.15315). NeurIPS 2020.
- Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). [BERTweet: A Pre-trained Language Model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2/). EMNLP 2020.
