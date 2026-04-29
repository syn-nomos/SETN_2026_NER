# Granular Embedding Selection for Legal Named Entity Recognition

Companion code repository for the **SETN 2026** submission
**"Granular Embedding Selection for Legal Named Entity Recognition"** (anonymous review).

This repository implements two new representation-selection mechanisms — **Grouped ACE
(G-ACE)** and **Hierarchical ACE (H-ACE)** — and benchmarks them against strong
single-embedding baselines, naïve concatenation, and the original Automated Concatenation
of Embeddings (ACE) framework on three legal NER corpora covering English, Greek, and
Romanian.

> **Anonymity notice.** Author / institution information has been removed. Please do not
> attempt to deanonymize the contributors of this repository during the review period.

---

## TL;DR — what this repo adds

| Method | Selection unit | Search space | Where in this repo |
|---|---|---|---|
| **ACE** *(re-implemented as a baseline)* | whole embedding | $L$ binary actions | `flair/trainers/reinforcement_trainer.py` (Eq. 4 controller) |
| **G-ACE (ours)** | sub-blocks within each embedding | $L \cdot K$ binary actions | `flair/trainers/reinforcement_trainer.py` (`grouped_*` paths) |
| **H-ACE (ours)** | family-level + block-level (two-stage) | $L + L \cdot K$ binary actions | `flair/trainers/reinforcement_trainer.py` (`hierarchical_*` paths, Eq. 7) |

Both G-ACE and H-ACE move the selection unit from *whole embeddings* (as in ACE) to
*structured sub-spaces*, exposing a much richer Pareto frontier between Micro-F1 and the
number of active dimensions.

The candidate pool is identical across the three datasets:

$$D = 3 \times 768 + 300 + 600 + 50 = 3{,}254 \text{ dimensions}$$

(three contextual encoders + FastText + BPEmb + character embedding).

---

## Headline results (Table 4 of the paper)

Mean ± SD over 3 seeds (44, 55, 66). Active-dim percentages are with respect to the
$D = 3{,}254$ candidate pool.

| Dataset | Method | Active Dims | Micro-F1 | Macro-F1 |
|---|---|---:|---:|---:|
| **InLegalNER** (English) | Best single (mDeBERTa-v3) | 768 | 85.99 | 82.91 |
| | Naïve concat | 3,254 (100%) | 88.35 ± 0.12 | 85.26 ± 0.16 |
| | ACE | 2,604 (80.0%) | 87.52 ± 0.66 | 83.70 ± 1.02 |
| | **G-ACE (K=8)** | **1,505 (46.3%)** | **88.33 ± 0.30** | **85.47 ± 0.38** |
| | H-ACE (K=4) | 1,418 (43.6%) | 86.88 ± 0.42 | 81.71 ± 0.81 |
| **GLNv2** (Greek) | Best single (mDeBERTa-v3) | 768 | 71.16 | 67.31 |
| | Naïve concat | 3,254 (100%) | 71.56 ± 0.43 | 68.16 ± 0.40 |
| | ACE | 2,436 (74.9%) | 72.29 ± 0.11 | 67.99 ± 0.51 |
| | **G-ACE (K=4)** | **805 (24.7%)** | **72.70 ± 0.19** | **69.24 ± 0.31** |
| | H-ACE (K=4) | 986 (30.3%) | 72.10 ± 0.13 | 68.38 ± 0.56 |
| **LegalNERo** (Romanian) | Best single (SeNER) | 768 | 77.64 | 72.50 |
| | Naïve concat | 3,254 (100%) | 79.02 ± 0.20 | 71.67 ± 4.05 |
| | ACE | 2,354 (72.3%) | 78.44 ± 1.25 | 69.54 ± 4.48 |
| | **G-ACE (K=8)** | **1,350 (41.5%)** | **79.71 ± 0.60** | 69.43 ± 1.27 |
| | **H-ACE (K=4)** | **818 (25.1%)** | **78.76 ± 0.77** | 71.13 ± 3.31 |

**Key takeaway.** G-ACE consistently sits on or above the Micro-F1 / compactness Pareto
frontier; H-ACE compresses the input down to roughly a quarter of the candidate features
while staying competitive with naïve concatenation.

---

## Datasets

| Dataset | Language | Sentences | Entity classes | Source |
|---|---|---:|---:|---|
| **InLegalNER** | English | 9,435 | 13 | OpenNyAI |
| **GreekLegalNER v2 (GLNv2)** | Greek | 26,002 | 8 | Anonymous repository (re-annotated v2) |
| **LegalNERo** | Romanian | — | — | `joelniklaus/legalnero` (Hugging Face) |

Pre-processed BIOES train/dev/test splits for one of the datasets are shipped under
`data/`. The remaining datasets can be regenerated with the helpers in `_temp_scripts_/`
(see `download_embeddings.py` and the corpus loaders in `flair/datasets.py`).

---

## Candidate embedding pool (per dataset)

The three contextual encoders are dataset-specific; FastText, BPEmb and the character
embedding are shared across datasets.

| Slot | InLegalNER | GLNv2 | LegalNERo | Dim |
|---|---|---|---|---:|
| Contextual #1 | InLegalBERT | GreekBERT | LegalRoBERTa-Base | 768 |
| Contextual #2 | LegalBERT | GreekLegalBERT | LegalRoBERTa-Large (proj.) | 768 |
| Contextual #3 | mDeBERTa-v3 | mDeBERTa-v3 | SeNER (mDeBERTa-v3) | 768 |
| FastText (CommonCrawl / Wiki-News) | en | el | ro | 300 |
| BPEmb (forward + backward, 300 each) | en | el | ro | 600 |
| Character embedding | trained | trained | trained | 50 |
| **Total** | | | | **3,254** |

---

## Repository layout

```
.
├── train.py                       # main entry point
├── extract_features.py            # offline feature extraction
├── generate_table.py              # rebuild result tables from logs
├── config/                        # 36 YAMLs covering the paper experiments
│   ├── gln_*.yaml                 # GreekLegalNER v2
│   ├── inlner_*.yaml              # InLegalNER
│   └── legalnero_*.yaml           # LegalNERo
├── flair/                         # heavily modified flair 0.4.3 (vendored)
│   ├── trainers/reinforcement_trainer.py   # ACE / G-ACE / H-ACE controllers
│   ├── embeddings.py              # candidate embedding wrappers
│   ├── models/                    # BiLSTM-CRF task model
│   └── ...
├── algorithms/, utils/, tools/, script/, tests/
├── data/                          # train/dev/test (BIOES) for one dataset
├── _temp_scripts_/                # analysis utilities
│   ├── compute_selection_stability.py     # Table 3 (J_fam, J_block, J_dim)
│   ├── compute_dim_jaccard.py             # per-dim Jaccard
│   ├── compute_dim_jaccard_full.py        # per-dim Jaccard, full pool
│   ├── collect_results.py                 # aggregate run-level metrics
│   ├── extract_log.py                     # parse training logs
│   ├── download_embeddings.py             # bootstrap candidate embeddings
│   └── run_baselines_3x.sh                # 3-seed baseline driver
├── RESULTS/                       # per-method MD reports & generator
│   ├── GLN/  INLNER/  LEGALNERO/
│   └── scripts/generate_report.py
├── figures/                       # paper figures (LaTeX/TikZ sources)
├── notes/                         # exploratory research notes (non-paper)
├── requirements.txt               # runtime dependencies
├── requirements_ace.txt           # extra dependencies for ACE controller
└── README_ACE_ORIGINAL.md         # original ACE README, kept for upstream credit
```

---

## Installation

```bash
# Python 3.11 recommended (tested under conda)
conda create -n setn26 python=3.11 -y
conda activate setn26

pip install -r requirements.txt
pip install -r requirements_ace.txt
```

The vendored `flair/` directory shadows the upstream `flair` package; do not install
`flair` from PyPI on top of it.

---

## Reproducing the experiments

Each YAML in `config/` is fully self-contained. The naming scheme is:

```
<dataset>_<method>[_seed{44,55,66}].yaml
   dataset ∈ {gln, inlner, legalnero}
   method  ∈ {baseline, ace_original, grouped_N{4,8}_original, grouped_N4_h_ace}
```

Train a single configuration:

```bash
python train.py --config config/gln_grouped_N4_original_seed44.yaml
```

Run the full 3-seed sweep for one method (example: G-ACE-4 on GLNv2):

```bash
for s in 44 55 66; do
  python train.py --config config/gln_grouped_N4_original_seed${s}.yaml
done
```

After training, regenerate the per-run Markdown reports:

```bash
python RESULTS/scripts/generate_report.py path/to/training.log \
    --config path/to/used_config.yaml \
    -o RESULTS/<DATASET>/<METHOD>/run_seedXX.md
```

Aggregate selection-stability statistics across the three seeds (Table 3):

```bash
python _temp_scripts_/compute_selection_stability.py \
    --runs RESULTS/GLN/GROUPED_ACE_4/seed{44,55,66}.md
```

---

## Hyperparameters (matches §4.3 of the paper)

| Component | Setting |
|---|---|
| Task model | single-layer BiLSTM (hidden 256) + CRF over BIOES |
| Locked dropout | 0.5 |
| Word dropout | 0.05 |
| Optimizer (task model) | SGD, lr 0.1, anneal ×0.5 after 5 stagnant dev epochs |
| Mini-batch | 8 sentences |
| Max epochs / episode | 150 |
| Episodes / run | up to 30 |
| Embeddings | frozen during training |
| Controller (ACE) | SGD, lr 0.1, $\gamma = 0.5$, zero-init |
| Controller (G-ACE / H-ACE) | tuned for the larger $L \cdot K$ action space (see code) |
| Group partitioning | $K \in \{4, 8\}$ disjoint blocks per embedding |
| Seeds | 44, 55, 66 (same value used as `group_seed`) |

---

## Selection stability (Table 3 of the paper)

We measure how reproducible a method’s selections are across the three independent
seeds at three granularities:

- $J_{\text{fam}}$ — embedding-family level (which of the 6 candidate families are kept)
- $J_{\text{block}}$ — block level (which of the $L \cdot K$ blocks are kept)
- $J_{\text{dim}}$ — individual dimensions, vs. a random-permutation null $J_{\text{rnd}}$

Across all three datasets we observe $J_{\text{fam}} \approx 1$, much lower
$J_{\text{block}}$, and $J_{\text{dim}} - J_{\text{rnd}}$ close to zero — i.e., the
controller agrees on *which* families are useful, but the specific sub-blocks chosen
within them vary substantially without degrading final F1, exposing functional
redundancy in pretrained encoders.

Reproduce with:

```bash
python _temp_scripts_/compute_dim_jaccard.py        # block-level Jaccard
python _temp_scripts_/compute_dim_jaccard_full.py   # dim-level Jaccard
python _temp_scripts_/compute_selection_stability.py
```

---

## Relation to upstream ACE

The original [ACE](https://aclanthology.org/2021.acl-long.206/) framework operates at the
level of whole embeddings. This repository started from the public ACE codebase and
substantially extended the vendored `flair` package to support:

1. **Block-level decisions** within each embedding (G-ACE).
2. **Two-stage hierarchical decisions** (family → block) with an outer-stage controller
   that gates the inner stage (H-ACE, Eq. 7 of the paper).
3. A modified reward / advantage flow that handles the much sparser
   $L \cdot K$-dimensional action space.
4. New corpus loaders, embedding wrappers, BIOES decoders and reporting utilities for
   the three legal-NER datasets.

The original ACE README is preserved verbatim as
[`README_ACE_ORIGINAL.md`](README_ACE_ORIGINAL.md) for upstream credit. Anything
described in §3 ("Method") of our paper — G-ACE, H-ACE, the controller modifications and
the granular selection analyses — is original work introduced by this repository.

---

## License

This repository inherits the MIT license of the upstream ACE codebase
(see [`LICENSE`](LICENSE)).

---

## Citation

A BibTeX entry will be added after the SETN 2026 review process. Until then, please
cite this work as:

```
@inproceedings{anonymous2026granular,
  title     = {Granular Embedding Selection for Legal Named Entity Recognition},
  author    = {Anonymous},
  booktitle = {Proceedings of the 14th Hellenic Conference on Artificial Intelligence (SETN)},
  year      = {2026},
  note      = {Under review}
}
```
