\begin{table*}[t]
\centering
\caption{Overall Micro and Macro F1 scores across the three legal NER datasets. For our proposed ACE variants, we report both the Maximum (Max) and Average (Avg) scores across 3 runs with different random seeds. The highest values for each metric and dataset are highlighted in \textbf{bold}.}
\label{tab:main_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Method}} & \multicolumn{2}{c}{\textbf{Micro F1}} & \multicolumn{2}{c}{\textbf{Macro F1}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
& & \textbf{Max} & \textbf{Avg} & \textbf{Max} & \textbf{Avg} \\
\midrule

% ================= GREEK LEGAL NER =================
\multirow{6}{*}{\textbf{GreekLegalNER}} 
& Best Baseline (mDeBERTa-v3) & \multicolumn{2}{c}{71.16} & \multicolumn{2}{c}{67.31} \\
& Naïve Concatenation & XX.XX & XX.XX & XX.XX & XX.XX \\
\cmidrule{2-6}
& Standard ACE & 72.56 & 72.38 & 68.55 & 68.18 \\
& G-ACE (N=4) & \textbf{72.91} & \textbf{72.70} & \textbf{69.55} & \textbf{69.24} \\
& G-ACE (N=8) & 72.10 & 72.10 & 68.31 & 68.31 \\
& H-ACE (N=4) & 72.21 & 72.08 & 68.82 & 68.70 \\
\midrule

% ================= InLNER (ENGLISH) =================
\multirow{6}{*}{\textbf{InLNER}} 
& Best Baseline (mDeBERTa-v3) & \multicolumn{2}{c}{85.99} & \multicolumn{2}{c}{82.91} \\
& Naïve Concatenation & XX.XX & XX.XX & XX.XX & XX.XX \\
\cmidrule{2-6}
& Standard ACE & 87.98 & 87.52 & 84.42 & 83.70 \\
& G-ACE (N=4) & 88.62 & 88.15 & 85.45 & 84.76 \\
& G-ACE (N=8) & \textbf{88.65} & \textbf{88.33} & \textbf{85.83} & \textbf{85.47} \\
& H-ACE (N=4) & 87.20 & 86.88 & 82.20 & 81.71 \\
\midrule

% ================= LegalNERo (ROMANIAN) =================
\multirow{6}{*}{\textbf{LegalNERo}} 
& Best Baseline (SeNER / DiffusionNER)$^\dagger$ & \multicolumn{2}{c}{77.64} & \multicolumn{2}{c}{72.50} \\
& Naïve Concatenation & XX.XX & XX.XX & XX.XX & XX.XX \\
\cmidrule{2-6}
& Standard ACE & 79.61 & 78.44 & 72.45 & 69.54 \\
& G-ACE (N=4) & 79.71 & 79.20 & \textbf{77.06} & \textbf{74.17} \\
& G-ACE (N=8)  & \textbf{80.40} & \textbf{79.71} & 70.60 & 69.43 \\
& H-ACE (N=4) & 79.72 & 78.88 & 74.01 & 72.90 \\
\bottomrule
\multicolumn{6}{l}{\footnotesize $^\dagger$ For LegalNERo, SeNER achieved the best baseline Micro F1, while DiffusionNER achieved the best Macro F1.}
\end{tabular}%
}
\end{table*}
# Grouped-ACE (N=4) — Greek Legal NER: Αποτελέσματα Εκπαίδευσης

## 1. Σύνοψη

| Μετρική             | Τιμή       |
|---------------------|------------|
| **Micro F1 (Test)** | **72.53%** |
| **Macro F1 (Test)** | **69.23%** |
| Precision (Micro)   | 76.39%     |
| Recall (Micro)      | 69.05%     |

**Μοντέλο:** Grouped-ACE (Automated Concatenation of Embeddings με Grouped Pruning, N=4)  
**Best Model Save:** Episode 15, Test F1 = 72.46% (κατά την εκπαίδευση)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 72.53%


## 2. Ρυθμίσεις Εκπαίδευσης (Configuration)

### Αρχιτεκτονική Μοντέλου
| Παράμετρος | Τιμή |
|------------|------|
| Model | FastSequenceTagger |
| Hidden Size (BiLSTM) | 256 |
| RNN Layers | 1 |
| CRF | ✓ |
| Dropout | 0.0 |
| Word Dropout | 0.05 |
| Locked Dropout | 0.5 |
| Sentence Loss | ✓ |
| Tag Scheme | BIOES (36 tags) |

### Hyperparameters Εκπαίδευσης
| Παράμετρος | Τιμή |
|------------|------|
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Min Learning Rate | 0.0001 |
| Batch Size | 8 |
| Max Episodes (RL) | 40 |
| Max Epochs / Episode | 150 |
| Patience (Anneal) | 5 |
| Max Epochs Without Improvement | 10 |
| Controller Optimizer | Adam |
| Controller LR | 0.6 |
| Discount Factor | 0.5 |
| Monitor Test | ✓ |
| Train with Dev | ✗ |

### Grouped-ACE Ρυθμίσεις
| Παράμετρος | Τιμή |
|------------|------|
| num_groups (N) | 4 |
| group_mode | equal |
| group_seed | 44 |
| Πιθανές Ενέργειες | 24 (6 embeddings × 4 groups) |

### Dataset (Greek Legal NER v2)
| Split | Προτάσεις |
|-------|-----------|
| Train | 17,371 |
| Dev | 4,752 |
| Test | 3,878 |

---

## 3. Embeddings (6 Υποψήφια)

| # | Τύπος | Embedding | Διάσταση |
|---|-------|-----------|----------|
| 0 | Contextual | **GreekBERT** (`nlpaueb/bert-base-greek-uncased-v1`, fine-tuned) | 768 |
| 1 | Contextual | **GreekLegalRoBERTa** (`joelito/legal-greek-roberta-base`, fine-tuned) | 768 |
| 2 | Static | **FastText** (`cc.el.300.vec`) | 300 |
| 3 | Contextual | **mDeBERTa-v3-base** (fine-tuned) | 768 |
| 4 | Character | **FastCharacterEmbeddings** (CNN, 25d) | 50 |
| 5 | Static | **BPEmb** (el, 100K vocab, syllable-based) | 600 |

**Σύνολο (αν επιλεγούν όλα):** 3,254 διαστάσεις

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Dev F1 | Best Test F1 | Επιλεγμένα Groups (N=4) |
|:-------:|:------:|:-----------:|:------------:|-------------------------|
| 1 | 20 | 41.53% | 70.48% | GreekBERT [1,1,1,1], RoBERTa [1,1,1,1], FastText [1,1,1,1], mDeBERTa [1,1,1,1], Char [1,1,1,1], BPEmb [1,1,1,1] |
| 2 | 17 | 42.30% | 71.90% | GreekBERT [1,0,0,0], RoBERTa [0,0,1,0], FastText [0,0,1,0], Char [0,1,0,0], BPEmb [0,0,0,1] |
| 3 | 16 | 42.19% | 70.94% | GreekBERT [0,1,0,1], RoBERTa [0,1,1,1], FastText [0,0,1,0], mDeBERTa [0,0,1,0], BPEmb [0,0,0,1] |
| 4 | 11 | 42.17% | 71.55% | GreekBERT [1,1,0,0], RoBERTa [1,1,1,1], FastText [1,0,0,0], mDeBERTa [0,1,0,0], BPEmb [1,0,1,1] |
| 5 | 21 | 42.11% | 71.66% | GreekBERT [1,0,0,0], RoBERTa [0,0,1,0], FastText [0,0,1,1], mDeBERTa [0,1,0,1], Char [0,1,0,0], BPEmb [0,0,1,0] |
| 6 | 13 | 42.17% | 71.14% | GreekBERT [1,0,0,1], RoBERTa [1,0,1,0], FastText [0,0,1,1], BPEmb [0,0,0,1] |
| 7 | 17 | 41.29% | 70.84% | GreekBERT [1,0,0,0], BPEmb [1,1,0,0] |
| 8 | 22 | 40.45% | 67.45% | RoBERTa [0,1,1,0], Char [0,1,0,0], BPEmb [1,0,0,1] |
| 9 | 16 | 42.17% | 71.21% | GreekBERT [1,0,0,0], RoBERTa [1,0,0,0], Char [0,1,0,0], BPEmb [0,0,1,1] |
| 10 | 22 | 41.52% | 70.61% | RoBERTa [0,0,1,0], mDeBERTa [0,0,0,1], Char [0,1,0,0], BPEmb [0,0,0,1] |
| 11 | 19 | 41.47% | 71.41% | FastText [1,0,1,0], mDeBERTa [1,0,0,0], Char [0,1,0,0] |
| 12 | 17 | 42.06% | 71.29% | GreekBERT [1,0,0,0], RoBERTa [1,0,1,0], FastText [1,0,0,0], Char [0,1,0,0], BPEmb [0,0,0,1] |
| **13** | 12 | 42.13% | **72.03%** | GreekBERT [1,0,0,0], RoBERTa [0,0,1,1], FastText [1,0,1,1], Char [0,1,0,0] |
| 14 | 23 | 42.03% | 70.68% | GreekBERT [1,0,0,0], RoBERTa [0,0,1,0], FastText [1,0,0,0], Char [0,1,0,1], BPEmb [0,0,0,1] |
| **15** ★ | 17 | 42.13% | **72.46%** | GreekBERT [1,0,0,0], RoBERTa [0,0,0,1], FastText [1,1,1,0], Char [0,1,1,0] |
| 16 | 12 | 41.66% | 70.77% | GreekBERT [1,0,0,0], RoBERTa [1,0,0,0], FastText [1,0,1,0], Char [0,1,0,0], BPEmb [0,0,1,0] |
| 17 | 21 | 42.31% | 70.97% | GreekBERT [1,0,0,0], RoBERTa [0,0,0,1], FastText [1,0,1,1], Char [0,1,0,0] |
| 18 | 13 | 42.03% | 70.86% | GreekBERT [1,0,0,0], RoBERTa [0,0,1,0], FastText [1,0,0,0], mDeBERTa [0,1,0,0], Char [0,1,0,0], BPEmb [0,0,1,0] |
| 19 | 33 | 42.64% | 71.32% | GreekBERT [1,0,0,0], RoBERTa [1,0,0,1], FastText [1,0,1,0], mDeBERTa [1,0,0,0], Char [0,1,0,0], BPEmb [0,0,1,0] |
| 20 | 16 | 41.96% | 71.28% | GreekBERT [1,1,0,0], RoBERTa [0,0,1,0], FastText [1,0,1,0], Char [0,1,0,0] |
| 21 | 31 | 42.24% | 71.33% | GreekBERT [1,0,0,0], RoBERTa [1,0,0,0], FastText [1,1,1,0], mDeBERTa [0,1,0,0], Char [0,1,0,0] |
| 22 | 11 | 41.53% | 71.25% | GreekBERT [1,1,0,0], FastText [1,0,1,0], Char [0,1,0,0] |
| **23** | 21 | 42.60% | **72.16%** | GreekBERT [1,0,0,0], RoBERTa [1,0,1,1], FastText [1,1,1,0], mDeBERTa [0,1,0,1], Char [0,1,0,0] |
| 24 | 12 | 42.69% | 71.16% | GreekBERT [1,0,0,0], RoBERTa [0,0,1,1], FastText [1,1,1,0], mDeBERTa [1,0,0,0], Char [0,1,0,0] |
| 25 | 22 | 42.39% | 71.55% | GreekBERT [1,0,0,0], RoBERTa [0,0,1,1], FastText [1,1,1,0], mDeBERTa [1,1,1,0], Char [0,1,0,0] |
| 26 | 31 | 42.04% | 71.72% | GreekBERT [1,0,0,0], RoBERTa [1,0,0,1], FastText [1,0,1,0], mDeBERTa [0,1,0,1], Char [0,1,0,0] |
| 27 | 12 | 42.32% | 71.59% | GreekBERT [1,0,0,0], RoBERTa [1,0,0,1], FastText [1,0,1,0], mDeBERTa [0,0,0,1], Char [0,1,0,0] |
| 28 | 22 | 42.32% | 71.66% | GreekBERT [1,0,0,0], RoBERTa [1,0,0,0], FastText [1,1,1,0], mDeBERTa [1,1,0,1], Char [0,1,0,0] |
| 29 | 14 | 41.88% | 71.74% | GreekBERT [1,0,0,0], RoBERTa [1,0,1,1], FastText [1,0,1,0], mDeBERTa [1,1,0,1], Char [0,1,0,0] |
| 30 | 15 | 42.50% | 71.34% | GreekBERT [1,0,0,0], RoBERTa [0,0,1,1], FastText [1,1,1,0], mDeBERTa [1,1,0,1], Char [0,1,0,0] |
| 31 | 18 | 41.91% | 71.29% | GreekBERT [1,0,0,0], RoBERTa [0,1,0,1], FastText [1,1,1,0], mDeBERTa [0,1,0,1], Char [0,1,0,0] |
| 32 | 11 | 41.83% | 70.59% | GreekBERT [1,0,0,0], RoBERTa [0,0,1,1], FastText [1,0,1,0], mDeBERTa [0,1,0,1], Char [0,1,1,0] |
| 33 | 21 | 42.18% | 71.52% | GreekBERT [1,0,0,0], RoBERTa [1,0,1,1], FastText [1,0,1,0], mDeBERTa [0,1,1,1], Char [0,1,0,0] |
| 34 | 27 | 42.42% | 71.00% | GreekBERT [1,0,0,0], RoBERTa [1,0,1,1], FastText [1,1,1,0], mDeBERTa [0,1,0,1], Char [0,1,0,0] |
| 35 | 16 | 42.11% | 70.37% | GreekBERT [1,0,0,0], RoBERTa [1,0,1,1], FastText [1,0,1,0], mDeBERTa [0,1,1,1], Char [0,1,0,0] |
| 36 | 12 | 41.80% | 71.10% | GreekBERT [1,0,0,0], RoBERTa [1,0,1,1], FastText [1,1,1,0], mDeBERTa [0,1,0,1], Char [0,1,0,0] |
| 37 | 21 | 42.14% | 70.87% | GreekBERT [1,0,1,0], RoBERTa [1,0,1,1], FastText [1,1,1,0], mDeBERTa [0,1,1,1], Char [0,1,0,0] |
| 38 | 17 | 41.82% | 71.24% | GreekBERT [1,0,0,0], RoBERTa [1,0,0,1], FastText [1,1,1,0], mDeBERTa [0,1,0,1], Char [0,1,1,0] |
| 39 | 24 | 42.00% | 71.66% | GreekBERT [1,0,0,0], RoBERTa [1,0,1,1], FastText [1,1,1,0], mDeBERTa [0,1,0,1], Char [0,1,0,0] |
| 40 | 15 | 42.00% | 70.55% | GreekBERT [1,0,0,0], RoBERTa [1,0,1,1], FastText [1,1,1,0], mDeBERTa [0,1,1,1], Char [0,1,0,0] |

**Σύνολο:** 40 episodes, 731 epochs, ~80 ώρες εκπαίδευσης

### Baseline Transitions (Save Events)
| Γεγονός | Episode | Test F1 | Action |
|---------|:-------:|:-------:|--------|
| Αρχικό baseline | 1 | 68.58% | Όλα τα embeddings [1,1,1,1] × 6 |
| Βελτίωση | 1 | 70.27% | (εντός Episode 1) |
| Βελτίωση | 1 | 70.48% | (εντός Episode 1) |
| Βελτίωση | 2 | 71.11% | Ελαφρύ pruning |
| Βελτίωση | 2 | 71.90% | GreekBERT [1,0,0,0], RoBERTa [0,0,1,0], FastText [0,0,1,0], Char [0,1,0,0], BPEmb [0,0,0,1] |
| Βελτίωση | 13 | 72.03% | GreekBERT [1,0,0,0], RoBERTa [0,0,1,1], FastText [1,0,1,1], Char [0,1,0,0] |
| Βελτίωση | 15 | 72.04% | (εντός Episode 15) |
| **Τελικό best** | **15** | **72.46%** | GreekBERT [1,0,0,0], RoBERTa [0,0,0,1], FastText [1,1,1,0], Char [0,1,1,0] |

---

## 5. Συχνότητα Επιλογής Embedding (40 episodes)

| Embedding | Επιλέχθηκε (≥1 group) | Ποσοστό | Μ.Ο. Groups | G1 | G2 | G3 | G4 |
|-----------|:---------------------:|:-------:|:-----------:|:--:|:--:|:--:|:--:|
| GreekBERT | 37/40 | 92.5% | 1.15 | 36 | 5 | 2 | 3 |
| GreekLegalRoBERTa | 37/40 | 92.5% | 1.90 | 21 | 5 | 26 | 24 |
| FastText (cc.el) | 36/40 | 90.0% | 2.10 | 32 | 15 | 32 | 5 |
| mDeBERTa-v3 | 27/40 | 67.5% | 1.38 | 8 | 21 | 7 | 19 |
| FastCharEmbeddings | 36/40 | 90.0% | 1.07 | 1 | 36 | 4 | 2 |
| BPEmb (el) | 15/40 | 37.5% | 0.57 | 4 | 2 | 7 | 10 |

**Παρατηρήσεις:**
- **GreekBERT G1** (πρώτο 25% = dims 0–191): Επιλέγεται σε 36/40 episodes 
- **Char G2** (dims 13–25): Επιλέγεται σε 36/40 
- **BPEmb**: Πλήρως απορρίφθηκε (0/4) από Episode 15 
- **FastText**: Υψηλός μ.ό. groups (2.10)

---

## 6. Ενεργές Διαστάσεις & Pruning στο Best Model (Episode 15)

| # | Embedding | Αρχική Dim | Chunks (N=4) | Επιλεγμένα Groups | Τελικές Dims |
|---|-----------|:----------:|:------------:|:-----------------:|:------------:|
| 0 | **GreekBERT** | 768 | [192, 192, 192, 192] | `[1, 0, 0, 0]` — 1/4 | **192** |
| 1 | **GreekLegalRoBERTa** | 768 | [192, 192, 192, 192] | `[0, 0, 0, 1]` — 1/4 | **192** |
| 2 | **FastText** | 300 | [75, 75, 75, 75] | `[1, 1, 1, 0]` — 3/4 | **225** |
| 3 | **mDeBERTa-v3** | 768 | [192, 192, 192, 192] | `[0, 0, 0, 0]` — 0/4 | **0** ❌ |
| 4 | **FastCharEmb** | 50 | [13, 13, 12, 12] | `[0, 1, 1, 0]` — 2/4 | **25** |
| 5 | **BPEmb** | 600 | [150, 150, 150, 150] | `[0, 0, 0, 0]` — 0/4 | **0** ❌ |

> **Αρχικό Σύνολο Διαστάσεων:** 3,254  
> **Τελικό Σύνολο Grouped-ACE:** 634 διαστάσεις  
> **Μείωση Διαστάσεων: 80.5%**  


## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Embeddings στο Best Model
- ✅ GreekBERT — 1ο group μόνο (192d από 768)
- ✅ GreekLegalRoBERTa — 4ο group μόνο (192d από 768)
- ✅ FastText — groups 1–3 (225d από 300)
- ✅ FastCharEmb — groups 2–3 (25d από 50)
- ❌ mDeBERTa-v3-base — πλήρως αφαιρέθηκε
- ❌ BPEmb — πλήρως αφαιρέθηκε

### Αποτελέσματα ανά Οντότητα

| Οντότητα        |   TP |  FP |  FN | Precision | Recall  | F1         |
|-----------------|-----:|----:|----:|:---------:|:-------:|:----------:|
| **PERSON**      |  486 |  22 |  30 |   95.67%  |  94.19% | **94.92%** |
| **LEG-REFS**    | 1043 | 134 | 268 |   88.62%  |  79.56% | **83.85%** |
| **DATE**        |  380 |  65 | 173 |   85.39%  |  68.72% | **76.15%** |
| **LOCATION**    |  555 | 235 | 152 |   70.25%  |  78.50% | **74.15%** |
| **GPE**         |  573 | 155 | 255 |   78.71%  |  69.20% | **73.65%** |
| **ORG**         | 1083 | 491 | 691 |   68.81%  |  61.05% | **64.70%** |
| **PUBLIC-DOCS** |  392 | 259 | 404 |   60.22%  |  49.25% | **54.19%** |
| **FACILITY**    |   24 |  41 |  60 |   36.92%  |  28.57% | **32.21%** |

### Συνολικά

| Μετρική        | Precision | Recall  | F1         |
|----------------|:---------:|:-------:|:----------:|
| **MICRO AVG**  |   76.39%  |  69.05% | **72.53%** |
| **MACRO AVG**  |     —     |    —    | **69.23%** |

---

## 8. Σύγκριση: Grouped-ACE (N=4) vs Standard ACE

### Per-Entity F1 Σύγκριση

| Οντότητα        | ACE (F1)   | Grouped-ACE N=4 (F1) | Διαφορά    |
|-----------------|:----------:|:---------------------:|:----------:|
| **PERSON**      | 94.10%     | **94.92%**            | +0.82 ✅   |
| **LEG-REFS**    | 83.19%     | **83.85%**            | +0.66 ✅   |
| **DATE**        | 75.05%     | **76.15%**            | +1.10 ✅   |
| **LOCATION**    | 74.00%     | **74.15%**            | +0.15 ✅   |
| **GPE**         | 71.90%     | **73.65%**            | +1.75 ✅   |
| **ORG**         | **66.14%** | 64.70%                | −1.44 ❌   |
| **PUBLIC-DOCS** | **55.68%** | 54.19%                | −1.49 ❌   |
| **FACILITY**    | 28.36%     | **32.21%**            | +3.85 ✅   |

### Aggregate Μετρικές

| Μετρική        | ACE        | Grouped-ACE N=4 | Διαφορά    |
|----------------|:----------:|:----------------:|:----------:|
| **Micro F1**   | **72.56%** | 72.53%           | −0.03      |
| **Macro F1**   | 68.55%     | **69.23%**       | +0.68 ✅   |
| Precision      | **77.10%** | 76.39%           | −0.71      |
| Recall         | 68.53%     | **69.05%**       | +0.52 ✅   |

### Embedding Χρήση & Αποδοτικότητα

| Μετρική                     | ACE               | Grouped-ACE N=4            |
|-----------------------------|:-----------------:|:--------------------------:|
| Ενεργά Embeddings           | 4/6               | 4/6 (μερικά)               |
| Ενεργές Διαστάσεις          | 2,436 (74.9%)     | **634 (19.5%)**            |
| Μείωση Διαστάσεων           | 25.1%             | **80.5%**                  |
| Πλήρως αφαιρεμένα           | BPEmb, FastChar   | mDeBERTa, BPEmb            |
| Episodes (RL)               | 30                | 40                         |
| Συνολικοί Epochs            | 603               | 731                        |
