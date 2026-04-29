# ACE — Indian Legal NER: Αποτελέσματα Εκπαίδευσης

## 1. Σύνοψη

| Μετρική | Τιμή |
|---------|------|
| **Micro F1 (Test)** | **87.92%** |
| **Macro F1 (Test)** | **84.91%** |
| Precision (Micro) | 88.11% |
| Recall (Micro) | 87.74% |
| Accuracy (Micro) | 78.45% |

**Μοντέλο:** ACE (Automated Concatenation of Embeddings) με RL Controller  
**Best Model Save:** Episode 29, Test F1 = 87.82% (κατά την εκπαίδευση)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 87.92%

---

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
| Tag Scheme | BIOES (56 tags) |

### Hyperparameters Εκπαίδευσης
| Παράμετρος | Τιμή |
|------------|------|
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Min Learning Rate | 0.0001 |
| Batch Size | 8 |
| Max Episodes (RL) | 30 |
| Max Epochs / Episode | 150 |
| Patience (Anneal) | 5 |
| Max Epochs Without Improvement | 15 |
| Controller Optimizer | SGD |
| Controller LR | 0.1 |
| Controller Momentum | 0.9 |
| Discount Factor | 0.5 |
| Monitor Test | ✓ |
| Train with Dev | ✗ |

### Dataset (InLNER — Indian Legal NER)
| Split | Προτάσεις |
|-------|-----------|
| Train | 10,725 |
| Dev | 1,070 |
| Test | 4,602 |

---

## 3. Embeddings (6 Υποψήφια)

| # | Τύπος | Embedding | Διάσταση |
|---|-------|-----------|----------|
| 0 | Contextual | **mDeBERTa-v3-base** (fine-tuned σε Greek Legal NER, checkpoint-9387) | 768 |
| 1 | Contextual | **XLM-R-Base** (fine-tuned σε Greek Legal NER) | 768 |
| 2 | Contextual | **DistilBERT-multilingual** (fine-tuned σε Greek Legal NER) | 768 |
| 3 | Static | **GloVe** (`glove-wiki-gigaword-300`) | 300 |
| 4 | Static | **FastText** (`fasttext-wiki-news-subwords-300`) | 300 |
| 5 | Static | **Word2Vec** (`word2vec-google-news-300`) | 300 |

**Σύνολο (αν επιλεγούν όλα):** 3,204 διαστάσεις

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | Επιλεγμένα Embeddings | Mask |
|:-------:|:------:|:------------:|----------------------|:----:|
| 1 | — | 86.91 | mDeBERTa, XLM-R, DistilBERT, GloVe, FastText, Word2Vec | [1,1,1,1,1,1] |
| 2 | — | 85.91 | mDeBERTa, DistilBERT, Word2Vec | [1,0,1,0,0,1] |
| 3 | — | 84.13 | DistilBERT, GloVe | [0,0,1,1,0,0] |
| 4 | — | 85.50 | mDeBERTa, DistilBERT, GloVe | [1,0,1,1,0,0] |
| 5 | — | 87.01 | mDeBERTa, DistilBERT, FastText | [1,0,1,0,1,0] |
| 6 | — | 87.44 | XLM-R, DistilBERT, GloVe, Word2Vec | [0,1,1,1,0,1] |
| 7 | — | 87.28 | mDeBERTa, XLM-R, DistilBERT, GloVe, Word2Vec | [1,1,1,1,0,1] |
| 8 | — | 87.41 | GloVe, FastText, Word2Vec | [0,0,0,1,1,1] |
| 9 | — | 86.80 | XLM-R, FastText, Word2Vec | [0,1,0,0,1,1] |
| 10 | — | 86.58 | XLM-R, FastText, Word2Vec | [0,1,0,0,1,1] |
| 11 | — | 87.54 | mDeBERTa, XLM-R, FastText, Word2Vec | [1,1,0,0,1,1] |
| 12 | — | 86.87 | XLM-R, GloVe, FastText, Word2Vec | [0,1,0,1,1,1] |
| 13 | — | 86.98 | XLM-R, GloVe, FastText, Word2Vec | [0,1,0,1,1,1] |
| 14 | — | 83.60 | XLM-R, GloVe | [0,1,0,1,0,0] |
| **15** ★ | — | **87.74** | XLM-R, DistilBERT, GloVe, FastText, Word2Vec | [0,1,1,1,1,1] |
| 16 | — | 87.25 | mDeBERTa, XLM-R, DistilBERT, FastText, Word2Vec | [1,1,1,0,1,1] |
| 17 | — | 87.22 | mDeBERTa, XLM-R, DistilBERT, FastText, Word2Vec | [1,1,1,0,1,1] |
| 18 | — | 87.14 | mDeBERTa, XLM-R, GloVe, FastText, Word2Vec | [1,1,0,1,1,1] |
| 19 | — | 87.15 | mDeBERTa, DistilBERT, GloVe, FastText, Word2Vec | [1,0,1,1,1,1] |
| 20 | — | 87.44 | Word2Vec _(μόνο)_ | [0,0,0,0,0,1] |
| 21 | — | 87.49 | mDeBERTa, XLM-R, DistilBERT, FastText, Word2Vec | [1,1,1,0,1,1] |
| 22 | — | 87.67 | DistilBERT, GloVe, Word2Vec | [0,0,1,1,0,1] |
| 23 | — | 86.94 | mDeBERTa, XLM-R, DistilBERT, FastText | [1,1,1,0,1,0] |
| 24 | 42 | 87.44 | mDeBERTa, XLM-R, DistilBERT, FastText, Word2Vec | [1,1,1,0,1,1] |
| 25 | 16 | 86.94 | mDeBERTa, XLM-R, DistilBERT, FastText, Word2Vec | [1,1,1,0,1,1] _(εκτίμηση)_ |
| 26 | 50 | 80.17 | GloVe _(μόνο)_ | [0,0,0,1,0,0] |
| 27 | 26 | 87.67 | mDeBERTa, XLM-R, DistilBERT, FastText, Word2Vec | [1,1,1,0,1,1] |
| 28 | 21 | 86.57 | mDeBERTa, XLM-R, DistilBERT, GloVe, FastText | [1,1,1,1,1,0] |
| **29** ★★ | 16 | **87.82** | DistilBERT, FastText, Word2Vec | [0,0,1,0,1,1] |
| 30 | 17 | 87.01 | mDeBERTa, GloVe, Word2Vec | [1,0,0,1,0,1] |

★ = Πρώτο νέο baseline (Episode 15)  
★★ = Τελικό best model (Episode 29)


### Baseline Transitions
| Γεγονός | Episode | Test F1 | Baseline Action |
|---------|:-------:|:-------:|-----------------|
| Αρχικό baseline | 1 | 86.91 | [1,1,1,1,1,1] — Όλα τα embeddings |
| Νέο baseline | 15 | 87.74 | [0,1,1,1,1,1] — XLM-R, DistilBERT, GloVe, FastText, Word2Vec |
| Νέο baseline | 29 | 87.82 | [0,0,1,0,1,1] — DistilBERT, FastText, Word2Vec |

---

## 5. Ανάλυση RL Controller

### Συχνότητα Επιλογής Embedding (30 episodes)

| Embedding | Επιλέχθηκε | Ποσοστό | Σημασία |
|-----------|:----------:|:-------:|---------|
| mDeBERTa-v3 | 16/30 | 53% | Μέτρια 
| XLM-R-Base | 17/30 | 57% | Μέτρια 
| DistilBERT-multilingual | 20/30 | 67% | **Υψηλή** 
| GloVe | 15/30 | 50% | Μέτρια 
| FastText | 18/30 | 60% | **Υψηλή** 
| Word2Vec | 20/30 | 67% | **Υψηλή**

---

## 6. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Embeddings στο Best Model
- ❌ mDeBERTa-v3 — _not selected, skipped_
- ❌ XLM-R-Base — _not selected, skipped_
- ✅ **DistilBERT-multilingual** (134.7M params)
- ❌ GloVe — _not selected, skipped_
- ✅ **FastText** (wiki-news-300d)
- ✅ **Word2Vec** (google-news-300d)

### Αποτελέσματα ανά Οντότητα

| Οντότητα | TP | FP | FN | Precision | Recall | F1 |
|----------|:--:|:--:|:--:|:---------:|:------:|:--:|
| **PROVISION** | 1,152 | 63 | 70 | 94.81% | 94.27% | **94.54%** |
| **DATE** | 1,050 | 71 | 62 | 93.67% | 94.42% | **94.04%** |
| **STATUTE** | 910 | 55 | 63 | 94.30% | 93.53% | **93.91%** |
| **COURT** | 715 | 46 | 84 | 93.96% | 89.49% | **91.67%** |
| **WITNESS** | 382 | 38 | 32 | 90.95% | 92.27% | **91.61%** |
| **OTHER_PERSON** | 985 | 93 | 126 | 91.37% | 88.66% | **89.99%** |
| **JUDGE** | 41 | 7 | 1 | 85.42% | 97.62% | **91.11%** |
| **GPE** | 580 | 147 | 139 | 79.78% | 80.67% | **80.22%** |
| **ORG** | 716 | 203 | 198 | 77.91% | 78.34% | **78.12%** |
| **CASE_NUMBER** | 563 | 118 | 105 | 82.67% | 84.28% | **83.47%** |
| **PRECEDENT** | 500 | 169 | 159 | 74.74% | 75.87% | **75.30%** |
| **PETITIONER** | 41 | 7 | 23 | 85.42% | 64.06% | **73.21%** |
| **RESPONDENT** | 25 | 17 | 8 | 59.52% | 75.76% | **66.67%** |

### Συνολικά

| Μετρική | Precision | Recall | F1 |
|---------|:---------:|:------:|:--:|
| **MICRO AVG** | 88.11% | 87.74% | **87.92%** |
| **MACRO AVG** | — | — | **84.91%** |
