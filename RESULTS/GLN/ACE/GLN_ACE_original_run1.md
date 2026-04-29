# ACE — Greek Legal NER: Αποτελέσματα Εκπαίδευσης

## 1. Σύνοψη

| Μετρική | Τιμή |
|---------|------|
| **Micro F1 (Test)** | **72.56%** |
| **Macro F1 (Test)** | **68.55%** |
| Precision (Micro) | 77.10% |
| Recall (Micro) | 68.53% |
| Accuracy (Micro) | 56.94% |

**Μοντέλο: ACE 
**Best Model Save:** Episode 5, Test F1 = 72.01% (κατά την εκπαίδευση)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 72.56%

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
| Tag Scheme | BIOES (36 tags) |

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
| 2 | Contextual | **mDeBERTa-v3-base** (fine-tuned) | 768 |
| 3 | Static | **FastText** (`cc.el.300.bin`) | 300 |
| 4 | Static | **BPEmb** (el, 100K vocab, syllable-based) | 600 |
| 5 | Character | **FastCharacterEmbeddings** (CNN, 25d) | 50 |

**Σύνολο (αν επιλεγούν όλα):** 3,254 διαστάσεις

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | Επιλεγμένα Embeddings |
|:-------:|:------:|:------------:|----------------------|
| **1** | 17 | 71.89 | GreekBERT, RoBERTa, mDeBERTa, FastText, BPEmb, FastChar |
| 2 | 42 | 59.01 | FastChar _(μόνο)_ |
| 3 | 19 | 71.22 | GreekBERT, RoBERTa, BPEmb |
| 4 | 16 | 71.05 | GreekBERT, RoBERTa |
| **5** ★ | 22 | **72.01** | GreekBERT, RoBERTa, FastText, BPEmb |
| 6 | 26 | 61.00 | FastText, BPEmb _(χωρίς transformers)_ |
| 7 | 21 | 71.10 | GreekBERT, RoBERTa, BPEmb |
| 8 | 16 | 71.85 | GreekBERT, RoBERTa, FastText, BPEmb |
| 9 | 25 | 71.54 | GreekBERT, RoBERTa, mDeBERTa, BPEmb, FastChar |
| 10 | 17 | 71.20 | GreekBERT, RoBERTa, mDeBERTa |
| 11 | 16 | 70.77 | GreekBERT, RoBERTa, mDeBERTa, FastText, FastChar |
| 12 | 16 | 70.95 | GreekBERT, RoBERTa, mDeBERTa, BPEmb |
| 13 | 25 | 70.57 | GreekBERT, RoBERTa, mDeBERTa, BPEmb, FastChar |
| 14 | 18 | 71.34 | GreekBERT, RoBERTa, mDeBERTa, BPEmb |
| 15 | 23 | 71.02 | GreekBERT, RoBERTa, mDeBERTa |
| 16 | 16 | 71.00 | GreekBERT, RoBERTa, mDeBERTa, FastText, BPEmb |
| 17 | 21 | 70.96 | GreekBERT, RoBERTa, mDeBERTa, BPEmb |
| 18 | 21 | 70.96 | GreekBERT, RoBERTa, FastText, BPEmb |
| 19 | 24 | 71.20 | GreekBERT, RoBERTa, mDeBERTa, BPEmb |
| 20 | 17 | 70.61 | GreekBERT, RoBERTa, mDeBERTa, FastText, BPEmb |
| 21 | 19 | 70.50 | GreekBERT, RoBERTa, mDeBERTa, BPEmb |
| 22 | 16 | 70.63 | GreekBERT, RoBERTa, mDeBERTa, FastText, BPEmb |
| 23 | 33 | 71.00 | GreekBERT, RoBERTa, mDeBERTa, BPEmb |
| 24 | 20 | 71.07 | GreekBERT, RoBERTa, mDeBERTa, FastText, BPEmb |
| 25 | 23 | 70.69 | GreekBERT, RoBERTa, mDeBERTa, BPEmb |
| 26 | 17 | 70.68 | GreekBERT, RoBERTa, mDeBERTa, FastText, BPEmb |
| 27 | 21 | 70.77 | GreekBERT, RoBERTa, mDeBERTa, BPEmb |
| 28 | 19 | 71.31 | GreekBERT, RoBERTa, mDeBERTa, BPEmb, FastChar |
| 29 | 16 | 71.05 | GreekBERT, RoBERTa, mDeBERTa, BPEmb |
| 30 | 21 | 70.72 | GreekBERT, RoBERTa, mDeBERTa, FastText, BPEmb |


**Σύνολο:** 30 episodes, 603 epochs, ~62 ώρες εκπαίδευσης

### Baseline Transitions
| Γεγονός | Episode | Test F1 | Baseline Action |
|---------|:-------:|:-------:|-----------------|
| Αρχικό baseline | 1 | 71.89 | [1,1,1,1,1,1] — Όλα τα embeddings |
| Νέο baseline | 5 | 72.01 | [1,1,0,1,1,0] — GreekBERT, RoBERTa, FastText, BPEmb |

---

### Συχνότητα Επιλογής Embedding (30 episodes)

| Embedding | Επιλέχθηκε | Ποσοστό |
|-----------|:----------:|:-------:|
| GreekBERT | 29/30 | 97% | 
| GreekLegalRoBERTa | 29/30 | 97% |
| mDeBERTa-v3 | 22/30 | 73% |
| FastText (cc.el) | 10/30 | 33% |
| BPEmb (el) | 25/30 | 83% | 
| FastCharEmbeddings | 6/30 | 20% 


## 6. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Embeddings στο Best Model (κατά την τελική αξιολόγηση)
- ✅ GreekBERT (112.9M params)
- ✅ GreekLegalRoBERTa (110.6M params)
- ✅ mDeBERTa-v3-base (278.2M params)
- ❌ BPEmb — _not selected,

### Αποτελέσματα ανά Οντότητα

| Οντότητα | TP | FP | FN | Precision | Recall | F1 |
|----------|:--:|:--:|:--:|:---------:|:------:|:--:|
| **PERSON** | 478 | 22 | 38 | 95.60% | 92.64% | **94.10%** |
| **LEG-REFS** | 1,017 | 117 | 294 | 89.68% | 77.57% | **83.19%** |
| **DATE** | 370 | 63 | 183 | 85.45% | 66.91% | **75.05%** |
| **LOCATION** | 545 | 221 | 162 | 71.15% | 77.09% | **74.00%** |
| **GPE** | 554 | 159 | 274 | 77.70% | 66.91% | **71.90%** |
| **ORG** | 1,132 | 517 | 642 | 68.65% | 63.81% | **66.14%** |
| **PUBLIC-DOCS** | 387 | 207 | 409 | 65.15% | 48.62% | **55.68%** |
| **FACILITY** | 19 | 31 | 65 | 38.00% | 22.62% | **28.36%** |

### Συνολικά

| Μετρική | Precision | Recall | F1 |
|---------|:---------:|:------:|:--:|
| **MICRO AVG** | 77.10% | 68.53% | **72.56%** |
| **MACRO AVG** | — | — | **68.55%** |
