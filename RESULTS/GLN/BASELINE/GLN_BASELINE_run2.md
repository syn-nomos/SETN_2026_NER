# ACE — GREEKLEGALNERV2: Seed ?

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **71.20%** |
| **Macro F1 (Test)**  | **67.69%** |
| Precision (Micro)    | 76.71%     |
| Recall (Micro)       | 66.43%     |
| Accuracy (Micro)     | 55.28%     |

**Μέθοδος:** ACE (N=1 groups / embedding, equal mode, seed ?)  
**Best Model Save:** Episode 1, Dev F1 = 71.20% (Test F1 = 71.20%)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 71.20%  
**Διάρκεια Εκπαίδευσης:** ~0h 56m (1 episodes)

---

## 2. Ρυθμίσεις Εκπαίδευσης (Configuration)

### Αρχιτεκτονική Μοντέλου

| Παράμετρος           | Τιμή                |
|----------------------|---------------------|
| Model                | FastSequenceTagger   |
| Hidden Size          | 256                 |
| Use Crf              | ✓                   |
| Use Rnn              | ✓                   |
| Rnn Layers           | 1                   |
| Dropout              | 0.0                 |
| Word Dropout         | 0.05                |
| Locked Dropout       | 0.5                 |
| Sentence Loss        | ✓                   |

### Grouped-ACE Parameters

| Παράμετρος     | Τιμή   |
|----------------|--------|
| model_structure | None  |

### Hyperparameters Εκπαίδευσης

| Παράμετρος                     | Τιμή  |
|--------------------------------|-------|
| Controller Learning Rate       | 0.1 |
| Controller Optimizer           | SGD |
| Distill Mode                   | ✗ |
| Optimizer                      | SGD |
| Sentence Level Batch           | ✓ |
| Max Episodes                   | 1 |
| Max Epochs                     | 150 |
| Monitor Test                   | ✓ |
| Continue Training              | ✗ |
| Mini Batch Size                | 8 |
| Learning Rate                  | 0.1 |
| Min Learning Rate              | 0.0001 |
| Max Epochs Without Improvement | 15 |
| Patience                       | 5 |
| Save Final Model               | ✓ |
| Train With Dev                 | ✗ |
| True Reshuffle                 | ✗ |
| Controller Momentum            | 0.9 |
| Discount                       | 0.5 |

### Dataset

| Split | Προτάσεις |
|-------|-----------|
| Train | 17,371    |
| Dev   | 4,752     |
| Test  | 3,879     |

---

## 3. Embeddings (6 Υποψήφια × 1 Groups = 6 Actions)

| #   | Embedding (short)    | Raw Path / ID                          |
|-----|----------------------|----------------------------------------|
| 0   | **GreekBERT** | `AIAI/GREEKLEGALNERV2/GreekBERT/greek_legal_ner/nlpaueb/bert-base-greek-uncased-v1/results_after_training_only_on_language_with_id__el/seed_1` |
| 1   | **GreekLegalRoBERTa** | `AIAI/GREEKLEGALNERV2/GreekLegalRoBERTa/greek_legal_ner/joelito/legal-greek-roberta-base/results_after_training_only_on_language_with_id__el/seed_1` |
| 2   | **cc.el.300.vec** | `AIAI/GREEKLEGALNERV2/cc.el.300.vec` |
| 3   | **mDeBERTa-v3** | `AIAI/GREEKLEGALNERV2/mdeberta-v3-base/seed_1` |
| 4   | **FastCharEmbeddings** | `Char` |
| 5   | **BPEmb** | `bpe-el-100000-300` |

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | GreekBERT | GreekLegalRoBERTa | cc.el.300.vec | mDeBERTa-v3 | FastCharEmbeddings |   BPEmb   | Kept  |
|:-------:|:------:|:------------:| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |:-----:|
|  **1**  | **24** | **71.20 ★**  |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    | **6/6** |

**Σύνολο:** 1 episodes, ~24 epochs, ~0h 56m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Dev F1  | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 71.20 | 6/6 | GreekBERT 1/1, GreekLegalRoBERTa 1/1, cc.el.300.vec 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |

---

## 5. Best Model Action (Episode 1)

| Embedding          | Groups     | Kept | Σημείωση                    |
|--------------------|------------|:----:|-----------------------------|
| GreekBERT          | [1]        | 1/1  | Πλήρης χρήση                |
| GreekLegalRoBERTa  | [1]        | 1/1  | Πλήρης χρήση                |
| cc.el.300.vec      | [1]        | 1/1  | Πλήρης χρήση                |
| mDeBERTa-v3        | [1]        | 1/1  | Πλήρης χρήση                |
| FastCharEmbeddings | [1]        | 1/1  | Πλήρης χρήση                |
| BPEmb              | [1]        | 1/1  | Πλήρης χρήση                |
| **Σύνολο**         |            | **6/6 (100.0%)** | |

---

## 6. Συχνότητα Επιλογής Embedding

| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |
|--------------------|:----------------------:|:-------:|
| GreekBERT          | 1/1                     | 100.0%   |
| GreekLegalRoBERTa  | 1/1                     | 100.0%   |
| cc.el.300.vec      | 1/1                     | 100.0%   |
| mDeBERTa-v3        | 1/1                     | 100.0%   |
| FastCharEmbeddings | 1/1                     | 100.0%   |
| BPEmb              | 1/1                     | 100.0%   |

---

## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Αποτελέσματα ανά Οντότητα

| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |
|-------------|-----:|----:|----:|:---------:|:------:|:----------:|
| PERSON      |  486 |  21 |  30 |    95.86% | 94.19% | **95.02%** |
| LEG-REFS    |  903 | 117 | 408 |    88.53% | 68.88% | **77.48%** |
| DATE        |  379 |  67 | 174 |    84.98% | 68.54% | **75.88%** |
| LOCATION    |  537 | 223 | 170 |    70.66% | 75.95% | **73.21%** |
| GPE         |  559 | 157 | 269 |    78.07% | 67.51% | **72.41%** |
| ORG         | 1101 | 488 | 673 |    69.29% | 62.06% | **65.48%** |
| PUBLIC-DOCS |  381 | 224 | 415 |    62.98% | 47.86% | **54.39%** |
| FACILITY    |   18 |  28 |  66 |    39.13% | 21.43% | **27.69%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    76.71% | 66.43% | **71.20%** |
| **MACRO AVG**  |     —     |   —    | **67.69%** |
