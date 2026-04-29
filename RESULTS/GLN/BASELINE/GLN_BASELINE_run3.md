# ACE — GREEKLEGALNERV2: Seed ?

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **71.44%** |
| **Macro F1 (Test)**  | **68.40%** |
| Precision (Micro)    | 78.16%     |
| Recall (Micro)       | 65.79%     |
| Accuracy (Micro)     | 55.57%     |

**Μέθοδος:** ACE (N=1 groups / embedding, equal mode, seed ?)  
**Best Model Save:** Episode 1, Dev F1 = 71.44% (Test F1 = 71.44%)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 71.44%  
**Διάρκεια Εκπαίδευσης:** ~0h 48m (1 episodes)

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
| 0   | **GreekBERT** | `DATASETS/GREEKLEGALNERV2/GreekBERT/greek_legal_ner/nlpaueb/bert-base-greek-uncased-v1/results_after_training_only_on_language_with_id__el/seed_1` |
| 1   | **GreekLegalRoBERTa** | `DATASETS/GREEKLEGALNERV2/GreekLegalRoBERTa/greek_legal_ner/joelito/legal-greek-roberta-base/results_after_training_only_on_language_with_id__el/seed_1` |
| 2   | **cc.el.300.vec** | `DATASETS/GREEKLEGALNERV2/cc.el.300.vec` |
| 3   | **mDeBERTa-v3** | `DATASETS/GREEKLEGALNERV2/mdeberta-v3-base/seed_1` |
| 4   | **FastCharEmbeddings** | `Char` |
| 5   | **BPEmb** | `bpe-el-100000-300` |

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | GreekBERT | GreekLegalRoBERTa | cc.el.300.vec | mDeBERTa-v3 | FastCharEmbeddings |   BPEmb   | Kept  |
|:-------:|:------:|:------------:| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |:-----:|
|  **1**  | **20** | **71.44 ★**  |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    | **6/6** |

**Σύνολο:** 1 episodes, ~20 epochs, ~0h 48m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Dev F1  | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 71.44 | 6/6 | GreekBERT 1/1, GreekLegalRoBERTa 1/1, cc.el.300.vec 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |

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
| PERSON      |  478 |  19 |  38 |    96.18% | 92.64% | **94.38%** |
| LEG-REFS    | 1001 | 132 | 310 |    88.35% | 76.35% | **81.91%** |
| DATE        |  355 |  47 | 198 |    88.31% | 64.20% | **74.35%** |
| GPE         |  561 | 140 | 267 |    80.03% | 67.75% | **73.38%** |
| LOCATION    |  553 | 252 | 154 |    68.70% | 78.22% | **73.15%** |
| ORG         |  997 | 417 | 777 |    70.51% | 56.20% | **62.55%** |
| PUBLIC-DOCS |  353 | 167 | 443 |    67.88% | 44.35% | **53.65%** |
| FACILITY    |   24 |  34 |  60 |    41.38% | 28.57% | **33.80%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    78.16% | 65.79% | **71.44%** |
| **MACRO AVG**  |     —     |   —    | **68.40%** |
