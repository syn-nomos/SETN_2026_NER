# Grouped-ACE (N=8) — GREEKLEGALNERV2: Seed 44

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **72.31%** |
| **Macro F1 (Test)**  | **69.27%** |
| Precision (Micro)    | 79.01%     |
| Recall (Micro)       | 66.65%     |
| Accuracy (Micro)     | 56.62%     |

**Μέθοδος:** Grouped-ACE (N=8) (N=8 groups / embedding, equal mode, seed 44)  
**Best Model Save:** Episode 0, Dev F1 = 0.00% (Test F1 = 0.00%)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 72.31%  
**Διάρκεια Εκπαίδευσης:** ~16h 04m (17 episodes)

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
| num_groups     | 8  |
| group_mode     | equal  |
| group_seed     | 44  |

### Hyperparameters Εκπαίδευσης

| Παράμετρος                     | Τιμή  |
|--------------------------------|-------|
| Controller Learning Rate       | 0.6 |
| Controller Optimizer           | Adam |
| Distill Mode                   | ✗ |
| Optimizer                      | SGD |
| Sentence Level Batch           | ✓ |
| Max Episodes                   | 40 |
| Max Epochs                     | 150 |
| Seed                           | 44 |
| Monitor Test                   | ✓ |
| Mini Batch Size                | 8 |
| Learning Rate                  | 0.1 |
| Min Learning Rate              | 0.0001 |
| Max Epochs Without Improvement | 10 |
| Patience                       | 5 |
| Save Final Model               | ✓ |
| Continue Training              | ✓ |
| Train With Dev                 | ✗ |
| True Reshuffle                 | ✗ |
| Discount                       | 0.5 |

### Dataset

| Split | Προτάσεις |
|-------|-----------|
| Train | 17,371    |
| Dev   | 4,752     |
| Test  | 3,879     |

---

## 3. Embeddings (6 Υποψήφια × 8 Groups = 48 Actions)

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
|   24    |   18   |     0.00     |    2/8    |    3/8    |    0/8    |    5/8    |    2/8    |    1/8    | 13/48 |
|   25    |   12   |     0.00     |    3/8    |    4/8    |    2/8    |    4/8    |    2/8    |    2/8    | 17/48 |
|   26    |   12   |     0.00     |    2/8    |    3/8    |    1/8    |    5/8    |    2/8    |    3/8    | 16/48 |
|   27    |   12   |     0.00     |    3/8    |    2/8    |    1/8    |    5/8    |    1/8    |    2/8    | 14/48 |
|   28    |   15   |     0.00     |    4/8    |    4/8    |    0/8    |    5/8    |    2/8    |    3/8    | 18/48 |
|   29    |   24   |     0.00     |    3/8    |    4/8    |    1/8    |    5/8    |    4/8    |    2/8    | 19/48 |
|   30    |   16   |     0.00     |    2/8    |    2/8    |    1/8    |    4/8    |    4/8    |    2/8    | 15/48 |
|   31    |   19   |     0.00     |    2/8    |    1/8    |    3/8    |    5/8    |    1/8    |    2/8    | 14/48 |
|   32    |   12   |     0.00     |    2/8    |    3/8    |    2/8    |    4/8    |    2/8    |    3/8    | 16/48 |
|   33    |   12   |     0.00     |    3/8    |    2/8    |    1/8    |    4/8    |    2/8    |    3/8    | 15/48 |
|   34    |   13   |     0.00     |    4/8    |    3/8    |    1/8    |    3/8    |    3/8    |    4/8    | 18/48 |
|   35    |   11   |     0.00     |    2/8    |    3/8    |    0/8    |    6/8    |    2/8    |    3/8    | 16/48 |
|   36    |   20   |     0.00     |    3/8    |    2/8    |    1/8    |    5/8    |    1/8    |    4/8    | 16/48 |
|   37    |   20   |     0.00     |    3/8    |    5/8    |    1/8    |    4/8    |    3/8    |    3/8    | 19/48 |
|   38    |   14   |     0.00     |    2/8    |    2/8    |    0/8    |    6/8    |    3/8    |    4/8    | 17/48 |
|   39    |   12   |     0.00     |    2/8    |    4/8    |    0/8    |    6/8    |    2/8    |    1/8    | 15/48 |
|   40    |   20   |     0.00     |    4/8    |    3/8    |    1/8    |    5/8    |    4/8    |    3/8    | 20/48 |

**Σύνολο:** 17 episodes, ~262 epochs, ~16h 04m εκπαίδευσης

---

## 5. Best Model Action (Episode 0)

| Embedding          | Groups     | Kept | Σημείωση                    |
|--------------------|------------|:----:|-----------------------------|
| GreekBERT          | [0, 0, 0, 0, 0, 1, 0, 0] | 1/8  | Μερική χρήση                |
| GreekLegalRoBERTa  | [0, 1, 0, 0, 0, 0, 0, 0] | 1/8  | Μερική χρήση                |
| cc.el.300.vec      | [0, 0, 0, 0, 0, 1, 0, 1] | 2/8  | Μερική χρήση                |
| mDeBERTa-v3        | [1, 0, 1, 1, 1, 1, 1, 1] | 7/8  | **Κυρίαρχο**                |
| FastCharEmbeddings | [1, 0, 0, 0, 0, 0, 0, 0] | 1/8  | Μερική χρήση                |
| BPEmb              | [1, 0, 0, 0, 0, 1, 0, 1] | 3/8  | Μερική χρήση                |
| **Σύνολο**         |            | **15/48 (31.2%)** | |

---

## 6. Συχνότητα Επιλογής Embedding

| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |
|--------------------|:----------------------:|:-------:|
| GreekBERT          | 17/17                     | 100.0%   |
| GreekLegalRoBERTa  | 17/17                     | 100.0%   |
| mDeBERTa-v3        | 17/17                     | 100.0%   |
| FastCharEmbeddings | 17/17                     | 100.0%   |
| BPEmb              | 17/17                     | 100.0%   |
| cc.el.300.vec      | 12/17                     | 70.6%   |

---

## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Αποτελέσματα ανά Οντότητα

| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |
|-------------|-----:|----:|----:|:---------:|:------:|:----------:|
| PERSON      |  485 |  20 |  31 |    96.04% | 93.99% | **95.00%** |
| LEG-REFS    |  965 |  97 | 346 |    90.87% | 73.61% | **81.33%** |
| DATE        |  384 |  74 | 169 |    83.84% | 69.44% | **75.96%** |
| LOCATION    |  565 | 243 | 142 |    69.93% | 79.92% | **74.59%** |
| GPE         |  516 |  92 | 312 |    84.87% | 62.32% | **71.87%** |
| ORG         | 1066 | 421 | 708 |    71.69% | 60.09% | **65.38%** |
| PUBLIC-DOCS |  373 | 187 | 423 |    66.61% | 46.86% | **55.02%** |
| FACILITY    |   24 |  29 |  60 |    45.28% | 28.57% | **35.03%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    79.01% | 66.65% | **72.31%** |
| **MACRO AVG**  |     —     |   —    | **69.27%** |
