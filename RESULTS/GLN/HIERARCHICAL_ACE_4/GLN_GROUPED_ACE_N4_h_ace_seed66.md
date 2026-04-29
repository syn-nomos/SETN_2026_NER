# Grouped-ACE (N=4) — GREEKLEGALNERV2: Seed 66

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **72.14%** |
| **Macro F1 (Test)**  | **67.75%** |
| Precision (Micro)    | 77.20%     |
| Recall (Micro)       | 67.71%     |
| Accuracy (Micro)     | 56.43%     |

**Μέθοδος:** Grouped-ACE (N=4) (N=4 groups / embedding, equal mode, seed 66)  
**Best Model Save:** Episode 30, Dev F1 = 72.14% (Test F1 = 72.14%)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 72.14%  
**Διάρκεια Εκπαίδευσης:** ~21h 37m (35 episodes)

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
| num_groups     | 4  |
| group_mode     | equal  |
| group_seed     | 66  |

### Hyperparameters Εκπαίδευσης

| Παράμετρος                     | Τιμή  |
|--------------------------------|-------|
| Controller Learning Rate       | 0.4 |
| Controller Optimizer           | Adam |
| Distill Mode                   | ✗ |
| Optimizer                      | SGD |
| Sentence Level Batch           | ✓ |
| Max Episodes                   | 35 |
| Max Epochs                     | 150 |
| Seed                           | 66 |
| Monitor Test                   | ✓ |
| Mini Batch Size                | 8 |
| Learning Rate                  | 0.1 |
| Min Learning Rate              | 0.0001 |
| Max Epochs Without Improvement | 10 |
| Patience                       | 5 |
| Save Final Model               | ✓ |
| Continue Training              | ✗ |
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

## 3. Embeddings (4 Υποψήφια × 4 Groups = 16 Actions)

| #   | Embedding (short)    | Raw Path / ID                          |
|-----|----------------------|----------------------------------------|
| 0   | **GreekBERT** | `DATASETS/GREEKLEGALNERV2/GreekBERT/greek_legal_ner/nlpaueb/bert-base-greek-uncased-v1/results_after_training_only_on_language_with_id__el/seed_1` |
| 1   | **GreekLegalRoBERTa** | `DATASETS/GREEKLEGALNERV2/GreekLegalRoBERTa/greek_legal_ner/joelito/legal-greek-roberta-base/results_after_training_only_on_language_with_id__el/seed_1` |
| 2   | **mDeBERTa-v3** | `DATASETS/GREEKLEGALNERV2/mdeberta-v3-base/seed_1` |
| 3   | **FastCharEmbeddings** | `Char` |

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | GreekBERT | GreekLegalRoBERTa | mDeBERTa-v3 | FastCharEmbeddings | Kept  |
|:-------:|:------:|:------------:| :-------: | :-------: | :-------: | :-------: |:-----:|
|    1    |   12   |   71.82 ★    |    4/4    |    4/4    |    4/4    |    4/4    | 16/16 |
|    2    |   13   |     0.00     |    2/4    |    1/4    |    3/4    |    2/4    | 8/16  |
|    3    |   12   |     0.00     |    4/4    |    2/4    |    1/4    |    3/4    | 10/16 |
|    4    |   13   |     0.00     |    1/4    |    3/4    |    3/4    |    4/4    | 11/16 |
|    5    |   28   |     0.00     |    2/4    |    4/4    |    2/4    |    3/4    | 11/16 |
|    6    |   11   |     0.00     |    2/4    |    4/4    |    3/4    |    1/4    | 10/16 |
|    7    |   14   |     0.00     |    2/4    |    2/4    |    2/4    |    3/4    | 9/16  |
|    8    |   20   |     0.00     |    3/4    |    3/4    |    3/4    |    2/4    | 11/16 |
|    9    |   13   |     0.00     |    1/4    |    2/4    |    3/4    |    2/4    | 8/16  |
|   10    |   21   |   71.91 ★    |    2/4    |    3/4    |    3/4    |    3/4    | 11/16 |
|   11    |   35   |     0.00     |    4/4    |    3/4    |    4/4    |    1/4    | 12/16 |
|   12    |   16   |     0.00     |    3/4    |    4/4    |    3/4    |    3/4    | 13/16 |
|   13    |   11   |     0.00     |    2/4    |    2/4    |    4/4    |    3/4    | 11/16 |
|   14    |   11   |     0.00     |    3/4    |    3/4    |    4/4    |    0/4    | 10/16 |
|   15    |   15   |     0.00     |    3/4    |    3/4    |    3/4    |    3/4    | 12/16 |
|   16    |   17   |     0.00     |    2/4    |    4/4    |    4/4    |    2/4    | 12/16 |
|   17    |   25   |     0.00     |    3/4    |    4/4    |    2/4    |    3/4    | 12/16 |
|   18    |   20   |     0.00     |    3/4    |    3/4    |    3/4    |    4/4    | 13/16 |
|   19    |   17   |     0.00     |    3/4    |    3/4    |    3/4    |    3/4    | 12/16 |
|   20    |   30   |     0.00     |    3/4    |    4/4    |    2/4    |    2/4    | 11/16 |
|   21    |   11   |     0.00     |    4/4    |    2/4    |    3/4    |    3/4    | 12/16 |
|   22    |   23   |     0.00     |    2/4    |    3/4    |    3/4    |    2/4    | 10/16 |
|   23    |   19   |     0.00     |    1/4    |    1/4    |    3/4    |    2/4    | 7/16  |
|   24    |   12   |     0.00     |    2/4    |    3/4    |    3/4    |    4/4    | 12/16 |
|   25    |   16   |     0.00     |    3/4    |    3/4    |    3/4    |    4/4    | 13/16 |
|   26    |   22   |     0.00     |    3/4    |    2/4    |    2/4    |    3/4    | 10/16 |
|   27    |   17   |     0.00     |    3/4    |    2/4    |    3/4    |    3/4    | 11/16 |
|   28    |   21   |     0.00     |    4/4    |    2/4    |    3/4    |    3/4    | 12/16 |
|   29    |   19   |     0.00     |    2/4    |    3/4    |    4/4    |    3/4    | 12/16 |
| **30**  | **17** | **72.14 ★**  |    3/4    |    3/4    |    4/4    |    2/4    | **12/16** |
|   31    |   27   |     0.00     |    2/4    |    2/4    |    4/4    |    3/4    | 11/16 |
|   32    |   26   |     0.00     |    2/4    |    3/4    |    3/4    |    1/4    | 9/16  |
|   33    |   18   |     0.00     |    3/4    |    3/4    |    4/4    |    2/4    | 12/16 |
|   34    |   15   |     0.00     |    2/4    |    2/4    |    3/4    |    2/4    | 9/16  |
|   35    |   16   |     0.00     |    2/4    |    3/4    |    3/4    |    2/4    | 10/16 |

**Σύνολο:** 35 episodes, ~633 epochs, ~21h 37m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Dev F1  | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 71.82 | 16/16 | GreekBERT 4/4, GreekLegalRoBERTa 4/4, mDeBERTa-v3 4/4, FastCharEmbeddings 4/4 |
| Νέο baseline | 10 | 71.91 | 11/16 | GreekBERT 2/4, GreekLegalRoBERTa 3/4, mDeBERTa-v3 3/4, FastCharEmbeddings 3/4 |
| **Τελικό** ★ | 30 | 72.14 | 12/16 | GreekBERT 3/4, GreekLegalRoBERTa 3/4, mDeBERTa-v3 4/4, FastCharEmbeddings 2/4 |

---

## 5. Best Model Action (Episode 30)

| Embedding          | Groups     | Kept | Σημείωση                    |
|--------------------|------------|:----:|-----------------------------|
| GreekBERT          | [1, 1, 0, 1] | 3/4  | **Κυρίαρχο**                |
| GreekLegalRoBERTa  | [1, 0, 1, 1] | 3/4  | **Κυρίαρχο**                |
| mDeBERTa-v3        | [1, 1, 1, 1] | 4/4  | Πλήρης χρήση                |
| FastCharEmbeddings | [0, 1, 0, 1] | 2/4  | Μερική χρήση                |
| **Σύνολο**         |            | **12/16 (75.0%)** | |

---

## 6. Συχνότητα Επιλογής Embedding

| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |
|--------------------|:----------------------:|:-------:|
| GreekBERT          | 35/35                     | 100.0%   |
| GreekLegalRoBERTa  | 35/35                     | 100.0%   |
| mDeBERTa-v3        | 35/35                     | 100.0%   |
| FastCharEmbeddings | 34/35                     | 97.1%   |

---

## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Αποτελέσματα ανά Οντότητα

| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |
|-------------|-----:|----:|----:|:---------:|:------:|:----------:|
| PERSON      |  481 |  19 |  35 |    96.20% | 93.22% | **94.69%** |
| LEG-REFS    | 1058 | 112 | 253 |    90.43% | 80.70% | **85.29%** |
| DATE        |  382 |  67 | 171 |    85.08% | 69.08% | **76.25%** |
| LOCATION    |  539 | 246 | 168 |    68.66% | 76.24% | **72.25%** |
| GPE         |  538 | 159 | 290 |    77.19% | 64.98% | **70.56%** |
| ORG         | 1056 | 457 | 718 |    69.80% | 59.53% | **64.26%** |
| PUBLIC-DOCS |  379 | 232 | 417 |    62.03% | 47.61% | **53.87%** |
| FACILITY    |   15 |  22 |  69 |    40.54% | 17.86% | **24.80%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    77.20% | 67.71% | **72.14%** |
| **MACRO AVG**  |     —     |   —    | **67.75%** |
