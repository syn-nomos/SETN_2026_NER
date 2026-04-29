# Grouped-ACE (N=4) — dataset: Seed 44

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **77.89%** |
| **Macro F1 (Test)**  | **71.86%** |
| Precision (Micro)    | 82.98%     |
| Recall (Micro)       | 73.38%     |
| Accuracy (Micro)     | 63.78%     |

**Μέθοδος:** Grouped-ACE (N=4) (N=4 groups / embedding, equal mode, seed 44)  
**Best Model Save:** Episode 11, Dev F1 = 81.63% (Test F1 = 77.47%)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 77.89%  
**Διάρκεια Εκπαίδευσης:** ~9h 33m (35 episodes)

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
| group_seed     | 44  |

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
| Seed                           | 44 |
| Monitor Test                   | ✗ |
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
| Train | 7,552    |
| Dev   | 966     |
| Test  | 907     |

---

## 3. Embeddings (4 Υποψήφια × 4 Groups = 16 Actions)

| #   | Embedding (short)    | Raw Path / ID                          |
|-----|----------------------|----------------------------------------|
| 0   | **LegalRoBERTa** | `AIAI/LegalNERO/logs/LegalRomanianRoBERTa/greek_legal_ner/joelito/legal-romanian-roberta-base/results_after_training_only_on_language_with_id__ro/seed_1/checkpoint-2832` |
| 1   | **RomanianBERT** | `AIAI/LegalNERO/logs/RomanianBERT/greek_legal_ner/dumitrescustefan/bert-base-romanian-cased-v1/results_after_training_only_on_language_with_id__all/seed_1/checkpoint-2832` |
| 2   | **mDeBERTa-v3** | `AIAI/LegalNERO/logs/mDeBERTa/greek_legal_ner/microsoft/mdeberta-v3-base/seed_1/checkpoint-2832` |
| 3   | **FastCharEmbeddings** | `Char` |

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | LegalRoBERTa | RomanianBERT | mDeBERTa-v3 | FastCharEmbeddings | Kept  |
|:-------:|:------:|:------------:| :-------: | :-------: | :-------: | :-------: |:-----:|
|    1    |   36   |   78.25 ★    |    4/4    |    4/4    |    4/4    |    4/4    | 16/16 |
|    2    |   11   |     0.00     |    2/4    |    2/4    |    4/4    |    2/4    | 10/16 |
|    3    |   15   |   76.72 ★    |    1/4    |    1/4    |    2/4    |    2/4    | 6/16  |
|    4    |   20   |   77.56 ★    |    3/4    |    2/4    |    2/4    |    3/4    | 10/16 |
|    5    |   20   |     0.00     |    3/4    |    0/4    |    1/4    |    4/4    | 8/16  |
|    6    |   59   |   77.86 ★    |    1/4    |    3/4    |    1/4    |    4/4    | 9/16  |
|    7    |   18   |   79.46 ★    |    3/4    |    3/4    |    2/4    |    2/4    | 10/16 |
|    8    |   27   |   77.70 ★    |    2/4    |    2/4    |    2/4    |    3/4    | 9/16  |
|    9    |   34   |     0.00     |    1/4    |    3/4    |    2/4    |    3/4    | 9/16  |
|   10    |   17   |   78.28 ★    |    1/4    |    2/4    |    2/4    |    2/4    | 7/16  |
| **11**  | **37** | **77.47 ★**  |    2/4    |    2/4    |    2/4    |    3/4    | **9/16** |
|   12    |   26   |     0.00     |    1/4    |    2/4    |    2/4    |    1/4    | 6/16  |
|   13    |   11   |     0.00     |    2/4    |    2/4    |    1/4    |    1/4    | 6/16  |
|   14    |   52   |     0.00     |    2/4    |    2/4    |    3/4    |    2/4    | 9/16  |
|   15    |   11   |     0.00     |    2/4    |    3/4    |    2/4    |    2/4    | 9/16  |
|   16    |   12   |     0.00     |    1/4    |    2/4    |    2/4    |    3/4    | 8/16  |
|   17    |   22   |     0.00     |    1/4    |    4/4    |    1/4    |    3/4    | 9/16  |
|   18    |   21   |     0.00     |    2/4    |    2/4    |    2/4    |    3/4    | 9/16  |
|   19    |   19   |     0.00     |    1/4    |    2/4    |    1/4    |    2/4    | 6/16  |
|   20    |   28   |     0.00     |    2/4    |    2/4    |    1/4    |    3/4    | 8/16  |
|   21    |   14   |     0.00     |    3/4    |    3/4    |    2/4    |    3/4    | 11/16 |
|   22    |   15   |     0.00     |    1/4    |    3/4    |    1/4    |    1/4    | 6/16  |
|   23    |   14   |     0.00     |    2/4    |    2/4    |    3/4    |    2/4    | 9/16  |
|   24    |   15   |     0.00     |    1/4    |    2/4    |    2/4    |    2/4    | 7/16  |
|   25    |   24   |     0.00     |    1/4    |    3/4    |    2/4    |    2/4    | 8/16  |
|   26    |   21   |     0.00     |    1/4    |    1/4    |    3/4    |    2/4    | 7/16  |
|   27    |   30   |     0.00     |    1/4    |    2/4    |    2/4    |    2/4    | 7/16  |
|   28    |   12   |     0.00     |    1/4    |    2/4    |    1/4    |    2/4    | 6/16  |
|   29    |   21   |     0.00     |    1/4    |    1/4    |    2/4    |    2/4    | 6/16  |
|   30    |   20   |     0.00     |    1/4    |    2/4    |    2/4    |    2/4    | 7/16  |
|   31    |   11   |     0.00     |    1/4    |    2/4    |    2/4    |    3/4    | 8/16  |
|   32    |   23   |     0.00     |    1/4    |    2/4    |    2/4    |    2/4    | 7/16  |
|   33    |   18   |     0.00     |    2/4    |    2/4    |    2/4    |    2/4    | 8/16  |
|   34    |   14   |     0.00     |    1/4    |    2/4    |    2/4    |    2/4    | 7/16  |
|   35    |   13   |     0.00     |    2/4    |    2/4    |    2/4    |    2/4    | 8/16  |

**Σύνολο:** 35 episodes, ~761 epochs, ~9h 33m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Dev F1  | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 79.86 | 16/16 | LegalRoBERTa 4/4, RomanianBERT 4/4, mDeBERTa-v3 4/4, FastCharEmbeddings 4/4 |
| Νέο baseline | 3 | 79.95 | 6/16 | LegalRoBERTa 1/4, RomanianBERT 1/4, mDeBERTa-v3 2/4, FastCharEmbeddings 2/4 |
| Νέο baseline | 4 | 80.62 | 10/16 | LegalRoBERTa 3/4, RomanianBERT 2/4, mDeBERTa-v3 2/4, FastCharEmbeddings 3/4 |
| Νέο baseline | 6 | 80.76 | 9/16 | LegalRoBERTa 1/4, RomanianBERT 3/4, mDeBERTa-v3 1/4, FastCharEmbeddings 4/4 |
| Νέο baseline | 7 | 81.00 | 10/16 | LegalRoBERTa 3/4, RomanianBERT 3/4, mDeBERTa-v3 2/4, FastCharEmbeddings 2/4 |
| Νέο baseline | 8 | 81.19 | 9/16 | LegalRoBERTa 2/4, RomanianBERT 2/4, mDeBERTa-v3 2/4, FastCharEmbeddings 3/4 |
| Νέο baseline | 10 | 81.57 | 7/16 | LegalRoBERTa 1/4, RomanianBERT 2/4, mDeBERTa-v3 2/4, FastCharEmbeddings 2/4 |
| **Τελικό** ★ | 11 | 81.63 | 9/16 | LegalRoBERTa 2/4, RomanianBERT 2/4, mDeBERTa-v3 2/4, FastCharEmbeddings 3/4 |

---

## 5. Best Model Action (Episode 11)

| Embedding          | Groups     | Kept | Σημείωση                    |
|--------------------|------------|:----:|-----------------------------|
| LegalRoBERTa       | [0, 1, 0, 1] | 2/4  | Μερική χρήση                |
| RomanianBERT       | [1, 1, 0, 0] | 2/4  | Μερική χρήση                |
| mDeBERTa-v3        | [1, 0, 1, 0] | 2/4  | Μερική χρήση                |
| FastCharEmbeddings | [0, 1, 1, 1] | 3/4  | **Κυρίαρχο**                |
| **Σύνολο**         |            | **9/16 (56.2%)** | |

---

## 6. Συχνότητα Επιλογής Embedding

| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |
|--------------------|:----------------------:|:-------:|
| LegalRoBERTa       | 35/35                     | 100.0%   |
| mDeBERTa-v3        | 35/35                     | 100.0%   |
| FastCharEmbeddings | 35/35                     | 100.0%   |
| RomanianBERT       | 34/35                     | 97.1%   |

---

## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Αποτελέσματα ανά Οντότητα

| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |
|-------------|-----:|----:|----:|:---------:|:------:|:----------:|
| PER         |    4 |   0 |   1 |   100.00% | 80.00% | **88.89%** |
| TIME        |  109 |  20 |  23 |    84.50% | 82.58% | **83.53%** |
| ORG         |  122 |  22 |  40 |    84.72% | 75.31% | **79.74%** |
| LEGAL       |   76 |  15 |  32 |    83.52% | 70.37% | **76.38%** |
| LOC         |    6 |   8 |  19 |    42.86% | 24.00% | **30.77%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    82.98% | 73.38% | **77.89%** |
| **MACRO AVG**  |     —     |   —    | **71.86%** |
