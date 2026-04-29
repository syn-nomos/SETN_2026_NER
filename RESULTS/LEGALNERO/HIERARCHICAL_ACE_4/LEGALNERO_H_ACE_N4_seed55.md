# Grouped-ACE (N=4) — dataset: Seed 55

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **79.36%** |
| **Macro F1 (Test)**  | **67.52%** |
| Precision (Micro)    | 84.55%     |
| Recall (Micro)       | 74.77%     |
| Accuracy (Micro)     | 65.78%     |

**Μέθοδος:** Grouped-ACE (N=4) (N=4 groups / embedding, equal mode, seed 55)  
**Best Model Save:** Episode 14, Dev F1 = 80.94% (Test F1 = 77.69%)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 79.36%  
**Διάρκεια Εκπαίδευσης:** ~7h 07m (35 episodes)

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
| group_seed     | 55  |

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
| Seed                           | 55 |
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
| 0   | **LegalRoBERTa** | `DATASETS/LegalNERO/logs/LegalRomanianRoBERTa/greek_legal_ner/joelito/legal-romanian-roberta-base/results_after_training_only_on_language_with_id__ro/seed_1/checkpoint-2832` |
| 1   | **RomanianBERT** | `DATASETS/LegalNERO/logs/RomanianBERT/greek_legal_ner/dumitrescustefan/bert-base-romanian-cased-v1/results_after_training_only_on_language_with_id__all/seed_1/checkpoint-2832` |
| 2   | **mDeBERTa-v3** | `DATASETS/LegalNERO/logs/mDeBERTa/greek_legal_ner/microsoft/mdeberta-v3-base/seed_1/checkpoint-2832` |
| 3   | **FastCharEmbeddings** | `Char` |

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | LegalRoBERTa | RomanianBERT | mDeBERTa-v3 | FastCharEmbeddings | Kept  |
|:-------:|:------:|:------------:| :-------: | :-------: | :-------: | :-------: |:-----:|
|    1    |   17   |   79.15 ★    |    4/4    |    4/4    |    4/4    |    4/4    | 16/16 |
|    2    |   21   |   77.67 ★    |    4/4    |    1/4    |    2/4    |    0/4    | 7/16  |
|    3    |   17   |   78.54 ★    |    1/4    |    2/4    |    1/4    |    2/4    | 6/16  |
|    4    |   16   |     0.00     |    1/4    |    0/4    |    3/4    |    2/4    | 6/16  |
|    5    |   25   |   76.33 ★    |    1/4    |    2/4    |    2/4    |    1/4    | 6/16  |
|    6    |   14   |   78.33 ★    |    1/4    |    3/4    |    1/4    |    2/4    | 7/16  |
|    7    |   22   |     0.00     |    1/4    |    2/4    |    2/4    |    1/4    | 6/16  |
|    8    |   13   |     0.00     |    1/4    |    2/4    |    2/4    |    1/4    | 6/16  |
|    9    |   35   |     0.00     |    0/4    |    3/4    |    1/4    |    1/4    | 5/16  |
|   10    |   21   |   78.12 ★    |    1/4    |    2/4    |    1/4    |    1/4    | 5/16  |
|   11    |   33   |     0.00     |    0/4    |    1/4    |    2/4    |    2/4    | 5/16  |
|   12    |   16   |     0.00     |    0/4    |    3/4    |    0/4    |    3/4    | 6/16  |
|   13    |   42   |     0.00     |    0/4    |    3/4    |    2/4    |    1/4    | 6/16  |
| **14**  | **16** | **77.69 ★**  |    2/4    |    3/4    |    2/4    |    2/4    | **9/16** |
|   15    |   11   |     0.00     |    1/4    |    3/4    |    2/4    |    1/4    | 7/16  |
|   16    |   14   |     0.00     |    1/4    |    1/4    |    2/4    |    0/4    | 4/16  |
|   17    |   20   |     0.00     |    1/4    |    1/4    |    1/4    |    2/4    | 5/16  |
|   18    |   19   |     0.00     |    2/4    |    3/4    |    3/4    |    0/4    | 8/16  |
|   19    |   15   |     0.00     |    1/4    |    1/4    |    1/4    |    1/4    | 4/16  |
|   20    |   16   |     0.00     |    1/4    |    1/4    |    1/4    |    0/4    | 3/16  |
|   21    |   29   |     0.00     |    2/4    |    2/4    |    2/4    |    0/4    | 6/16  |
|   22    |   14   |     0.00     |    1/4    |    3/4    |    2/4    |    1/4    | 7/16  |
|   23    |   12   |     0.00     |    2/4    |    3/4    |    1/4    |    2/4    | 8/16  |
|   24    |   13   |     0.00     |    2/4    |    1/4    |    1/4    |    1/4    | 5/16  |
|   25    |   14   |     0.00     |    2/4    |    1/4    |    2/4    |    1/4    | 6/16  |
|   26    |   15   |     0.00     |    2/4    |    2/4    |    1/4    |    1/4    | 6/16  |
|   27    |   23   |     0.00     |    2/4    |    1/4    |    2/4    |    2/4    | 7/16  |
|   28    |   22   |     0.00     |    1/4    |    1/4    |    2/4    |    1/4    | 5/16  |
|   29    |   18   |     0.00     |    2/4    |    2/4    |    2/4    |    1/4    | 7/16  |
|   30    |   24   |     0.00     |    1/4    |    1/4    |    2/4    |    0/4    | 4/16  |
|   31    |   25   |     0.00     |    2/4    |    1/4    |    1/4    |    1/4    | 5/16  |
|   32    |   17   |     0.00     |    2/4    |    1/4    |    1/4    |    2/4    | 6/16  |
|   33    |   19   |     0.00     |    2/4    |    1/4    |    2/4    |    2/4    | 7/16  |
|   34    |   21   |     0.00     |    2/4    |    1/4    |    1/4    |    0/4    | 4/16  |
|   35    |   17   |     0.00     |    2/4    |    2/4    |    1/4    |    0/4    | 5/16  |

**Σύνολο:** 35 episodes, ~686 epochs, ~7h 07m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Dev F1  | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 78.13 | 16/16 | LegalRoBERTa 4/4, RomanianBERT 4/4, mDeBERTa-v3 4/4, FastCharEmbeddings 4/4 |
| Νέο baseline | 2 | 79.05 | 7/16 | LegalRoBERTa 4/4, RomanianBERT 1/4, mDeBERTa-v3 2/4 |
| Νέο baseline | 3 | 80.09 | 6/16 | LegalRoBERTa 1/4, RomanianBERT 2/4, mDeBERTa-v3 1/4, FastCharEmbeddings 2/4 |
| Νέο baseline | 5 | 80.47 | 6/16 | LegalRoBERTa 1/4, RomanianBERT 2/4, mDeBERTa-v3 2/4, FastCharEmbeddings 1/4 |
| Νέο baseline | 6 | 80.67 | 7/16 | LegalRoBERTa 1/4, RomanianBERT 3/4, mDeBERTa-v3 1/4, FastCharEmbeddings 2/4 |
| Νέο baseline | 10 | 80.67 | 5/16 | LegalRoBERTa 1/4, RomanianBERT 2/4, mDeBERTa-v3 1/4, FastCharEmbeddings 1/4 |
| **Τελικό** ★ | 14 | 80.94 | 9/16 | LegalRoBERTa 2/4, RomanianBERT 3/4, mDeBERTa-v3 2/4, FastCharEmbeddings 2/4 |

---

## 5. Best Model Action (Episode 14)

| Embedding          | Groups     | Kept | Σημείωση                    |
|--------------------|------------|:----:|-----------------------------|
| LegalRoBERTa       | [1, 0, 0, 1] | 2/4  | Μερική χρήση                |
| RomanianBERT       | [1, 1, 0, 1] | 3/4  | **Κυρίαρχο**                |
| mDeBERTa-v3        | [0, 0, 1, 1] | 2/4  | Μερική χρήση                |
| FastCharEmbeddings | [0, 0, 1, 1] | 2/4  | Μερική χρήση                |
| **Σύνολο**         |            | **9/16 (56.2%)** | |

---

## 6. Συχνότητα Επιλογής Embedding

| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |
|--------------------|:----------------------:|:-------:|
| RomanianBERT       | 34/35                     | 97.1%   |
| mDeBERTa-v3        | 34/35                     | 97.1%   |
| LegalRoBERTa       | 31/35                     | 88.6%   |
| FastCharEmbeddings | 27/35                     | 77.1%   |

---

## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Αποτελέσματα ανά Οντότητα

| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |
|-------------|-----:|----:|----:|:---------:|:------:|:----------:|
| TIME        |  109 |  20 |  23 |    84.50% | 82.58% | **83.53%** |
| ORG         |  124 |  17 |  38 |    87.94% | 76.54% | **81.84%** |
| LEGAL       |   81 |  14 |  27 |    85.26% | 75.00% | **79.80%** |
| PER         |    3 |   2 |   2 |    60.00% | 60.00% | **60.00%** |
| LOC         |    6 |   6 |  19 |    50.00% | 24.00% | **32.43%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    84.55% | 74.77% | **79.36%** |
| **MACRO AVG**  |     —     |   —    | **67.52%** |
