# ACE — NER: Seed ?

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **79.61%** |
| **Macro F1 (Test)**  | **72.45%** |
| Precision (Micro)    | 83.67%     |
| Recall (Micro)       | 75.93%     |
| Accuracy (Micro)     | 66.13%     |

**Μέθοδος:** ACE (N=1 groups / embedding, equal mode, seed ?)  
**Best Model Save:** Episode 0, Test F1 = 0.00%  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 79.61%  
**Διάρκεια Εκπαίδευσης:** ~6h 14m (30 episodes)

---

## 2. Ρυθμίσεις Εκπαίδευσης (Configuration)

### Dataset

| Split | Προτάσεις |
|-------|-----------|
| Train | 7,552    |
| Dev   | 966     |
| Test  | 907     |

---

## 3. Embeddings (6 Υποψήφια × 1 Groups = 6 Actions)

| #   | Embedding (short)    | Raw Path / ID                          |
|-----|----------------------|----------------------------------------|
| 0   | **FastText** | `DATASETS/LegalNERO/cc.ro.300.vec` |
| 1   | **LegalRoBERTa** | `DATASETS/LegalNERO/logs/LegalRomanianRoBERTa/greek_legal_ner/joelito/legal-romanian-roberta-base/results_after_training_only_on_language_with_id__ro/seed_1/checkpoint-2832` |
| 2   | **RomanianBERT** | `DATASETS/LegalNERO/logs/RomanianBERT/greek_legal_ner/dumitrescustefan/bert-base-romanian-cased-v1/results_after_training_only_on_language_with_id__all/seed_1/checkpoint-2832` |
| 3   | **mDeBERTa-v3** | `DATASETS/LegalNERO/logs/mDeBERTa/greek_legal_ner/microsoft/mdeberta-v3-base/seed_1/checkpoint-2832` |
| 4   | **FastCharEmbeddings** | `Char` |
| 5   | **BPEmb** | `bpe-ro-100000-300` |

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | FastText  | LegalRoBERTa | RomanianBERT | mDeBERTa-v3 | FastCharEmbeddings |   BPEmb   | Kept  |
|:-------:|:------:|:------------:| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |:-----:|
|    1    |   29   |   80.68 ★    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  6/6  |
|    2    |   17   |   80.82 ★    |    1/1    |    1/1    |    0/1    |    1/1    |    1/1    |    1/1    |  5/6  |
|    3    |   21   |    80.76     |    0/1    |    1/1    |    0/1    |    1/1    |    0/1    |    1/1    |  3/6  |
|    4    |   28   |    80.43     |    1/1    |    0/1    |    0/1    |    1/1    |    0/1    |    1/1    |  3/6  |
|    5    |   16   |   80.90 ★    |    0/1    |    1/1    |    0/1    |    1/1    |    0/1    |    1/1    |  3/6  |
|    6    |   30   |    77.26     |    1/1    |    0/1    |    0/1    |    0/1    |    0/1    |    1/1    |  2/6  |
|    7    |   27   |   81.29 ★    |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |    0/1    |  4/6  |
|    8    |   17   |    80.10     |    1/1    |    1/1    |    1/1    |    1/1    |    0/1    |    1/1    |  5/6  |
|    9    |   22   |    80.99     |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    0/1    |  5/6  |
|   10    |   17   |    80.05     |    0/1    |    1/1    |    0/1    |    1/1    |    0/1    |    0/1    |  2/6  |
|   11    |   20   |    80.43     |    1/1    |    1/1    |    0/1    |    1/1    |    1/1    |    1/1    |  5/6  |
|   12    |   43   |    79.71     |    0/1    |    1/1    |    0/1    |    1/1    |    1/1    |    0/1    |  3/6  |
|   13    |   28   |    80.20     |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |    0/1    |  4/6  |
|   14    |   21   |    80.53     |    0/1    |    1/1    |    1/1    |    0/1    |    1/1    |    1/1    |  4/6  |
|   15    |   27   |    79.12     |    1/1    |    1/1    |    0/1    |    1/1    |    0/1    |    1/1    |  4/6  |
|   16    |   32   |    79.71     |    1/1    |    1/1    |    1/1    |    0/1    |    1/1    |    0/1    |  4/6  |
|   17    |   18   |   81.37 ★    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   18    |   23   |    80.92     |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |    0/1    |  4/6  |
|   19    |   19   |    80.78     |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   20    |   16   |    80.92     |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  5/6  |
|   21    |   22   |   81.88 ★    |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |    0/1    |  4/6  |
|   22    |   18   |    80.98     |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  6/6  |
|   23    |   26   |    81.84     |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |    0/1    |  3/6  |
|   24    |   27   |    79.41     |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |    0/1    |  4/6  |
|   25    |   25   |    81.06     |    1/1    |    0/1    |    1/1    |    1/1    |    1/1    |    0/1    |  4/6  |
|   26    |   17   |    81.22     |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |    0/1    |  3/6  |
|   27    |   16   |    81.32     |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   28    |   23   |    81.19     |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  5/6  |
|   29    |   19   |    80.53     |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |    0/1    |  3/6  |
|   30    |   18   |    80.33     |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |  4/6  |

**Σύνολο:** 30 episodes, ~682 epochs, ~6h 14m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Test F1 | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 80.68 | 6/6 | FastText 1/1, LegalRoBERTa 1/1, RomanianBERT 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |
| Νέο baseline | 2 | 80.82 | 5/6 | FastText 1/1, LegalRoBERTa 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |
| Νέο baseline | 5 | 80.90 | 3/6 | LegalRoBERTa 1/1, mDeBERTa-v3 1/1, BPEmb 1/1 |
| Νέο baseline | 7 | 81.29 | 4/6 | LegalRoBERTa 1/1, RomanianBERT 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1 |
| Νέο baseline | 17 | 81.37 | 4/6 | RomanianBERT 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |
| **Τελικό** ★ | 21 | 81.88 | 4/6 | LegalRoBERTa 1/1, RomanianBERT 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1 |

---

## 5. Best Model Action (Episode 0)

| Embedding          | Groups     | Kept | Σημείωση                    |
|--------------------|------------|:----:|-----------------------------|
| FastText           | [0]        | 0/1  | **Αποκλείστηκε**            |
| LegalRoBERTa       | [1]        | 1/1  | Πλήρης χρήση                |
| RomanianBERT       | [1]        | 1/1  | Πλήρης χρήση                |
| mDeBERTa-v3        | [1]        | 1/1  | Πλήρης χρήση                |
| FastCharEmbeddings | [1]        | 1/1  | Πλήρης χρήση                |
| BPEmb              | [0]        | 0/1  | **Αποκλείστηκε**            |
| **Σύνολο**         |            | **4/6 (66.7%)** | |

---

## 6. Συχνότητα Επιλογής Embedding

| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |
|--------------------|:----------------------:|:-------:|
| mDeBERTa-v3        | 27/30                     | 90.0%   |
| FastCharEmbeddings | 23/30                     | 76.7%   |
| RomanianBERT       | 21/30                     | 70.0%   |
| LegalRoBERTa       | 20/30                     | 66.7%   |
| BPEmb              | 17/30                     | 56.7%   |
| FastText           | 11/30                     | 36.7%   |

---

## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Αποτελέσματα ανά Οντότητα

| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |
|-------------|-----:|----:|----:|:---------:|:------:|:----------:|
| PER         |    4 |   0 |   1 |   100.00% | 80.00% | **88.89%** |
| TIME        |  111 |  21 |  21 |    84.09% | 84.09% | **84.09%** |
| ORG         |  128 |  20 |  34 |    86.49% | 79.01% | **82.58%** |
| LEGAL       |   80 |  19 |  28 |    80.81% | 74.07% | **77.29%** |
| LOC         |    5 |   4 |  20 |    55.56% | 20.00% | **29.41%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    83.67% | 75.93% | **79.61%** |
| **MACRO AVG**  |     —     |   —    | **72.45%** |
