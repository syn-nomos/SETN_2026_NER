# ACE — NER: Seed ?

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **77.13%** |
| **Macro F1 (Test)**  | **64.38%** |
| Precision (Micro)    | 81.28%     |
| Recall (Micro)       | 73.38%     |
| Accuracy (Micro)     | 62.77%     |

**Μέθοδος:** ACE (N=1 groups / embedding, equal mode, seed ?)  
**Best Model Save:** Episode 0, Test F1 = 0.00%  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 77.13%  
**Διάρκεια Εκπαίδευσης:** ~8h 43m (30 episodes)

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
|    1    |   27   |   80.38 ★    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  6/6  |
|    2    |   18   |   80.82 ★    |    1/1    |    1/1    |    0/1    |    1/1    |    1/1    |    1/1    |  5/6  |
|    3    |   26   |    79.91     |    1/1    |    1/1    |    1/1    |    1/1    |    0/1    |    1/1    |  5/6  |
|    4    |   20   |    78.65     |    0/1    |    1/1    |    0/1    |    0/1    |    0/1    |    0/1    |  1/6  |
|    5    |   28   |    79.48     |    1/1    |    0/1    |    1/1    |    0/1    |    1/1    |    0/1    |  3/6  |
|    6    |   21   |    80.23     |    1/1    |    1/1    |    0/1    |    1/1    |    0/1    |    0/1    |  3/6  |
|    7    |   22   |    79.56     |    0/1    |    1/1    |    0/1    |    1/1    |    1/1    |    0/1    |  3/6  |
|    8    |   20   |    79.76     |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    0/1    |  5/6  |
|    9    |   17   |    78.54     |    1/1    |    1/1    |    1/1    |    0/1    |    1/1    |    1/1    |  5/6  |
|   10    |   19   |    78.61     |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    0/1    |  5/6  |
|   11    |   37   |   81.83 ★    |    1/1    |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |  3/6  |
|   12    |   36   |    77.97     |    0/1    |    1/1    |    0/1    |    0/1    |    0/1    |    1/1    |  2/6  |
|   13    |   36   |   82.12 ★    |    1/1    |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |  3/6  |
|   14    |   45   |   82.15 ★    |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   15    |   21   |    81.67     |    1/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   16    |   16   |    78.89     |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  6/6  |
|   17    |   20   |    80.82     |    1/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   18    |   44   |    78.51     |    1/1    |    0/1    |    0/1    |    0/1    |    0/1    |    1/1    |  2/6  |
|   19    |   37   |    80.44     |    1/1    |    0/1    |    0/1    |    1/1    |    1/1    |    0/1    |  3/6  |
|   20    |   29   |    82.07     |    1/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   21    |   21   |    82.03     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   22    |   27   |    80.92     |    1/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   23    |   28   |    80.62     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   24    |   22   |    81.65     |    1/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   25    |   54   |    81.40     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   26    |   30   |    81.06     |    1/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   27    |   18   |    81.59     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   28    |   29   |    81.53     |    1/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   29    |   24   |    81.14     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   30    |   26   |    81.67     |    1/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  4/6  |

**Σύνολο:** 30 episodes, ~818 epochs, ~8h 43m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Test F1 | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 80.38 | 6/6 | FastText 1/1, LegalRoBERTa 1/1, RomanianBERT 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |
| Νέο baseline | 2 | 80.82 | 5/6 | FastText 1/1, LegalRoBERTa 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |
| Νέο baseline | 11 | 81.83 | 3/6 | FastText 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |
| Νέο baseline | 13 | 82.12 | 3/6 | FastText 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |
| **Τελικό** ★ | 14 | 82.15 | 3/6 | mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |

---

## 5. Best Model Action (Episode 0)

| Embedding          | Groups     | Kept | Σημείωση                    |
|--------------------|------------|:----:|-----------------------------|
| FastText           | [0]        | 0/1  | **Αποκλείστηκε**            |
| LegalRoBERTa       | [0]        | 0/1  | **Αποκλείστηκε**            |
| RomanianBERT       | [0]        | 0/1  | **Αποκλείστηκε**            |
| mDeBERTa-v3        | [1]        | 1/1  | Πλήρης χρήση                |
| FastCharEmbeddings | [1]        | 1/1  | Πλήρης χρήση                |
| BPEmb              | [1]        | 1/1  | Πλήρης χρήση                |
| **Σύνολο**         |            | **3/6 (50.0%)** | |

---

## 6. Συχνότητα Επιλογής Embedding

| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |
|--------------------|:----------------------:|:-------:|
| FastCharEmbeddings | 25/30                     | 83.3%   |
| mDeBERTa-v3        | 23/30                     | 76.7%   |
| BPEmb              | 23/30                     | 76.7%   |
| FastText           | 21/30                     | 70.0%   |
| LegalRoBERTa       | 11/30                     | 36.7%   |
| RomanianBERT       | 7/30                     | 23.3%   |

---

## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Αποτελέσματα ανά Οντότητα

| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |
|-------------|-----:|----:|----:|:---------:|:------:|:----------:|
| TIME        |  107 |  20 |  25 |    84.25% | 81.06% | **82.62%** |
| LEGAL       |   86 |  19 |  22 |    81.90% | 79.63% | **80.75%** |
| ORG         |  117 |  23 |  45 |    83.57% | 72.22% | **77.48%** |
| PER         |    3 |   2 |   2 |    60.00% | 60.00% | **60.00%** |
| LOC         |    4 |   9 |  21 |    30.77% | 16.00% | **21.05%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    81.28% | 73.38% | **77.13%** |
| **MACRO AVG**  |     —     |   —    | **64.38%** |
