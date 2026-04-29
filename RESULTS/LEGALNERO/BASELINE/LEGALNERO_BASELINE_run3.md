# ACE — dataset: Seed ?

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **79.01%** |
| **Macro F1 (Test)**  | **75.67%** |
| Precision (Micro)    | 84.66%     |
| Recall (Micro)       | 74.07%     |
| Accuracy (Micro)     | 65.31%     |

**Μέθοδος:** ACE (N=1 groups / embedding, equal mode, seed ?)  
**Best Model Save:** Episode 0, Dev F1 = 0.00% (Test F1 = 0.00%)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 79.01%  
**Διάρκεια Εκπαίδευσης:** ~0h 13m (1 episodes)

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
| Train With Dev                 | ✓ |
| True Reshuffle                 | ✗ |
| Controller Momentum            | 0.9 |
| Discount                       | 0.5 |

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
| 0   | **FastText** | `AIAI/LegalNERO/cc.ro.300.vec` |
| 1   | **LegalRoBERTa** | `AIAI/LegalNERO/logs/LegalRomanianRoBERTa/greek_legal_ner/joelito/legal-romanian-roberta-base/results_after_training_only_on_language_with_id__ro/seed_1/checkpoint-2832` |
| 2   | **RomanianBERT** | `AIAI/LegalNERO/logs/RomanianBERT/greek_legal_ner/dumitrescustefan/bert-base-romanian-cased-v1/results_after_training_only_on_language_with_id__all/seed_1/checkpoint-2832` |
| 3   | **mDeBERTa-v3** | `AIAI/LegalNERO/logs/mDeBERTa/greek_legal_ner/microsoft/mdeberta-v3-base/seed_1/checkpoint-2832` |
| 4   | **FastCharEmbeddings** | `Char` |
| 5   | **BPEmb** | `bpe-ro-100000-300` |

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | FastText  | LegalRoBERTa | RomanianBERT | mDeBERTa-v3 | FastCharEmbeddings |   BPEmb   | Kept  |
|:-------:|:------:|:------------:| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |:-----:|
|    1    |   20   |    0.00 ★    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  6/6  |

**Σύνολο:** 1 episodes, ~20 epochs, ~0h 13m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Dev F1  | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 81.41 | 6/6 | FastText 1/1, LegalRoBERTa 1/1, RomanianBERT 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |

---

## 5. Best Model Action (Episode 0)

| Embedding          | Groups     | Kept | Σημείωση                    |
|--------------------|------------|:----:|-----------------------------|
| FastText           | [1]        | 1/1  | Πλήρης χρήση                |
| LegalRoBERTa       | [1]        | 1/1  | Πλήρης χρήση                |
| RomanianBERT       | [1]        | 1/1  | Πλήρης χρήση                |
| mDeBERTa-v3        | [1]        | 1/1  | Πλήρης χρήση                |
| FastCharEmbeddings | [1]        | 1/1  | Πλήρης χρήση                |
| BPEmb              | [1]        | 1/1  | Πλήρης χρήση                |
| **Σύνολο**         |            | **6/6 (100.0%)** | |

---

## 6. Συχνότητα Επιλογής Embedding

| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |
|--------------------|:----------------------:|:-------:|
| FastText           | 1/1                     | 100.0%   |
| LegalRoBERTa       | 1/1                     | 100.0%   |
| RomanianBERT       | 1/1                     | 100.0%   |
| mDeBERTa-v3        | 1/1                     | 100.0%   |
| FastCharEmbeddings | 1/1                     | 100.0%   |
| BPEmb              | 1/1                     | 100.0%   |

---

## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Αποτελέσματα ανά Οντότητα

| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |
|-------------|-----:|----:|----:|:---------:|:------:|:----------:|
| PER         |    5 |   0 |   0 |   100.00% | 100.00% | **100.00%** |
| TIME        |  113 |  26 |  19 |    81.29% | 85.61% | **83.39%** |
| ORG         |  120 |  15 |  42 |    88.89% | 74.07% | **80.81%** |
| LEGAL       |   75 |  11 |  33 |    87.21% | 69.44% | **77.32%** |
| LOC         |    7 |   6 |  18 |    53.85% | 28.00% | **36.84%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    84.66% | 74.07% | **79.01%** |
| **MACRO AVG**  |     —     |   —    | **75.67%** |
