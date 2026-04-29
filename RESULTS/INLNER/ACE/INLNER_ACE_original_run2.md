# ACE — dataset: Seed ?

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **87.05%** |
| **Macro F1 (Test)**  | **82.98%** |
| Precision (Micro)    | 87.31%     |
| Recall (Micro)       | 86.80%     |
| Accuracy (Micro)     | 77.08%     |

**Μέθοδος:** ACE (N=1 groups / embedding, equal mode, seed ?)  
**Best Model Save:** Episode 0, Test F1 = 0.00%  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 87.05%  
**Διάρκεια Εκπαίδευσης:** ~12h 37m (30 episodes)

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
| Max Episodes                   | 30 |
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
| Train | 10,725    |
| Dev   | 1,070     |
| Test  | 4,602     |

---

## 3. Embeddings (6 Υποψήφια × 1 Groups = 6 Actions)

| #   | Embedding (short)    | Raw Path / ID                          |
|-----|----------------------|----------------------------------------|
| 0   | **DistilBERT** | `DATASETS/InLNER/DistilBERT/greek_legal_ner/distilbert-base-multilingual-cased/seed_1` |
| 1   | **FastText** | `DATASETS/InLNER/Non_Contexual_embeddings/fasttext-wiki-news-subwords-300.model` |
| 2   | **GloVe** | `DATASETS/InLNER/Non_Contexual_embeddings/glove-wiki-gigaword-300.model` |
| 3   | **Word2Vec** | `DATASETS/InLNER/Non_Contexual_embeddings/word2vec-google-news-300.model` |
| 4   | **XLM-R-Base** | `DATASETS/InLNER/XLM-R-Base/greek_legal_ner/xlm-roberta-base/seed_1` |
| 5   | **mDeBERTa-v3** | `DATASETS/InLNER/mDeBERTa/greek_legal_ner/microsoft/mdeberta-v3-base/seed_1/checkpoint-9387` |

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | DistilBERT | FastText  |   GloVe   | Word2Vec  | XLM-R-Base | mDeBERTa-v3 | Kept  |
|:-------:|:------:|:------------:| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |:-----:|
|    1    |   25   |   87.86 ★    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  6/6  |
|    2    |   29   |    87.84     |    0/1    |    0/1    |    0/1    |    1/1    |    0/1    |    1/1    |  2/6  |
|    3    |   17   |    87.55     |    1/1    |    1/1    |    1/1    |    0/1    |    0/1    |    1/1    |  4/6  |
|    4    |   39   |    85.86     |    1/1    |    1/1    |    1/1    |    0/1    |    0/1    |    0/1    |  3/6  |
|    5    |   17   |    87.37     |    1/1    |    0/1    |    1/1    |    0/1    |    1/1    |    0/1    |  3/6  |
|    6    |   23   |    85.96     |    1/1    |    0/1    |    1/1    |    0/1    |    0/1    |    0/1    |  2/6  |
|    7    |   46   |   87.90 ★    |    0/1    |    1/1    |    0/1    |    1/1    |    0/1    |    1/1    |  3/6  |
|    8    |   20   |    87.80     |    0/1    |    1/1    |    0/1    |    0/1    |    0/1    |    1/1    |  2/6  |
|    9    |   20   |   88.25 ★    |    0/1    |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |  2/6  |
|   10    |   18   |    88.03     |    0/1    |    0/1    |    1/1    |    0/1    |    1/1    |    1/1    |  3/6  |
|   11    |   31   |    87.77     |    0/1    |    1/1    |    1/1    |    1/1    |    0/1    |    1/1    |  4/6  |
|   12    |   23   |    87.83     |    0/1    |    0/1    |    1/1    |    1/1    |    0/1    |    1/1    |  3/6  |
|   13    |   21   |    88.01     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   14    |   21   |    87.04     |    0/1    |    1/1    |    0/1    |    1/1    |    1/1    |    0/1    |  3/6  |
|   15    |   17   |    87.84     |    0/1    |    1/1    |    1/1    |    0/1    |    1/1    |    1/1    |  4/6  |
|   16    |   24   |    87.54     |    1/1    |    1/1    |    0/1    |    1/1    |    0/1    |    1/1    |  4/6  |
|   17    |   23   |    87.79     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   18    |   33   |    87.67     |    0/1    |    1/1    |    0/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   19    |   21   |    87.66     |    0/1    |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |  2/6  |
|   20    |   30   |    87.44     |    0/1    |    1/1    |    0/1    |    0/1    |    1/1    |    1/1    |  3/6  |
|   21    |   27   |    87.58     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   22    |   25   |    87.55     |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   23    |   19   |    87.41     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   24    |   27   |    87.85     |    0/1    |    1/1    |    0/1    |    1/1    |    0/1    |    1/1    |  3/6  |
|   25    |   16   |    87.82     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |
|   26    |   20   |    88.14     |    1/1    |    0/1    |    1/1    |    0/1    |    1/1    |    1/1    |  4/6  |
|   27    |   20   |    87.57     |    0/1    |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |  2/6  |
|   28    |   20   |    87.45     |    0/1    |    1/1    |    1/1    |    0/1    |    1/1    |    1/1    |  4/6  |
|   29    |   17   |    87.47     |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |    1/1    |  4/6  |
|   30    |   31   |    87.55     |    0/1    |    0/1    |    0/1    |    1/1    |    1/1    |    1/1    |  3/6  |

**Σύνολο:** 30 episodes, ~720 epochs, ~12h 37m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Test F1 | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 87.86 | 6/6 | DistilBERT 1/1, FastText 1/1, GloVe 1/1, Word2Vec 1/1, XLM-R-Base 1/1, mDeBERTa-v3 1/1 |
| Νέο baseline | 7 | 87.90 | 3/6 | FastText 1/1, Word2Vec 1/1, mDeBERTa-v3 1/1 |
| **Τελικό** ★ | 9 | 88.25 | 2/6 | XLM-R-Base 1/1, mDeBERTa-v3 1/1 |

---

## 5. Best Model Action (Episode 0)

| Embedding          | Groups     | Kept | Σημείωση                    |
|--------------------|------------|:----:|-----------------------------|
| DistilBERT         | [0]        | 0/1  | **Αποκλείστηκε**            |
| FastText           | [0]        | 0/1  | **Αποκλείστηκε**            |
| GloVe              | [0]        | 0/1  | **Αποκλείστηκε**            |
| Word2Vec           | [0]        | 0/1  | **Αποκλείστηκε**            |
| XLM-R-Base         | [1]        | 1/1  | Πλήρης χρήση                |
| mDeBERTa-v3        | [1]        | 1/1  | Πλήρης χρήση                |
| **Σύνολο**         |            | **2/6 (33.3%)** | |

---

## 6. Συχνότητα Επιλογής Embedding

| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |
|--------------------|:----------------------:|:-------:|
| mDeBERTa-v3        | 26/30                     | 86.7%   |
| XLM-R-Base         | 20/30                     | 66.7%   |
| Word2Vec           | 17/30                     | 56.7%   |
| FastText           | 13/30                     | 43.3%   |
| GloVe              | 13/30                     | 43.3%   |
| DistilBERT         | 7/30                     | 23.3%   |

---

## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Αποτελέσματα ανά Οντότητα

| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |
|-------------|-----:|----:|----:|:---------:|:------:|:----------:|
| PROVISION   | 1148 |  67 |  74 |    94.49% | 93.94% | **94.21%** |
| DATE        | 1050 |  74 |  62 |    93.42% | 94.42% | **93.92%** |
| STATUTE     |  901 |  62 |  72 |    93.56% | 92.60% | **93.08%** |
| WITNESS     |  381 |  33 |  33 |    92.03% | 92.03% | **92.03%** |
| COURT       |  730 |  63 |  69 |    92.06% | 91.36% | **91.71%** |
| JUDGE       |   39 |   7 |   3 |    84.78% | 92.86% | **88.64%** |
| OTHER_PERSON |  970 | 108 | 141 |    89.98% | 87.31% | **88.62%** |
| CASE_NUMBER |  564 | 123 | 104 |    82.10% | 84.43% | **83.25%** |
| GPE         |  563 | 150 | 156 |    78.96% | 78.30% | **78.63%** |
| ORG         |  680 | 199 | 234 |    77.36% | 74.40% | **75.85%** |
| PRECEDENT   |  492 | 181 | 167 |    73.11% | 74.66% | **73.88%** |
| PETITIONER  |   37 |  14 |  27 |    72.55% | 57.81% | **64.35%** |
| RESPONDENT  |   23 |  20 |  10 |    53.49% | 69.70% | **60.53%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    87.31% | 86.80% | **87.05%** |
| **MACRO AVG**  |     —     |   —    | **82.98%** |
