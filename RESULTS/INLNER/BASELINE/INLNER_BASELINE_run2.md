# ACE — dataset: Seed ?

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **88.34%** |
| **Macro F1 (Test)**  | **85.35%** |
| Precision (Micro)    | 88.22%     |
| Recall (Micro)       | 88.47%     |
| Accuracy (Micro)     | 79.12%     |

**Μέθοδος:** ACE (N=1 groups / embedding, equal mode, seed ?)  
**Best Model Save:** Episode 0, Dev F1 = 0.00% (Test F1 = 0.00%)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 88.34%  
**Διάρκεια Εκπαίδευσης:** ~0h 36m (1 episodes)

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
| Train | 10,725    |
| Dev   | 1,070     |
| Test  | 4,602     |

---

## 3. Embeddings (6 Υποψήφια × 1 Groups = 6 Actions)

| #   | Embedding (short)    | Raw Path / ID                          |
|-----|----------------------|----------------------------------------|
| 0   | **InLegalBERT** | `AIAI/InLNER/InLegalBERT/greek_legal_ner/law-ai/InLegalBERT/results_after_training_only_on_language_with_id__en/seed_1` |
| 1   | **LegalBERT** | `AIAI/InLNER/LegalBERT/greek_legal_ner/nlpaueb/legal-bert-base-uncased/results_after_training_only_on_language_with_id__en/seed_1` |
| 2   | **FastText** | `AIAI/InLNER/Non_Contexual_embeddings/fasttext-wiki-news-subwords-300.model` |
| 3   | **mDeBERTa-v3** | `AIAI/InLNER/mDeBERTa/greek_legal_ner/microsoft/mdeberta-v3-base/seed_1/checkpoint-8046` |
| 4   | **FastCharEmbeddings** | `Char` |
| 5   | **BPEmb** | `bpe-en-100000-300` |

---

## 4. Πρόοδος Εκπαίδευσης ανά Episode

| Episode | Epochs | Best Test F1 | InLegalBERT | LegalBERT | FastText  | mDeBERTa-v3 | FastCharEmbeddings |   BPEmb   | Kept  |
|:-------:|:------:|:------------:| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |:-----:|
|    1    |   25   |    0.00 ★    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  6/6  |

**Σύνολο:** 1 episodes, ~25 epochs, ~0h 36m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Dev F1  | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 88.46 | 6/6 | InLegalBERT 1/1, LegalBERT 1/1, FastText 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |

---

## 5. Best Model Action (Episode 0)

| Embedding          | Groups     | Kept | Σημείωση                    |
|--------------------|------------|:----:|-----------------------------|
| InLegalBERT        | [1]        | 1/1  | Πλήρης χρήση                |
| LegalBERT          | [1]        | 1/1  | Πλήρης χρήση                |
| FastText           | [1]        | 1/1  | Πλήρης χρήση                |
| mDeBERTa-v3        | [1]        | 1/1  | Πλήρης χρήση                |
| FastCharEmbeddings | [1]        | 1/1  | Πλήρης χρήση                |
| BPEmb              | [1]        | 1/1  | Πλήρης χρήση                |
| **Σύνολο**         |            | **6/6 (100.0%)** | |

---

## 6. Συχνότητα Επιλογής Embedding

| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |
|--------------------|:----------------------:|:-------:|
| InLegalBERT        | 1/1                     | 100.0%   |
| LegalBERT          | 1/1                     | 100.0%   |
| FastText           | 1/1                     | 100.0%   |
| mDeBERTa-v3        | 1/1                     | 100.0%   |
| FastCharEmbeddings | 1/1                     | 100.0%   |
| BPEmb              | 1/1                     | 100.0%   |

---

## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)

### Αποτελέσματα ανά Οντότητα

| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |
|-------------|-----:|----:|----:|:---------:|:------:|:----------:|
| STATUTE     |  921 |  48 |  52 |    95.05% | 94.66% | **94.85%** |
| PROVISION   | 1153 |  57 |  69 |    95.29% | 94.35% | **94.82%** |
| DATE        | 1045 |  70 |  67 |    93.72% | 93.97% | **93.84%** |
| JUDGE       |   41 |   5 |   1 |    89.13% | 97.62% | **93.18%** |
| COURT       |  737 |  52 |  62 |    93.41% | 92.24% | **92.82%** |
| WITNESS     |  378 |  34 |  36 |    91.75% | 91.30% | **91.52%** |
| OTHER_PERSON |  977 |  88 | 134 |    91.74% | 87.94% | **89.80%** |
| CASE_NUMBER |  583 | 127 |  85 |    82.11% | 87.28% | **84.62%** |
| GPE         |  589 | 148 | 130 |    79.92% | 81.92% | **80.91%** |
| ORG         |  734 | 203 | 180 |    78.34% | 80.31% | **79.31%** |
| PRECEDENT   |  499 | 174 | 160 |    74.15% | 75.72% | **74.93%** |
| PETITIONER  |   41 |   7 |  23 |    85.42% | 64.06% | **73.21%** |
| RESPONDENT  |   25 |  18 |   8 |    58.14% | 75.76% | **65.79%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    88.22% | 88.47% | **88.34%** |
| **MACRO AVG**  |     —     |   —    | **85.35%** |
