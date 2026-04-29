# ACE — dataset: Seed ?

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **88.24%** |
| **Macro F1 (Test)**  | **85.07%** |
| Precision (Micro)    | 88.32%     |
| Recall (Micro)       | 88.16%     |
| Accuracy (Micro)     | 78.95%     |

**Μέθοδος:** ACE (N=1 groups / embedding, equal mode, seed ?)  
**Best Model Save:** Episode 0, Dev F1 = 0.00% (Test F1 = 0.00%)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 88.24%  
**Διάρκεια Εκπαίδευσης:** ~0h 43m (1 episodes)

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
|    1    |   30   |    0.00 ★    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  6/6  |

**Σύνολο:** 1 episodes, ~30 epochs, ~0h 43m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Dev F1  | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 88.48 | 6/6 | InLegalBERT 1/1, LegalBERT 1/1, FastText 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |

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
| STATUTE     |  913 |  48 |  60 |    95.01% | 93.83% | **94.42%** |
| PROVISION   | 1144 |  63 |  78 |    94.78% | 93.62% | **94.20%** |
| DATE        | 1040 |  78 |  72 |    93.02% | 93.53% | **93.27%** |
| COURT       |  732 |  50 |  67 |    93.61% | 91.61% | **92.60%** |
| JUDGE       |   41 |   6 |   1 |    87.23% | 97.62% | **92.13%** |
| WITNESS     |  380 |  36 |  34 |    91.35% | 91.79% | **91.57%** |
| OTHER_PERSON |  970 |  84 | 141 |    92.03% | 87.31% | **89.61%** |
| CASE_NUMBER |  583 | 117 |  85 |    83.29% | 87.28% | **85.24%** |
| GPE         |  591 | 152 | 128 |    79.54% | 82.20% | **80.85%** |
| ORG         |  734 | 194 | 180 |    79.09% | 80.31% | **79.70%** |
| PRECEDENT   |  503 | 165 | 156 |    75.30% | 76.33% | **75.81%** |
| PETITIONER  |   42 |   7 |  22 |    85.71% | 65.62% | **74.33%** |
| RESPONDENT  |   23 |  18 |  10 |    56.10% | 69.70% | **62.16%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    88.32% | 88.16% | **88.24%** |
| **MACRO AVG**  |     —     |   —    | **85.07%** |
