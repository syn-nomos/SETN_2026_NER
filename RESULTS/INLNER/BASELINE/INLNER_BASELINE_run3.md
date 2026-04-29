# ACE — dataset: Seed ?

## 1. Σύνοψη

| Μετρική              | Τιμή       |
|----------------------|------------|
| **Micro F1 (Test)**  | **88.47%** |
| **Macro F1 (Test)**  | **85.36%** |
| Precision (Micro)    | 88.44%     |
| Recall (Micro)       | 88.51%     |
| Accuracy (Micro)     | 79.33%     |

**Μέθοδος:** ACE (N=1 groups / embedding, equal mode, seed ?)  
**Best Model Save:** Episode 0, Dev F1 = 0.00% (Test F1 = 0.00%)  
**Τελική Αξιολόγηση (best-model.pt):** Test F1 = 88.47%  
**Διάρκεια Εκπαίδευσης:** ~0h 52m (1 episodes)

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
|    1    |   38   |    0.00 ★    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |    1/1    |  6/6  |

**Σύνολο:** 1 episodes, ~38 epochs, ~0h 52m εκπαίδευσης

### Baseline Transitions

| Γεγονός         | Episode | Dev F1  | Groups Kept | Σημείωση |
|-----------------|:-------:|:-------:|:-----------:|----------|
| Αρχικό baseline | 1 | 88.63 | 6/6 | InLegalBERT 1/1, LegalBERT 1/1, FastText 1/1, mDeBERTa-v3 1/1, FastCharEmbeddings 1/1, BPEmb 1/1 |

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
| STATUTE     |  921 |  45 |  52 |    95.34% | 94.66% | **95.00%** |
| PROVISION   | 1153 |  57 |  69 |    95.29% | 94.35% | **94.82%** |
| DATE        | 1046 |  70 |  66 |    93.73% | 94.06% | **93.89%** |
| JUDGE       |   41 |   5 |   1 |    89.13% | 97.62% | **93.18%** |
| COURT       |  736 |  56 |  63 |    92.93% | 92.12% | **92.52%** |
| WITNESS     |  379 |  36 |  35 |    91.33% | 91.55% | **91.44%** |
| OTHER_PERSON |  972 |  86 | 139 |    91.87% | 87.49% | **89.63%** |
| CASE_NUMBER |  591 | 118 |  77 |    83.36% | 88.47% | **85.84%** |
| GPE         |  595 | 148 | 124 |    80.08% | 82.75% | **81.39%** |
| ORG         |  729 | 190 | 185 |    79.33% | 79.76% | **79.54%** |
| PRECEDENT   |  498 | 173 | 161 |    74.22% | 75.57% | **74.89%** |
| PETITIONER  |   42 |   7 |  22 |    85.71% | 65.62% | **74.33%** |
| RESPONDENT  |   24 |  19 |   9 |    55.81% | 72.73% | **63.16%** |

### Συνολικά

| Μετρική        | Precision | Recall | F1         |
|----------------|:---------:|:------:|:----------:|
| **MICRO AVG**  |    88.44% | 88.51% | **88.47%** |
| **MACRO AVG**  |     —     |   —    | **85.36%** |
