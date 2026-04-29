"""Download non-contextual embeddings for InLNER from HuggingFace."""
from huggingface_hub import hf_hub_download
import os
import sys

dest = r'D:\Το Drive μου\AEGEAN UNIVERSITY\LEGAL DOCUMENTS ARCHIVE\ΠΑΙΓΑΙΟΥ\CODE\03_ML_MODELS\NER_MODELS\ACE\AIAI\InLNER\Non_Contexual_embeddings'

downloads = [
    # GloVe (~501 MB total)
    ('fse/glove-wiki-gigaword-300', 'glove-wiki-gigaword-300.model'),
    ('fse/glove-wiki-gigaword-300', 'glove-wiki-gigaword-300.model.vectors.npy'),
    # FastText wiki-news-subwords (~1.25 GB total)
    ('fse/fasttext-wiki-news-subwords-300', 'fasttext-wiki-news-subwords-300.model'),
    ('fse/fasttext-wiki-news-subwords-300', 'fasttext-wiki-news-subwords-300.model.vectors.npy'),
    # Word2Vec Google News (~3.78 GB total)
    ('fse/word2vec-google-news-300', 'word2vec-google-news-300.model'),
    ('fse/word2vec-google-news-300', 'word2vec-google-news-300.model.vectors.npy'),
]

for repo, filename in downloads:
    target = os.path.join(dest, filename)
    if os.path.exists(target):
        size_mb = os.path.getsize(target) / 1024 / 1024
        print(f"SKIP {filename} (already exists, {size_mb:.1f} MB)")
        continue
    print(f"DOWNLOADING {repo}/{filename} ...")
    try:
        hf_hub_download(repo, filename, local_dir=dest)
        size_mb = os.path.getsize(os.path.join(dest, filename)) / 1024 / 1024
        print(f"  OK ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)

print("\n=== DONE ===")
print("Files in destination:")
for f in sorted(os.listdir(dest)):
    if not f.startswith('.'):
        size_mb = os.path.getsize(os.path.join(dest, f)) / 1024 / 1024
        print(f"  {f}: {size_mb:.1f} MB")
