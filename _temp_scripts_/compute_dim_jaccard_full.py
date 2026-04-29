"""Compute dim-level Jaccard stability across seeds for ALL (dataset, method) pairs.

For each (dataset, method, seed) we replay the training-time random
permutation (torch.randperm seeded with `group_seed`), expand the action
vector to the *actual* dim indices selected, and compute mean pairwise
Jaccard over those true dim sets.

We compare:
  index-J : Jaccard on chunk-indices (the metric currently in the table)
  dim-J   : Jaccard on actual selected dim indices
  random  : expected Jaccard if dims were chosen uniformly at random of same size

The dim-J vs random comparison answers the question:
  "Is the algorithm consistently picking the same dimensions, or is it
   essentially random?"

NOTE: We do NOT need the trained .pt files for this — only the action
vectors (already in compute_selection_stability.py) and the embedding
dimensions (verified once from local checkpoints / configs).
"""
from itertools import combinations
import torch

# ------------------------------------------------------------------
# Embedding dimensions per dataset (verified from local best-model.pt
# files + configs).
# Order matches the canonical lists used in compute_selection_stability.py.
# GLN     : GreekBERT, GreekLegalRoBERTa, FastText, mDeBERTa, Char, BPEmb
# InLNER  : InLegalBERT, LegalBERT, FastText, mDeBERTa, Char, BPEmb
# LegalNERO: FastText, LegalRoBERTa, RomanianBERT, mDeBERTa, Char, BPEmb
# ------------------------------------------------------------------
GLN_EMBS = ["GreekBERT", "GreekLegalRoBERTa", "FastText", "mDeBERTa", "Char", "BPEmb"]
GLN_DIMS = [768,         768,                 300,         768,        50,     600]

INL_EMBS = ["InLegalBERT", "LegalBERT", "FastText", "mDeBERTa", "Char", "BPEmb"]
INL_DIMS = [768,           768,         300,         768,        50,     600]

LNG_EMBS = ["FastText", "LegalRoBERTa", "RomanianBERT", "mDeBERTa", "Char", "BPEmb"]
LNG_DIMS = [300,         768,            768,            768,        50,     600]

HACE_GLN_EMBS = ["GreekBERT", "GreekLegalRoBERTa", "mDeBERTa", "Char"]
HACE_GLN_DIMS = [768, 768, 768, 50]
HACE_INL_EMBS = ["InLegalBERT", "FastText", "Char", "BPEmb"]
HACE_INL_DIMS = [768, 300, 50, 600]
HACE_LNG_EMBS = ["LegalRoBERTa", "RomanianBERT", "mDeBERTa", "Char"]
HACE_LNG_DIMS = [768, 768, 768, 50]

# ------------------------------------------------------------------
# Action vectors (copied from compute_selection_stability.py).
# Each value: dict[embedding_name] = list[0/1] of length N (= num_groups).
# ------------------------------------------------------------------
DATA = {
    # ============ GLN ============
    ("GLN", "G-ACE-4"): (GLN_EMBS, GLN_DIMS, 4, {
        44: dict(zip(GLN_EMBS, [
            [0,0,1,0], [1,1,0,1], [1,1,1,0], [1,1,0,1], [0,0,0,1], [0,0,1,0]])),
        441: dict(zip(GLN_EMBS, [
            [1,0,0,0], [0,0,0,1], [1,1,1,0], [0,0,0,0], [0,1,1,0], [0,0,0,0]])),
        66: dict(zip(GLN_EMBS, [
            [1,0,0,1], [0,0,1,0], [0,0,0,0], [0,0,1,0], [0,1,1,1], [0,0,0,0]])),
    }),
    ("GLN", "G-ACE-8"): (GLN_EMBS, GLN_DIMS, 8, {
        44: dict(zip(GLN_EMBS, [
            [0,0,0,0,0,1,0,0], [0,1,0,0,0,0,0,0], [0,0,0,0,0,1,0,1],
            [1,0,1,1,1,1,1,1], [1,0,0,0,0,0,0,0], [1,0,0,0,0,1,0,1]])),
        55: dict(zip(GLN_EMBS, [
            [0,0,0,1,1,1,0,0], [1,0,1,0,0,0,1,0], [1,0,1,0,1,1,1,1],
            [0,0,1,1,1,1,1,1], [1,0,0,1,1,1,1,1], [0,0,1,1,1,1,1,1]])),
        66: dict(zip(GLN_EMBS, [
            [0,0,0,1,0,0,1,0], [0,1,1,1,1,1,0,1], [1,1,0,0,1,0,1,0],
            [1,1,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [0,1,1,0,0,1,1,0]])),
    }),
    ("GLN", "H-ACE-4"): (HACE_GLN_EMBS, HACE_GLN_DIMS, 4, {
        44: dict(zip(HACE_GLN_EMBS, [
            [0,0,1,0], [1,0,0,1], [1,1,0,0], [1,1,0,1]])),
        55: dict(zip(HACE_GLN_EMBS, [
            [1,0,1,1], [1,0,0,0], [1,0,0,0], [1,1,0,0]])),
        66: dict(zip(HACE_GLN_EMBS, [
            [1,1,0,1], [1,0,1,1], [1,1,1,1], [0,1,0,1]])),
    }),
    # ============ InLNER ============
    ("InLNER", "G-ACE-4"): (INL_EMBS, INL_DIMS, 4, {
        44: dict(zip(INL_EMBS, [
            [1,1,1,1], [1,1,1,1], [1,1,1,1], [0,0,1,0], [1,1,0,1], [1,1,1,1]])),
        55: dict(zip(INL_EMBS, [
            [0,0,1,1], [0,1,1,1], [0,0,1,0], [0,0,1,0], [1,1,0,0], [1,1,1,0]])),
        66: dict(zip(INL_EMBS, [
            [0,1,1,0], [0,0,0,0], [1,0,0,0], [0,0,1,0], [0,0,1,0], [1,1,1,0]])),
    }),
}

# Add InLNER G-ACE-8 and H-ACE-4 + LegalNERO from compute_selection_stability if they
# exist in the original DATA dict.  We do that below by importing.
import importlib.util, sys, os
_spec = importlib.util.spec_from_file_location(
    "css", os.path.join(os.path.dirname(__file__), "compute_selection_stability.py"))
_css = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_css)

# Translate the legacy DATA into our richer structure.
_DIMS_LOOKUP = {
    ("GLN", 6): GLN_DIMS, ("GLN", 4): HACE_GLN_DIMS,
    ("InLNER", 6): INL_DIMS, ("InLNER", 4): HACE_INL_DIMS,
    ("LegalNERO", 6): LNG_DIMS, ("LegalNERO", 4): HACE_LNG_DIMS,
}
_NAMES_LOOKUP = {
    ("GLN", 6): GLN_EMBS, ("GLN", 4): HACE_GLN_EMBS,
    ("InLNER", 6): INL_EMBS, ("InLNER", 4): HACE_INL_EMBS,
    ("LegalNERO", 6): LNG_EMBS, ("LegalNERO", 4): HACE_LNG_EMBS,
}
for (ds, method), seed_dict in _css.DATA.items():
    if (ds, method) in DATA: continue
    n_embs = len(next(iter(seed_dict.values())))
    n_groups = len(next(iter(next(iter(seed_dict.values())).values())))
    names = _NAMES_LOOKUP[(ds, n_embs)]
    dims  = _DIMS_LOOKUP[(ds, n_embs)]
    DATA[(ds, method)] = (names, dims, n_groups, seed_dict)

# ------------------------------------------------------------------
def selected_dims_per_emb(action_vec, perm, n_groups):
    chunks = perm.chunk(n_groups)
    out = set()
    for i, keep in enumerate(action_vec):
        if keep:
            out.update(chunks[i].tolist())
    return out

def jaccard(a, b):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)

def mean_pairwise(sets):
    pairs = list(combinations(sets, 2))
    if not pairs: return None
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs)

# ------------------------------------------------------------------
print("="*98)
print("Dim-level Jaccard stability (replaying training-time permutations)")
print("="*98)
print(f"{'Dataset':<10} {'Method':<10} {'#sd':>3}  {'index-J':>8}  {'dim-J':>8}  "
      f"{'random':>8}  {'Δ vs rnd':>9}  {'global-J':>9}")
print("-"*98)

results_table = []
for (ds, method), (names, dims, n_groups, seed_dict) in DATA.items():
    seeds = sorted(seed_dict.keys())
    if len(seeds) < 2:
        continue

    # Build per-seed permutations
    seed_perms = {}
    for s in seeds:
        rng = torch.Generator(); rng.manual_seed(int(s))
        seed_perms[s] = [torch.randperm(D, generator=rng) for D in dims]

    # Per-embedding stats
    per_emb_idx, per_emb_dim, per_emb_rnd = [], [], []
    global_dim_sets = {s: set() for s in seeds}
    offset = 0
    for ei, (name, D) in enumerate(zip(names, dims)):
        idx_sets, dim_sets, kept = [], [], []
        for s in seeds:
            a = seed_dict[s][name]
            idx_sets.append({i for i, v in enumerate(a) if v})
            dset = selected_dims_per_emb(a, seed_perms[s][ei], n_groups)
            dim_sets.append(dset)
            kept.append(sum(a))
            global_dim_sets[s].update(d + offset for d in dset)
        offset += D

        ji = mean_pairwise(idx_sets)
        jd = mean_pairwise(dim_sets)
        # random baseline (expected J for two random subsets of size m1,m2 of D)
        rnd_pairs = []
        for k1, k2 in combinations(kept, 2):
            m1 = k1 * D / n_groups
            m2 = k2 * D / n_groups
            inter = m1 * m2 / D if D else 0
            union = m1 + m2 - inter
            rnd_pairs.append(inter / union if union else 1.0)
        jr = sum(rnd_pairs) / len(rnd_pairs) if rnd_pairs else None
        per_emb_idx.append(ji); per_emb_dim.append(jd); per_emb_rnd.append(jr)

    # Aggregates
    avg_idx = sum(per_emb_idx)/len(per_emb_idx)
    avg_dim = sum(per_emb_dim)/len(per_emb_dim)
    avg_rnd = sum(per_emb_rnd)/len(per_emb_rnd)
    global_J = mean_pairwise([global_dim_sets[s] for s in seeds])

    print(f"{ds:<10} {method:<10} {len(seeds):>3}  {avg_idx:>8.3f}  {avg_dim:>8.3f}  "
          f"{avg_rnd:>8.3f}  {(avg_dim-avg_rnd):>+9.3f}  {global_J:>9.3f}")
    results_table.append((ds, method, len(seeds), avg_idx, avg_dim, avg_rnd, global_J))

print("-"*98)
print()
print("Legend:")
print("  index-J  = mean pairwise Jaccard over chunk-INDEX sets per embedding (= old J_blk)")
print("  dim-J    = mean pairwise Jaccard over ACTUAL DIM-INDEX sets (replaying permutation)")
print("  random   = expected Jaccard if dims were drawn uniformly at random of same size")
print("  Δ vs rnd = (dim-J - random) ; near 0 ⇒ selection is statistically random")
print("  global-J = Jaccard over the concatenated dim-mask of ALL embeddings together")
print()
print("Rule of thumb:")
print("  • Δ near 0   →  no specific dim is consistently selected → strong functional redundancy")
print("  • Δ > +0.10  →  algorithm is converging to specific dims across runs")
print()
print("="*98)
print("Coverage report:")
print("="*98)
have = {(ds, method) for (ds, method) in DATA}
expected = []
for ds in ("GLN", "InLNER", "LegalNERO"):
    for method in ("G-ACE-4", "G-ACE-8", "H-ACE-4"):
        expected.append((ds, method))
missing = [k for k in expected if k not in have]
print(f"  Have data for : {len(have)}/{len(expected)} (dataset, method) pairs")
for ds, m in expected:
    flag = "OK" if (ds, m) in have else "missing"
    n = len(DATA[(ds, m)][3]) if (ds, m) in DATA else 0
    print(f"    {ds:<10} {m:<10} : {flag}  (seeds={n})")
print()
print("Local taggers actually present (for re-verification of permutations):")
for d in sorted(os.listdir("resources/taggers")):
    p = os.path.join("resources/taggers", d, "best-model.pt")
    if os.path.exists(p): print(f"    ✓ resources/taggers/{d}/")
print()
print("Note: the dim-J computation above does NOT require the .pt files; it only")
print("replays torch.randperm(D, seed=group_seed) which is deterministic.")
