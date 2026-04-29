"""Compute dim-level Jaccard stability across seeds for GLN G-ACE-8 / H-ACE-4.

Replicates the per-embedding random permutation used at training time
(torch.randperm seeded with `group_seed`), expands each (embedding, seed)
action vector to the *actual* dim indices selected, and computes
mean pairwise Jaccard over those true dim sets.

Compares index-level vs dim-level Jaccard to test whether the chunk
randomization inflates apparent block volatility.
"""
from itertools import combinations
import torch

# --- canonical GLN embeddings (sorted alphabetically by full name; matches runtime) ---
GLN_FULL_NAMES = [
    "AIAI/GREEKLEGALNERV2/GreekBERT/...",            # GreekBERT
    "AIAI/GREEKLEGALNERV2/GreekLegalRoBERTa/...",    # GreekLegalRoBERTa
    "AIAI/GREEKLEGALNERV2/cc.el.300.vec",            # FastText
    "AIAI/GREEKLEGALNERV2/mdeberta-v3-base/seed_1",  # mDeBERTa
    "Char",                                           # Char
    "bpe-el-100000-300",                              # BPEmb
]
GLN_SHORT = ["GreekBERT", "GreekLegalRoBERTa", "FastText", "mDeBERTa", "Char", "BPEmb"]
GLN_DIMS  = [768,         768,                300,         768,        50,     600]

# --- action vectors: copied from compute_selection_stability.py ---
GLN_GACE8 = {
    44: dict(zip(GLN_SHORT, [
        [0,0,0,0,0,1,0,0], [0,1,0,0,0,0,0,0], [0,0,0,0,0,1,0,1],
        [1,0,1,1,1,1,1,1], [1,0,0,0,0,0,0,0], [1,0,0,0,0,1,0,1]])),
    55: dict(zip(GLN_SHORT, [
        [0,0,0,1,1,1,0,0], [1,0,1,0,0,0,1,0], [1,0,1,0,1,1,1,1],
        [0,0,1,1,1,1,1,1], [1,0,0,1,1,1,1,1], [0,0,1,1,1,1,1,1]])),
    66: dict(zip(GLN_SHORT, [
        [0,0,0,1,0,0,1,0], [0,1,1,1,1,1,0,1], [1,1,0,0,1,0,1,0],
        [1,1,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [0,1,1,0,0,1,1,0]])),
}

N_GROUPS = 8

def selected_dims_per_emb(action_vec, perm, n_groups):
    """Return set of true dim indices selected by this action."""
    chunks = perm.chunk(n_groups)
    out = set()
    for i, keep in enumerate(action_vec):
        if keep:
            out.update(chunks[i].tolist())
    return out

def jaccard(a, b):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)

def mean_pairwise(seed_sets):
    pairs = list(combinations(seed_sets, 2))
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs) if pairs else None

def index_level_jaccard(action_seed_a, action_seed_b):
    """Old metric: Jaccard over (group_idx) sets, ignoring permutations."""
    sa = {i for i, v in enumerate(action_seed_a) if v}
    sb = {i for i, v in enumerate(action_seed_b) if v}
    return jaccard(sa, sb)

# --- main ---
print(f"GLN G-ACE-8 (N={N_GROUPS}, seeds={sorted(GLN_GACE8.keys())})")
print(f"{'Embedding':<22} {'D':>5}  {'index-J':>9}  {'dim-J':>9}  {'random':>9}  {'Δ (dim-rnd)':>12}")
print("-"*82)

# Build per-seed permutations (deterministic from group_seed)
seed_perms = {}
for seed in GLN_GACE8:
    perms = []
    rng = torch.Generator(); rng.manual_seed(seed)
    for D in GLN_DIMS:
        perms.append(torch.randperm(D, generator=rng))
    seed_perms[seed] = perms

# For each embedding compute both metrics + a random baseline (expected J for
# selecting the same #dims uniformly at random from D)
all_dim_sets = {s: set() for s in GLN_GACE8}  # global concat across embs
total_idx_pairs, total_idx_sum = 0, 0.0
total_dim_pairs, total_dim_sum = 0, 0.0
total_rnd_sum, total_rnd_pairs = 0.0, 0

offset = 0
for ei, (name, D) in enumerate(zip(GLN_SHORT, GLN_DIMS)):
    # collect per-seed sets
    idx_sets, dim_sets, kept_counts = [], [], []
    for s in sorted(GLN_GACE8):
        a = GLN_GACE8[s][name]
        idx_sets.append({i for i, v in enumerate(a) if v})
        dims = selected_dims_per_emb(a, seed_perms[s][ei], N_GROUPS)
        dim_sets.append(dims)
        kept_counts.append(sum(a))
        all_dim_sets[s].update(d + offset for d in dims)

    j_idx = mean_pairwise(idx_sets)
    j_dim = mean_pairwise(dim_sets)
    # random baseline: expected Jaccard for two random subsets of size k1,k2 of D
    # E[|A∩B|] = k1*k2/D ; E[|A∪B|] ≈ k1+k2 - k1*k2/D
    rnd = []
    for (k1, k2) in combinations(kept_counts, 2):
        # convert kept groups to expected #dims = k * D / N
        m1 = k1 * D / N_GROUPS
        m2 = k2 * D / N_GROUPS
        inter = m1 * m2 / D if D else 0
        union = m1 + m2 - inter
        rnd.append(inter / union if union else 1.0)
    j_rnd = sum(rnd)/len(rnd)
    print(f"{name:<22} {D:>5}  {j_idx:>9.3f}  {j_dim:>9.3f}  {j_rnd:>9.3f}  {(j_dim-j_rnd):>+12.3f}")
    total_idx_sum += j_idx; total_idx_pairs += 1
    total_dim_sum += j_dim; total_dim_pairs += 1
    total_rnd_sum += j_rnd; total_rnd_pairs += 1
    offset += D

# global Jaccard (concat all embeddings) on dim sets
all_seeds = sorted(all_dim_sets)
g_pairs = list(combinations([all_dim_sets[s] for s in all_seeds], 2))
g_dim_j = sum(jaccard(a,b) for a,b in g_pairs) / len(g_pairs)
print("-"*82)
print(f"{'AVG (per-emb)':<22} {'':>5}  {total_idx_sum/total_idx_pairs:>9.3f}  "
      f"{total_dim_sum/total_dim_pairs:>9.3f}  {total_rnd_sum/total_rnd_pairs:>9.3f}")
print(f"{'GLOBAL concat':<22} {sum(GLN_DIMS):>5}  {'—':>9}  {g_dim_j:>9.3f}  {'—':>9}")
print()
print("Interpretation:")
print(" - 'index-J' = current J_blk metric (over chunk indices, ignores permutation).")
print(" - 'dim-J'   = Jaccard over the actual selected dim indices after permutation.")
print(" - 'random'  = expected Jaccard for two uniformly random subsets of the same size.")
print(" - dim-J - random ≈ 0  →  selection is essentially indistinguishable from random.")
print(" - dim-J >> random  →  the algorithm consistently picks the SAME dims across seeds.")
