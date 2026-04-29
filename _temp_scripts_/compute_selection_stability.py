"""Compute selection-stability statistics across seeds and emit LaTeX rows."""
from itertools import combinations

# Action vectors per (dataset, method, seed).
# Order of embeddings: domain-specific tuple per dataset (canonical embedding list).
# Each value = list of 0/1 of length N (groups per embedding).

# embedding-name canonical lists per dataset
GLN_EMBS = ["GreekBERT", "GreekLegalRoBERTa", "FastText", "mDeBERTa", "Char", "BPEmb"]
INL_EMBS = ["InLegalBERT", "LegalBERT", "FastText", "mDeBERTa", "Char", "BPEmb"]
LNG_EMBS = ["FastText", "LegalRoBERTa", "RomanianBERT", "mDeBERTa", "Char", "BPEmb"]

# H-ACE upper-level kept 4 embeddings (subset of canonical 6)
HACE_GLN_EMBS = ["GreekBERT", "GreekLegalRoBERTa", "mDeBERTa", "Char"]
HACE_INL_EMBS = ["InLegalBERT", "FastText", "Char", "BPEmb"]
HACE_LNG_EMBS = ["LegalRoBERTa", "RomanianBERT", "mDeBERTa", "Char"]

# Action vectors transcribed from MD files (per Best Model Action section).
DATA = {
    # ============ GLN ============
    ("GLN", "G-ACE-4"): {
        # seed44 (BPEmb=300 canonical config)
        44: dict(zip(GLN_EMBS, [
            [0,0,1,0], [1,1,0,1], [1,1,1,0], [1,1,0,1], [0,0,0,1], [0,0,1,0]])),
        # seed44 alternative run (run1)
        441: dict(zip(GLN_EMBS, [
            [1,0,0,0], [0,0,0,1], [1,1,1,0], [0,0,0,0], [0,1,1,0], [0,0,0,0]])),
        66: dict(zip(GLN_EMBS, [
            [1,0,0,1], [0,0,1,0], [0,0,0,0], [0,0,1,0], [0,1,1,1], [0,0,0,0]])),
    },
    ("GLN", "G-ACE-8"): {
        44: dict(zip(GLN_EMBS, [
            [0,0,0,0,0,1,0,0], [0,1,0,0,0,0,0,0], [0,0,0,0,0,1,0,1],
            [1,0,1,1,1,1,1,1], [1,0,0,0,0,0,0,0], [1,0,0,0,0,1,0,1]])),
        55: dict(zip(GLN_EMBS, [
            [0,0,0,1,1,1,0,0], [1,0,1,0,0,0,1,0], [1,0,1,0,1,1,1,1],
            [0,0,1,1,1,1,1,1], [1,0,0,1,1,1,1,1], [0,0,1,1,1,1,1,1]])),
        66: dict(zip(GLN_EMBS, [
            [0,0,0,1,0,0,1,0], [0,1,1,1,1,1,0,1], [1,1,0,0,1,0,1,0],
            [1,1,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [0,1,1,0,0,1,1,0]])),
    },
    ("GLN", "H-ACE-4"): {
        44: dict(zip(HACE_GLN_EMBS, [
            [0,0,1,0], [1,0,0,1], [1,1,0,0], [1,1,0,1]])),
        55: dict(zip(HACE_GLN_EMBS, [
            [1,0,1,1], [1,0,0,0], [1,0,0,0], [1,1,0,0]])),
        66: dict(zip(HACE_GLN_EMBS, [
            [1,1,0,1], [1,0,1,1], [1,1,1,1], [0,1,0,1]])),
    },
    # ============ InLNER ============
    ("InLNER", "G-ACE-4"): {
        44: dict(zip(INL_EMBS, [
            [1,1,1,1], [1,1,1,1], [1,1,1,1], [0,0,1,0], [1,1,0,1], [1,1,1,1]])),
        55: dict(zip(INL_EMBS, [
            [0,0,1,1], [0,1,1,1], [0,0,1,0], [0,0,1,0], [1,1,0,0], [1,1,1,0]])),
        66: dict(zip(INL_EMBS, [
            [0,1,1,0], [0,0,0,0], [1,0,0,0], [0,0,1,0], [0,0,1,0], [1,1,1,0]])),
    },
    ("InLNER", "G-ACE-8"): {
        44: dict(zip(INL_EMBS, [
            [1,1,0,1,1,0,0,1], [1,0,1,0,0,0,1,0], [0,1,1,1,0,1,0,1],
            [0,0,1,0,1,1,0,0], [1,1,0,0,1,1,1,0], [1,1,0,1,1,1,1,0]])),
        55: dict(zip(INL_EMBS, [
            [1,0,0,0,1,0,0,0], [1,1,1,1,0,0,1,1], [0,0,1,1,1,1,1,0],
            [1,0,1,1,0,0,1,1], [1,0,0,0,1,1,1,1], [0,1,1,0,1,1,1,0]])),
        66: dict(zip(INL_EMBS, [
            [0,0,0,1,0,1,1,1], [0,0,0,1,0,1,0,0], [1,1,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0], [0,0,1,0,0,1,1,1], [0,0,1,1,1,0,0,0]])),
    },
    ("InLNER", "H-ACE-4"): {
        44: dict(zip(HACE_INL_EMBS, [
            [0,1,0,0], [1,1,0,1], [0,1,1,0], [1,1,1,0]])),
        55: dict(zip(HACE_INL_EMBS, [
            [0,1,0,0], [1,1,0,1], [1,1,1,1], [1,0,0,0]])),
        66: dict(zip(HACE_INL_EMBS, [
            [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]])),
    },
    # ============ LegalNERO ============
    ("LegalNERO", "G-ACE-4"): {
        44: dict(zip(LNG_EMBS, [
            [0,0,0,1], [0,0,0,1], [0,1,0,1], [0,0,0,0], [0,1,0,1], [0,0,1,0]])),
        55: dict(zip(LNG_EMBS, [
            [0,0,0,0], [1,1,0,0], [1,1,0,0], [0,1,0,0], [0,1,0,0], [1,1,1,0]])),
        66: dict(zip(LNG_EMBS, [
            [0,1,0,0], [1,1,1,0], [1,1,1,0], [0,0,0,0], [1,0,0,1], [1,1,0,1]])),
    },
    ("LegalNERO", "G-ACE-8"): {
        44: dict(zip(LNG_EMBS, [
            [1,1,1,0,0,0,0,1], [1,0,1,0,0,1,1,0], [1,0,0,1,0,1,0,0],
            [0,0,0,1,0,0,0,1], [0,1,0,0,1,0,0,0], [1,1,0,0,0,0,0,0]])),
        55: dict(zip(LNG_EMBS, [
            [1,1,0,0,1,0,0,0], [0,1,0,1,1,0,1,1], [1,0,1,0,0,1,1,1],
            [0,0,0,0,0,0,0,1], [1,1,0,0,1,1,0,0], [0,1,0,0,1,1,1,0]])),
        66: dict(zip(LNG_EMBS, [
            [0,1,1,1,1,1,1,0], [0,1,1,0,0,1,0,0], [1,0,0,0,0,1,1,1],
            [0,0,1,1,1,0,0,0], [1,0,0,0,1,0,1,1], [0,1,0,0,1,1,0,0]])),
    },
    ("LegalNERO", "H-ACE-4"): {
        44: dict(zip(HACE_LNG_EMBS, [
            [0,1,0,1], [1,1,0,0], [1,0,1,0], [0,1,1,1]])),
        55: dict(zip(HACE_LNG_EMBS, [
            [0,0,1,1], [1,0,0,0], [1,0,0,0], [1,1,1,1]])),
        66: dict(zip(HACE_LNG_EMBS, [
            [1,0,1,0], [1,0,1,1], [1,0,0,1], [0,1,1,1]])),
    },
}

# Need to fetch LegalNERO H-ACE seed44 action — placeholder above.

def emb_set(seed_action):
    """Embeddings with at least one group selected."""
    return frozenset(e for e, gs in seed_action.items() if any(gs))

def group_set(seed_action):
    """Set of (embedding, group_idx) with group selected."""
    return frozenset((e, i) for e, gs in seed_action.items() for i, v in enumerate(gs) if v)

def jaccard(a, b):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0

def mean_pairwise_jaccard(sets):
    if len(sets) < 2: return None
    return sum(jaccard(a, b) for a, b in combinations(sets, 2)) / sum(1 for _ in combinations(sets, 2))

def core_volatile(actions):
    """Return (core_embs, volatile_embs) across seeds.
    core: in every seed; volatile: in some but not all."""
    seed_emb_sets = [emb_set(a) for a in actions.values()]
    if not seed_emb_sets: return set(), set()
    core = set.intersection(*[set(s) for s in seed_emb_sets])
    union = set.union(*[set(s) for s in seed_emb_sets])
    return core, union - core

# Print summary
print(f"{'Dataset':<10} {'Method':<8} {'#seeds':>6} {'JaccEmb':>8} {'JaccGrp':>8}  {'Core':<40}")
print("-"*92)
rows = []
for (ds, method), actions in DATA.items():
    n = len(actions)
    e_sets = [emb_set(a) for a in actions.values()]
    g_sets = [group_set(a) for a in actions.values()]
    je = mean_pairwise_jaccard(e_sets)
    jg = mean_pairwise_jaccard(g_sets)
    core, vol = core_volatile(actions)
    je_s = f"{je:.2f}" if je is not None else "—"
    jg_s = f"{jg:.2f}" if jg is not None else "—"
    print(f"{ds:<10} {method:<8} {n:>6} {je_s:>8} {jg_s:>8}  {','.join(sorted(core)):<40}")
    rows.append((ds, method, n, je, jg, core, vol))

print()
print("Volatile embeddings (in some seeds but not all):")
for (ds, method, n, je, jg, core, vol) in rows:
    if n >= 2:
        print(f"  {ds:<10} {method:<8}: {sorted(vol) if vol else '—'}")

print()
print("Average per-seed dim usage and selected-group counts:")
import torch
def chunk(d,n): return [t.shape[0] for t in torch.zeros(d).chunk(n)]
DIMS_GLN = {"GreekBERT":768,"GreekLegalRoBERTa":768,"FastText":300,"mDeBERTa":768,"Char":50,"BPEmb":300}
DIMS_INL = {"InLegalBERT":768,"LegalBERT":768,"FastText":300,"mDeBERTa":768,"Char":50,"BPEmb":300}
DIMS_LNG = {"FastText":300,"LegalRoBERTa":768,"RomanianBERT":768,"mDeBERTa":768,"Char":50,"BPEmb":300}
DIMS = {"GLN":DIMS_GLN,"InLNER":DIMS_INL,"LegalNERO":DIMS_LNG}
def dims_used(ds, action):
    dims_map = DIMS[ds]
    total = 0
    for emb, mask in action.items():
        total += sum(c for c,m in zip(chunk(dims_map[emb], len(mask)), mask) if m)
    return total
import statistics
for (ds, method), actions in DATA.items():
    duses = [dims_used(ds, a) for a in actions.values()]
    grp_kept = [sum(sum(g) for g in a.values()) for a in actions.values()]
    grp_tot  = sum(len(g) for g in next(iter(actions.values())).values())
    print(f"  {ds:<10} {method:<8} dims={statistics.mean(duses):>6.0f} ±{(statistics.pstdev(duses) if len(duses)>1 else 0):>5.0f}   groups={statistics.mean(grp_kept):>4.1f}/{grp_tot}  ±{(statistics.pstdev(grp_kept) if len(grp_kept)>1 else 0):.1f}")
