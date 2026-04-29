"""
collect_results.py
==================
Συλλέγει τα αποτελέσματα από τα training.log των Grouped-ACE πειραμάτων
και υπολογίζει mean ± std ανά τιμή N.

Χρήση:
    python collect_results.py
    python collect_results.py --pattern "inlner_grouped_N*_seed*"
    python collect_results.py --taggers_dir path/to/resources/taggers
"""

import os
import re
import math
import argparse
from collections import defaultdict


def parse_best_f1(log_path: str) -> float | None:
    """
    Διαβάζει ένα training.log και επιστρέφει το μέγιστο Test Average (= best F1).
    Γραμμές της μορφής:  ... Test Average: 87.08   Test avg loss: 4.10
    """
    pattern = re.compile(r"Test Average:\s*([\d.]+)")
    best = None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    val = float(m.group(1))
                    if best is None or val > best:
                        best = val
    except FileNotFoundError:
        pass
    return best


def parse_experiment_name(name: str):
    """
    Από το όνομα φακέλου π.χ. 'inlner_grouped_N4_seed2'
    επιστρέφει (prefix, N, seed) π.χ. ('inlner_grouped', 4, 2).
    Αν δεν ταιριάζει το pattern επιστρέφει None.
    """
    m = re.match(r"(.+)_N(\d+)_seed(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None


def mean_std(values: list[float]):
    n = len(values)
    if n == 0:
        return None, None
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    variance = sum((v - mu) ** 2 for v in values) / (n - 1)
    return mu, math.sqrt(variance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taggers_dir",
        default=os.path.join("resources", "taggers"),
        help="Path to the taggers directory (default: resources/taggers)",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional substring filter for experiment folder names (e.g. 'inlner_grouped')",
    )
    args = parser.parse_args()

    taggers_dir = args.taggers_dir
    if not os.path.isdir(taggers_dir):
        print(f"[ERROR] Directory not found: {taggers_dir}")
        return

    # Συλλογή αποτελεσμάτων
    # results[prefix][N] = list of (seed, f1)
    results: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    skipped = []

    for entry in sorted(os.listdir(taggers_dir)):
        if args.pattern and args.pattern not in entry:
            continue
        parsed = parse_experiment_name(entry)
        if parsed is None:
            continue
        prefix, N, seed = parsed
        log_path = os.path.join(taggers_dir, entry, "training.log")
        best_f1 = parse_best_f1(log_path)
        if best_f1 is None:
            skipped.append(entry)
            continue
        results[prefix][N].append((seed, best_f1))

    if not results:
        print("Δεν βρέθηκαν αποτελέσματα.")
        if skipped:
            print(f"Φάκελοι χωρίς έγκυρο log ({len(skipped)}): {skipped}")
        return

    # Εκτύπωση πίνακα ανά prefix
    for prefix in sorted(results.keys()):
        print(f"\n{'='*60}")
        print(f"  Dataset/Prefix: {prefix}")
        print(f"{'='*60}")
        print(f"  {'N':>4}  {'Seeds':>25}  {'Mean F1':>10}  {'Std':>8}  {'Max F1':>8}")
        print(f"  {'-'*4}  {'-'*25}  {'-'*10}  {'-'*8}  {'-'*8}")

        n_results = results[prefix]
        for N in sorted(n_results.keys()):
            runs = sorted(n_results[N], key=lambda x: x[0])  # sort by seed
            f1_values = [r[1] for r in runs]
            seeds_str = ", ".join(f"s{r[0]}={r[1]:.2f}" for r in runs)
            mu, std = mean_std(f1_values)
            best = max(f1_values)
            label = "N=1 (baseline)" if N == 1 else f"N={N}"
            print(f"  {label:>14}  {seeds_str:>25}  {mu:>9.2f}%  {std:>7.2f}%  {best:>7.2f}%")

    if skipped:
        print(f"\n[SKIP] {len(skipped)} φάκελοι χωρίς αποτελέσματα ακόμα: {skipped}")

    # Αποθήκευση σε CSV
    csv_path = "grouped_ace_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("prefix,N,seed,best_f1\n")
        for prefix in sorted(results.keys()):
            for N in sorted(results[prefix].keys()):
                for seed, f1 in sorted(results[prefix][N]):
                    f.write(f"{prefix},{N},{seed},{f1:.4f}\n")
    print(f"\n[CSV] Αποθηκεύτηκε: {csv_path}")


if __name__ == "__main__":
    main()
