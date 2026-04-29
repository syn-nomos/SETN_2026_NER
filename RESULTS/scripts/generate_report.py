#!/usr/bin/env python3
"""
Generate a Markdown report from an ACE / Grouped-ACE training log.

Usage:
    python generate_report.py <training.log> [--config <config.yaml>] [-o <output.md>]

If --config is given, extra info (hyperparameters, grouped-ACE settings) is
pulled from the YAML.  If -o is omitted, the report is printed to stdout.
"""
import argparse
import re
import sys
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------
RE_CORPUS = re.compile(r'Corpus:\s*"Corpus:\s*(\d+)\s*train\s*\+\s*(\d+)\s*dev\s*\+\s*(\d+)\s*test\s*sentences"')
RE_EPISODE_START = re.compile(r'=+ Start episode (\d+) =+')
RE_EPISODE_ACTION = re.compile(r'Episode (\d+) action \(num_groups=(\d+)\):')
RE_EMB_LINE = re.compile(r'^\s*(.*?)\s+groups=\[([01,\s]+)\]\s+kept=(\d+)/(\d+)')
RE_EPOCH_DONE = re.compile(r'EPISODE (\d+), EPOCH (\d+) done: loss ([\d.]+) - lr ([\d.e-]+)')
RE_TEST_AVG = re.compile(r'Test Average:\s*([\d.]+)')
RE_BASELINE_SCORE = re.compile(r'Setting baseline score to:\s*([\d.]+)')
RE_BASELINE_ACTION = re.compile(r'Setting baseline action \(num_groups=(\d+)\):')
RE_BEST_SAVE = re.compile(r'Saving the current overall best model:\s*([\d.]+)')
RE_MICRO = re.compile(r'MICRO_AVG:\s*acc\s*([\d.]+)\s*-\s*f1-score\s*([\d.]+)')
RE_MACRO = re.compile(r'MACRO_AVG:\s*acc\s*([\d.]+)\s*-\s*f1-score\s*([\d.]+)')
RE_ENTITY = re.compile(
    r'^(\S[\S ]*?)\s+tp:\s*(\d+)\s*-\s*fp:\s*(\d+)\s*-\s*fn:\s*(\d+)\s*-\s*tn:\s*\d+\s*-\s*'
    r'precision:\s*([\d.]+)\s*-\s*recall:\s*([\d.]+)\s*-\s*accuracy:\s*[\d.]+\s*-\s*f1-score:\s*([\d.]+)'
)
RE_FINAL_MASK = re.compile(r'Setting embedding mask to the best action \(num_groups=(\d+)\):')
RE_TIMESTAMP = re.compile(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})')


# ---------------------------------------------------------------------------
# Short name mapping for common embedding paths
# Rules are checked in order; first match wins.
# Each rule is (nice_name, list_of_substrings_that_must_appear_in_lowercased_path).
# ---------------------------------------------------------------------------
SHORT_NAME_RULES = [
    # ── LegalNERO-specific ──────────────────────────────────────────────────
    ('LegalRoBERTa',       ['legalromanianoroberta']),
    ('LegalRoBERTa',       ['legal-romanian-roberta']),
    ('RomanianBERT',       ['romanianbert', 'bert-base-romanian']),
    ('GreekLegalRoBERTa',  ['greeklegalroberta', 'legal-greek-roberta']),
    # ── InLNER-specific ─────────────────────────────────────────────────────
    ('InLegalBERT',        ['inlegalbert', 'law-ai/inlegalbert']),
    ('LegalBERT',          ['nlpaueb/legal-bert', 'legal-bert-base']),
    ('XLM-R-Base',         ['xlm-roberta-base', 'xlm-r-base']),
    ('DistilBERT',         ['distilbert']),
    # ── GLN-specific ────────────────────────────────────────────────────────
    ('GreekBERT',          ['greekbert', 'bert-base-greek-uncased']),
    ('GreekLegalNER',      ['greeklegalner', 'greek_legal_ner']),
    # ── Generic transformer models ──────────────────────────────────────────
    ('mDeBERTa-v3',        ['mdeberta', 'deberta']),
    # ── Non-contextual ──────────────────────────────────────────────────────
    ('GloVe',              ['glove']),
    ('Word2Vec',           ['word2vec']),
    ('FastText',           ['fasttext']),
    ('FastText',           ['cc.ro.']),
    ('FastText',           ['cc.en.']),
    ('BPEmb',              ['bpe-']),
    ('BPEmb',              ['bpemb']),
    ('BPEmb',              ['bytepair']),
    ('FastCharEmbeddings', ['fastchar']),
    ('FastCharEmbeddings', ['char']),
    ('FlairEmbeddings',    ['flair']),
]


def short_name(raw: str) -> str:
    """Map a raw embedding path / id to a human-readable short name."""
    low = raw.lower().strip().replace('\\', '/')
    for nice, patterns in SHORT_NAME_RULES:
        if all(p in low for p in patterns):
            return nice
    # Fallback: last meaningful path segment (skip 'seed_N' and checkpoint dirs)
    parts = [p for p in raw.replace('\\', '/').split('/') if p.strip()]
    # Walk from deepest to find a meaningful token
    skip = {'seed_1', 'seed_2', 'seed_3', 'results_after_training_only_on_language_with_id__en',
            'results_after_training_only_on_language_with_id__ro',
            'results_after_training_only_on_language_with_id__all'}
    for part in reversed(parts):
        if part.lower() not in skip and not part.lower().startswith('checkpoint-'):
            return part
    return parts[-1] if parts else raw


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------
class LogParser:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.lines = Path(log_path).read_text(errors='replace').splitlines()

    def parse(self) -> dict:
        data = {
            'corpus': {},
            'episodes': [],       # list of dicts
            'baselines': [],      # list of (episode, score, action_groups)
            'final_test': {},     # micro, macro, entities
            'best_action': {},    # embedding_name -> groups list
            'emb_names': [],      # ordered list of short names
            'raw_emb_names': [],  # ordered list of raw names  
            'num_groups': 1,
            'start_time': None,
            'end_time': None,
            'model_dir': str(Path(self.log_path).parent),
        }

        cur_ep = None
        cur_ep_data = None
        collecting_action = None   # 'episode' | 'baseline' | 'final_mask'
        action_lines = []
        baseline_score = None
        last_test_f1 = None
        best_dev_f1 = 0.0       # best DEV F1 seen (used for global best model tracking)
        best_global_test_f1 = 0.0  # test F1 at the globally best save
        best_episode = 0
        epoch_count = 0
        total_epochs = 0
        emb_order_set = False

        for line in self.lines:
            # Timestamps
            ts_m = RE_TIMESTAMP.match(line)
            if ts_m:
                ts = ts_m.group(1)
                if data['start_time'] is None:
                    data['start_time'] = ts
                data['end_time'] = ts

            # Corpus
            m = RE_CORPUS.search(line)
            if m:
                data['corpus'] = {
                    'train': int(m.group(1)),
                    'dev': int(m.group(2)),
                    'test': int(m.group(3)),
                }

            # Episode start
            m = RE_EPISODE_START.search(line)
            if m:
                # Flush any pending action block before changing episode
                if collecting_action and action_lines:
                    groups_dict = OrderedDict()
                    for raw, groups, kept, total in action_lines:
                        sn = short_name(raw)
                        groups_dict[sn] = groups
                        if not emb_order_set:
                            data['emb_names'].append(sn)
                            data['raw_emb_names'].append(raw)
                    if not emb_order_set and data['emb_names']:
                        emb_order_set = True
                    if collecting_action == 'episode' and cur_ep_data is not None:
                        cur_ep_data['action_groups'] = groups_dict
                    elif collecting_action == 'baseline':
                        if baseline_score is not None:
                            data['baselines'].append({
                                'episode': cur_ep,
                                'score': baseline_score,
                                'action_groups': groups_dict,
                            })
                    elif collecting_action == 'final_mask':
                        data['best_action'] = groups_dict
                    collecting_action = None
                    action_lines = []

                ep_num = int(m.group(1))
                # Save previous episode
                if cur_ep_data is not None:
                    cur_ep_data['epochs'] = epoch_count
                    data['episodes'].append(cur_ep_data)
                    total_epochs += epoch_count
                cur_ep = ep_num
                epoch_count = 0
                cur_ep_data = {
                    'episode': ep_num,
                    'epochs': 0,
                    'best_test_f1': 0.0,
                    'is_baseline': False,
                    'action_groups': {},
                }
                last_test_f1 = None

            # Episode action header
            m = RE_EPISODE_ACTION.search(line)
            if m:
                data['num_groups'] = int(m.group(2))
                collecting_action = 'episode'
                action_lines = []
                continue

            # Baseline action header
            m = RE_BASELINE_ACTION.search(line)
            if m:
                collecting_action = 'baseline'
                action_lines = []
                continue

            # Final mask header
            m = RE_FINAL_MASK.search(line)
            if m:
                collecting_action = 'final_mask'
                action_lines = []
                continue

            # Embedding lines (for any action block)
            if collecting_action:
                # Always strip timestamp first
                stripped = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s*', '', line)
                m2 = RE_EMB_LINE.match(stripped)
                if m2:
                    raw = m2.group(1).strip()
                    groups = [int(x.strip()) for x in m2.group(2).split(',')]
                    kept = int(m2.group(3))
                    total = int(m2.group(4))
                    action_lines.append((raw, groups, kept, total))
                    continue
                else:
                    # End of embedding block
                    if action_lines:
                        groups_dict = OrderedDict()
                        for raw, groups, kept, total in action_lines:
                            sn = short_name(raw)
                            groups_dict[sn] = groups
                            if not emb_order_set:
                                data['emb_names'].append(sn)
                                data['raw_emb_names'].append(raw)

                        if not emb_order_set and data['emb_names']:
                            emb_order_set = True

                        if collecting_action == 'episode' and cur_ep_data is not None:
                            cur_ep_data['action_groups'] = groups_dict
                        elif collecting_action == 'baseline':
                            if baseline_score is not None:
                                data['baselines'].append({
                                    'episode': cur_ep,
                                    'score': baseline_score,
                                    'action_groups': groups_dict,
                                })
                        elif collecting_action == 'final_mask':
                            data['best_action'] = groups_dict

                    collecting_action = None
                    action_lines = []

            # Baseline score
            m = RE_BASELINE_SCORE.search(line)
            if m:
                baseline_score = float(m.group(1))
                if cur_ep_data is not None:
                    cur_ep_data['is_baseline'] = True

            # Epoch done
            m = RE_EPOCH_DONE.search(line)
            if m:
                epoch_count = int(m.group(2))

            # Test average per epoch
            m = RE_TEST_AVG.search(line)
            if m:
                last_test_f1 = float(m.group(1))
                # Don't update cur_ep_data['best_test_f1'] here; update it on each save instead
                # so we show the test F1 at the best-dev epoch (last save), not max test F1

            # Best model save — score here is the DEV F1 (Macro Avg), not test F1
            m = RE_BEST_SAVE.search(line)
            if m:
                dev_score = float(m.group(1))
                # Update per-episode test F1: overwrite with latest save's test F1
                # (last save = highest dev = most meaningful test F1 for this episode)
                if cur_ep_data is not None and last_test_f1 is not None:
                    cur_ep_data['best_test_f1'] = last_test_f1
                # Track global best dev and corresponding test F1
                if dev_score > best_dev_f1:
                    best_dev_f1 = dev_score
                    best_episode = cur_ep
                    best_global_test_f1 = last_test_f1 if last_test_f1 is not None else 0.0

            # MICRO_AVG (keep the last occurrence = final test)
            m = RE_MICRO.search(line)
            if m:
                data['final_test']['micro_acc'] = float(m.group(1))
                data['final_test']['micro_f1'] = float(m.group(2))

            # MACRO_AVG
            m = RE_MACRO.search(line)
            if m:
                data['final_test']['macro_acc'] = float(m.group(1))
                data['final_test']['macro_f1'] = float(m.group(2))

            # Per-entity (use a dict keyed by name to deduplicate — last occurrence wins)
            stripped_ent = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s*', '', line).strip()
            m = RE_ENTITY.match(stripped_ent)
            if m:
                ent_name = m.group(1).strip()
                if ent_name not in ('MICRO_AVG', 'MACRO_AVG'):
                    ent_entry = {
                        'name': ent_name,
                        'tp': int(m.group(2)),
                        'fp': int(m.group(3)),
                        'fn': int(m.group(4)),
                        'precision': float(m.group(5)),
                        'recall': float(m.group(6)),
                        'f1': float(m.group(7)),
                    }
                    data['final_test'].setdefault('_entity_dict', OrderedDict())[ent_name] = ent_entry

        # Save last episode
        if cur_ep_data is not None:
            cur_ep_data['epochs'] = epoch_count
            data['episodes'].append(cur_ep_data)
            total_epochs += epoch_count

        # Convert entity dict to deduplicated list
        entity_dict = data['final_test'].pop('_entity_dict', OrderedDict())
        data['final_test']['entities'] = list(entity_dict.values())

        data['total_epochs'] = total_epochs
        data['best_episode'] = best_episode
        data['best_dev_f1'] = best_dev_f1          # DEV F1 at global best save
        data['best_global_test_f1'] = best_global_test_f1  # Test F1 at global best save

        # Duration
        if data['start_time'] and data['end_time']:
            try:
                fmt = '%Y-%m-%d %H:%M:%S'
                t0 = datetime.strptime(data['start_time'], fmt)
                t1 = datetime.strptime(data['end_time'], fmt)
                delta = t1 - t0
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                minutes = remainder // 60
                data['duration_str'] = f'~{hours}h {minutes:02d}m'
            except Exception:
                data['duration_str'] = 'N/A'
        else:
            data['duration_str'] = 'N/A'

        return data


# ---------------------------------------------------------------------------
# Config parser (optional YAML)
# ---------------------------------------------------------------------------
def parse_config(config_path: str) -> dict:
    if yaml is None:
        print("WARNING: PyYAML not installed — skipping config parsing.", file=sys.stderr)
        return {}
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------
def extract_seed_from_path(model_dir: str) -> str:
    """Try to extract seed from directory name like gln_grouped_N4_original_seed66."""
    m = re.search(r'seed(\d+)', model_dir)
    return m.group(1) if m else '?'


def extract_experiment_name(model_dir: str) -> str:
    """Derive a human title from the model directory name."""
    base = Path(model_dir).name
    # Try to detect grouped vs vanilla
    if 'grouped' in base.lower():
        m_n = re.search(r'N(\d+)', base)
        n = m_n.group(1) if m_n else '?'
        seed = extract_seed_from_path(base)
        return f'Grouped-ACE (N={n})', seed
    else:
        seed = extract_seed_from_path(base)
        return 'ACE', seed


def generate_report(data: dict, cfg: dict = None) -> str:
    """Generate a Markdown report string from parsed data."""
    lines = []
    w = lines.append  # shortcut

    exp_name, seed = extract_experiment_name(data['model_dir'])
    num_groups = data['num_groups']
    num_emb = len(data['emb_names'])
    num_actions = num_emb * num_groups

    # Determine dataset name from config or path
    dataset_name = 'NER'
    if cfg:
        for key in cfg:
            if key == 'ner' and isinstance(cfg[key], dict):
                for sub in cfg[key].values():
                    if isinstance(sub, dict) and 'data_folder' in sub:
                        dataset_name = sub['data_folder'].split('/')[-1]
                        break

    micro_f1 = data['final_test'].get('micro_f1', 0)
    macro_f1 = data['final_test'].get('macro_f1', 0)
    micro_acc = data['final_test'].get('micro_acc', 0)

    # ── Title ──
    w(f'# {exp_name} — {dataset_name}: Seed {seed}\n')

    # ── 1. Summary ──
    w('## 1. Σύνοψη\n')
    w('| Μετρική              | Τιμή       |')
    w('|----------------------|------------|')
    w(f'| **Micro F1 (Test)**  | **{micro_f1*100:.2f}%** |')
    w(f'| **Macro F1 (Test)**  | **{macro_f1*100:.2f}%** |')

    # Precision / Recall from entity sums
    entities = data['final_test'].get('entities', [])
    if entities:
        total_tp = sum(e['tp'] for e in entities)
        total_fp = sum(e['fp'] for e in entities)
        total_fn = sum(e['fn'] for e in entities)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
        w(f'| Precision (Micro)    | {precision*100:.2f}%     |')
        w(f'| Recall (Micro)       | {recall*100:.2f}%     |')
    w(f'| Accuracy (Micro)     | {micro_acc*100:.2f}%     |')
    w('')

    best_ep = data.get('best_episode', '?')
    best_dev_f1_val = data.get('best_dev_f1', 0.0)
    best_ep_test_f1 = data.get('best_global_test_f1', micro_f1 * 100)
    w(f'**Μέθοδος:** {exp_name} (N={num_groups} groups / embedding, equal mode, seed {seed})  ')
    w(f'**Best Model Save:** Episode {best_ep}, Dev F1 = {best_dev_f1_val:.2f}% (Test F1 = {best_ep_test_f1:.2f}%)  ')
    w(f'**Τελική Αξιολόγηση (best-model.pt):** Test F1 = {micro_f1*100:.2f}%  ')
    w(f'**Διάρκεια Εκπαίδευσης:** {data["duration_str"]} ({len(data["episodes"])} episodes)')
    w('')
    w('---\n')

    # ── 2. Configuration ──
    w('## 2. Ρυθμίσεις Εκπαίδευσης (Configuration)\n')

    if cfg:
        # Model architecture
        model_cfg = cfg.get('model', {})
        model_key = list(model_cfg.keys())[0] if model_cfg else 'FastSequenceTagger'
        model_params = model_cfg.get(model_key, {}) if model_cfg else {}

        w('### Αρχιτεκτονική Μοντέλου\n')
        w('| Παράμετρος           | Τιμή                |')
        w('|----------------------|---------------------|')
        w(f'| Model                | {model_key}   |')
        for k, v in model_params.items():
            nice_k = k.replace('_', ' ').title()
            if isinstance(v, bool):
                v_str = '✓' if v else '✗'
            else:
                v_str = str(v)
            w(f'| {nice_k:<20s} | {v_str:<19s} |')
        w('')

        # Grouped-ACE params
        ctrl = cfg.get('Controller', {})
        if ctrl:
            w('### Grouped-ACE Parameters\n')
            w('| Παράμετρος     | Τιμή   |')
            w('|----------------|--------|')
            for k, v in ctrl.items():
                w(f'| {k:<14s} | {v}  |')
            w('')

        # Training hyperparams
        train_cfg = cfg.get('train', {})
        trainer_name = cfg.get('trainer', 'ReinforcementTrainer')
        trainer_cfg = cfg.get(trainer_name, {})
        w('### Hyperparameters Εκπαίδευσης\n')
        w('| Παράμετρος                     | Τιμή  |')
        w('|--------------------------------|-------|')
        for k, v in {**trainer_cfg, **train_cfg}.items():
            nice_k = k.replace('_', ' ').title()
            if isinstance(v, bool):
                v_str = '✓' if v else '✗'
            else:
                v_str = str(v)
            w(f'| {nice_k:<30s} | {v_str} |')
        w('')

    # Dataset
    if data['corpus']:
        w('### Dataset\n')
        w('| Split | Προτάσεις |')
        w('|-------|-----------|')
        w(f'| Train | {data["corpus"]["train"]:,}    |')
        w(f'| Dev   | {data["corpus"]["dev"]:,}     |')
        w(f'| Test  | {data["corpus"]["test"]:,}     |')
        w('')
    w('---\n')

    # ── 3. Embeddings ──
    w(f'## 3. Embeddings ({num_emb} Υποψήφια × {num_groups} Groups = {num_actions} Actions)\n')
    w('| #   | Embedding (short)    | Raw Path / ID                          |')
    w('|-----|----------------------|----------------------------------------|')
    for i, (sn, raw) in enumerate(zip(data['emb_names'], data['raw_emb_names'])):
        w(f'| {i}   | **{sn}** | `{raw}` |')
    w('')
    w('---\n')

    # ── 4. Episode progress ──
    w('## 4. Πρόοδος Εκπαίδευσης ανά Episode\n')

    # Header
    emb_headers = ' | '.join(f'{n:^9s}' for n in data['emb_names'])
    header = f'| Episode | Epochs | Best Test F1 | {emb_headers} | Kept  |'
    sep_emb = ' | '.join(':-------:' for _ in data['emb_names'])
    separator = f'|:-------:|:------:|:------------:| {sep_emb} |:-----:|'
    w(header)
    w(separator)

    for ep in data['episodes']:
        ep_num = ep['episode']
        epochs = ep['epochs']
        f1 = ep['best_test_f1']
        is_bl = ep.get('is_baseline', False)
        groups = ep.get('action_groups', {})

        f1_str = f'{f1:.2f}'
        if is_bl:
            f1_str += ' ★'

        # Bold the best episode
        if ep_num == data.get('best_episode'):
            ep_str = f'**{ep_num}**'
            epoch_str = f'**{epochs}**'
            f1_str = f'**{f1_str}**'
        else:
            ep_str = str(ep_num)
            epoch_str = str(epochs)

        # Per-embedding group counts
        emb_cells = []
        total_kept = 0
        for name in data['emb_names']:
            if name in groups:
                g = groups[name]
                kept = sum(g)
                total_kept += kept
                cell = f'{kept}/{num_groups}'
            else:
                cell = '?'
            emb_cells.append(f'{cell:^9s}')

        emb_str = ' | '.join(emb_cells)
        kept_str = f'{total_kept}/{num_actions}'
        if ep_num == data.get('best_episode'):
            kept_str = f'**{kept_str}**'

        w(f'| {ep_str:^7s} | {epoch_str:^6s} | {f1_str:^12s} | {emb_str} | {kept_str:^5s} |')

    w('')
    total_eps = len(data['episodes'])
    w(f'**Σύνολο:** {total_eps} episodes, ~{data["total_epochs"]} epochs, {data["duration_str"]} εκπαίδευσης')
    w('')

    # Baseline transitions
    if data['baselines']:
        w('### Baseline Transitions\n')
        w('| Γεγονός         | Episode | Dev F1  | Groups Kept | Σημείωση |')
        w('|-----------------|:-------:|:-------:|:-----------:|----------|')
        for i, bl in enumerate(data['baselines']):
            ep = bl['episode']
            sc = bl['score']
            groups = bl['action_groups']
            total_kept = sum(sum(g) for g in groups.values())
            label = 'Αρχικό baseline' if i == 0 else ('**Τελικό** ★' if i == len(data['baselines']) - 1 else 'Νέο baseline')
            desc_parts = []
            for name, g in groups.items():
                k = sum(g)
                if k > 0:
                    desc_parts.append(f'{name} {k}/{num_groups}')
            desc = ', '.join(desc_parts) if len(desc_parts) <= 6 else f'{total_kept}/{num_actions} groups'
            w(f'| {label} | {ep} | {sc:.2f} | {total_kept}/{num_actions} | {desc} |')
        w('')
    w('---\n')

    # ── 5. Best model action ──
    if data['best_action']:
        w('## 5. Best Model Action (Episode ' + str(data.get('best_episode', '?')) + ')\n')
        w('| Embedding          | Groups     | Kept | Σημείωση                    |')
        w('|--------------------|------------|:----:|-----------------------------|')
        total_kept = 0
        for name in data['emb_names']:
            g = data['best_action'].get(name, [])
            kept = sum(g)
            total_kept += kept
            g_str = str(g)
            if kept == 0:
                note = '**Αποκλείστηκε**'
            elif kept == num_groups:
                note = 'Πλήρης χρήση'
            elif kept >= num_groups * 0.75:
                note = '**Κυρίαρχο**'
            else:
                note = 'Μερική χρήση'
            w(f'| {name:<18s} | {g_str:<10s} | {kept}/{num_groups}  | {note:<27s} |')
        pct = total_kept / num_actions * 100 if num_actions else 0
        w(f'| **Σύνολο**         |            | **{total_kept}/{num_actions} ({pct:.1f}%)** | |')
        w('')
        w('---\n')

    # ── 6. Embedding selection frequency ──
    w('## 6. Συχνότητα Επιλογής Embedding\n')
    w('| Embedding          | Επιλέχθηκε (>0 groups) | Ποσοστό |')
    w('|--------------------|:----------------------:|:-------:|')
    freq = {n: 0 for n in data['emb_names']}
    for ep in data['episodes']:
        for name in data['emb_names']:
            g = ep.get('action_groups', {}).get(name, [])
            if sum(g) > 0:
                freq[name] += 1
    total_eps = len(data['episodes'])
    for name in sorted(freq, key=lambda n: -freq[n]):
        cnt = freq[name]
        pct = cnt / total_eps * 100 if total_eps else 0
        w(f'| {name:<18s} | {cnt}/{total_eps}{"":>20s} | {pct:.1f}%   |')
    w('')
    w('---\n')

    # ── 7. Final test results ──
    w('## 7. Τελική Αξιολόγηση στο Test Set (best-model.pt)\n')

    if entities:
        w('### Αποτελέσματα ανά Οντότητα\n')
        w('| Οντότητα    |   TP |  FP |  FN | Precision | Recall | F1         |')
        w('|-------------|-----:|----:|----:|:---------:|:------:|:----------:|')
        # Sort by F1 descending
        for e in sorted(entities, key=lambda x: -x['f1']):
            w(f'| {e["name"]:<11s} | {e["tp"]:>4d} | {e["fp"]:>3d} | {e["fn"]:>3d} | '
              f'{e["precision"]*100:>8.2f}% | {e["recall"]*100:>5.2f}% | **{e["f1"]*100:.2f}%** |')
        w('')

    w('### Συνολικά\n')
    w('| Μετρική        | Precision | Recall | F1         |')
    w('|----------------|:---------:|:------:|:----------:|')
    if entities:
        total_tp = sum(e['tp'] for e in entities)
        total_fp = sum(e['fp'] for e in entities)
        total_fn = sum(e['fn'] for e in entities)
        prec = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) else 0
        rec = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) else 0
    else:
        prec = rec = 0
    w(f'| **MICRO AVG**  | {prec:>8.2f}% | {rec:>5.2f}% | **{micro_f1*100:.2f}%** |')
    w(f'| **MACRO AVG**  |     —     |   —    | **{macro_f1*100:.2f}%** |')
    w('')

    # ── 8. Selected dimension indices ──
    if data['best_action'] and data.get('selected_dims') is not None:
        sel = data['selected_dims']     # list[dict]: {name, dim, kept_groups, total_groups, kept_dims, indices}
        group_seed = data.get('group_seed')
        w('---\n')
        w('## 8. Επιλεγμένες Διαστάσεις (Dimension Indices)\n')
        w('Τα groups δημιουργήθηκαν με **τυχαία permutation** των dimensions κάθε '
          f'embedding (mode=`equal`, `group_seed={group_seed}`). Ο παρακάτω πίνακας '
          'δείχνει *ακριβώς πόσες* διαστάσεις κράτησε το best-action ανά embedding· '
          'τα συγκεκριμένα indices (στην αρχική σειρά του embedding) είναι μέσα '
          'στις πτυσσόμενες λεπτομέρειες.\n')

        # Compact summary table
        w('| # | Embedding | D | Groups (kept/N) | Dims (kept/D) | % |')
        w('|:-:|-----------|--:|:---------------:|:-------------:|--:|')
        total_d_kept = 0
        total_d_all = 0
        for i, s in enumerate(sel, start=1):
            kg, ng = s['kept_groups'], s['total_groups']
            kd, dd = s['kept_dims'], s['dim']
            total_d_kept += kd
            total_d_all += dd
            pct = (kd / dd * 100) if dd else 0
            label = f'**{s["name"]}**' if kg > 0 else f'~~{s["name"]}~~'
            w(f'| {i} | {label} | {dd} | {kg}/{ng} | {kd}/{dd} | {pct:.1f}% |')
        tot_pct = (total_d_kept / total_d_all * 100) if total_d_all else 0
        w(f'| **Σύνολο** |  | **{total_d_all}** |  | **{total_d_kept}/{total_d_all}** | **{tot_pct:.1f}%** |')
        w('')

        # Per-embedding details (collapsible)
        for i, s in enumerate(sel, start=1):
            name = s['name']
            kd, dd = s['kept_dims'], s['dim']
            kg, ng = s['kept_groups'], s['total_groups']
            if kg == 0:
                w(f'<details><summary><b>{i}. {name}</b> — αποκλεισμένο '
                  f'(0/{ng} groups, 0/{dd} dims)</summary>\n')
                w('\n*Καμία διάσταση δεν χρησιμοποιήθηκε.*\n')
                w('\n</details>\n')
                continue
            indices = s['indices']    # already sorted ascending
            # group_chunks: list of lists, one per kept group, with chunk-local indices
            chunks = s.get('chunks', [])
            w(f'<details><summary><b>{i}. {name}</b> — {kg}/{ng} groups, '
              f'<b>{kd}/{dd}</b> dims ({kd/dd*100:.1f}%)</summary>\n')
            w('')
            # Show chunk-by-chunk breakdown
            for ci, (gid, chunk_idx) in enumerate(chunks):
                w(f'- Group **#{gid+1}** ({len(chunk_idx)} dims): `'
                  + _format_indices(chunk_idx) + '`')
            w('')
            w('**Όλες οι επιλεγμένες διαστάσεις (αύξουσα σειρά):**\n')
            w('```')
            w(_format_indices(indices, wrap=16))
            w('```')
            w('\n</details>\n')
        w('')

    return '\n'.join(lines)


def _format_indices(idx_list, wrap=None):
    """Format a list of integers compactly. If wrap is set, insert newlines every wrap items."""
    if not idx_list:
        return '—'
    if wrap is None:
        return ', '.join(str(x) for x in idx_list)
    out = []
    for i in range(0, len(idx_list), wrap):
        out.append(', '.join(str(x) for x in idx_list[i:i+wrap]))
    return ',\n'.join(out)


def compute_selected_dims(data, cfg, model_path):
    """Reconstruct random permutations from group_seed and slice by best_action mask.

    Returns: list[dict] aligned with data['emb_names'], each dict has keys
    name, dim, total_groups, kept_groups, kept_dims, indices (sorted asc), chunks.
    Returns None if model file is missing or seed is unavailable.
    """
    if not model_path or not os.path.isfile(model_path):
        return None, None
    group_seed = None
    if cfg:
        ctrl = cfg.get('Controller', {}) or {}
        group_seed = ctrl.get('group_seed')
    if group_seed is None:
        return None, None

    try:
        import torch
    except ImportError:
        return None, None

    try:
        m = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f'WARNING: could not load {model_path}: {e}', file=sys.stderr)
        return None, group_seed

    emb_module = m.get('embeddings') if isinstance(m, dict) else getattr(m, 'embeddings', None)
    if emb_module is None or not hasattr(emb_module, 'embeddings'):
        return None, group_seed

    pairs = sorted([(e.name, int(e.embedding_length)) for e in emb_module.embeddings],
                   key=lambda x: x[0])
    runtime_names = [p[0] for p in pairs]
    embed_dims = [p[1] for p in pairs]

    # Map runtime emb names → log short names by position (both sorted by name)
    short_names = data.get('emb_names', [])
    if len(short_names) != len(runtime_names):
        print(f'WARNING: emb count mismatch (log={len(short_names)} vs '
              f'model={len(runtime_names)})', file=sys.stderr)
        return None, group_seed

    # Reconstruct random permutations
    rng = torch.Generator()
    rng.manual_seed(int(group_seed))
    perms = [torch.randperm(d, generator=rng).tolist() for d in embed_dims]

    num_groups = data.get('num_groups', 1)

    out = []
    for sn, dim, perm in zip(short_names, embed_dims, perms):
        # torch.chunk semantics: chunk_size = ceil(dim/num_groups)
        chunk_size = (dim + num_groups - 1) // num_groups
        groups = [perm[i*chunk_size:(i+1)*chunk_size] for i in range(num_groups)]
        mask = data['best_action'].get(sn, [0]*num_groups)
        kept_groups = sum(mask)
        all_indices = []
        kept_chunks = []
        for gi, g in enumerate(groups):
            if gi < len(mask) and mask[gi] == 1:
                all_indices.extend(g)
                kept_chunks.append((gi, sorted(g)))
        all_indices.sort()
        out.append({
            'name': sn,
            'dim': dim,
            'total_groups': num_groups,
            'kept_groups': kept_groups,
            'kept_dims': len(all_indices),
            'indices': all_indices,
            'chunks': kept_chunks,
        })
    return out, group_seed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate MD report from ACE training log.')
    parser.add_argument('log', help='Path to training.log')
    parser.add_argument('--config', '-c', help='Path to YAML config (optional)', default=None)
    parser.add_argument('--model', '-m',
                        help='Path to best-model.pt (default: <log_dir>/best-model.pt). '
                             'Used to reconstruct random dim permutations.',
                        default=None)
    parser.add_argument('-o', '--output', help='Output .md file (default: stdout)', default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.log):
        print(f'ERROR: Log file not found: {args.log}', file=sys.stderr)
        sys.exit(1)

    cfg = {}
    if args.config:
        cfg = parse_config(args.config)

    log_parser = LogParser(args.log)
    data = log_parser.parse()

    # Default model path: alongside the log
    model_path = args.model or str(Path(args.log).parent / 'best-model.pt')
    sel_dims, group_seed = compute_selected_dims(data, cfg, model_path)
    data['selected_dims'] = sel_dims
    data['group_seed'] = group_seed

    report = generate_report(data, cfg)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding='utf-8')
        print(f'Report written to {args.output}')
    else:
        print(report)


if __name__ == '__main__':
    main()
