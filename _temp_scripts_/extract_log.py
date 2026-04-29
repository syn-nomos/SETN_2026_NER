import os, re

log_path = r'resources/taggers/gln_grouped_N4_Adam_seed44/training.log'
if not os.path.exists(log_path):
    print("Log file not found at:", log_path)
    exit(1)

with open(log_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

current_ep = None
max_epochs = 0
best_f1 = 0.0
# List to hold actions per embedding. Will store lists of strings like "[1, 0, 0, 0]"
ep_action = []

# List to hold the history of episodes
history = []

for idx, line in enumerate(lines):
    m_ep = re.search(r'Episode (\d+) action \(num_groups=4\)', line)
    if m_ep:
        # Save previous episode stats if we have any
        if current_ep is not None:
            history.append({
                "episode": current_ep,
                "epochs": max_epochs,
                "best_f1": best_f1,
                "action": ep_action
            })
            
        current_ep = int(m_ep.group(1))
        max_epochs = 0
        best_f1 = 0.0
        ep_action = []
        
        # Next ~6 lines contain the embedding action selections
        for j in range(1, 10):
            if idx + j < len(lines):
                next_line = lines[idx + j]
                if 'groups=' in next_line:
                    # Extract the part between groups= and kept=
                    grp_match = re.search(r'groups=(\[.*?\])', next_line)
                    if grp_match:
                        ep_action.append(grp_match.group(1))
                if len(ep_action) == 6: # We have our 6 embeddings
                    break

    # Look for epoch numbers
    m_epoch = re.search(r'Epoch (\d+) done', line)
    if m_epoch:
        ep_val = int(m_epoch.group(1))
        if ep_val > max_epochs:
            max_epochs = ep_val

    # Look for Test Average or Test Dev MICRO_AVG F1
    m_f1 = re.search(r'MICRO_AVG: acc [\d\.]+ - f1-score ([\d\.]+)', line)
    if m_f1:
        f1_val = float(m_f1.group(1)) * 100
        if f1_val > best_f1:
            best_f1 = f1_val

# Don't forget the last one
if current_ep is not None:
    history.append({
        "episode": current_ep,
        "epochs": max_epochs,
        "best_f1": best_f1,
        "action": ep_action
    })

print("| Episode | Epochs | Best Test F1 | Επιλεγμένα Embeddings / Groups |")
print("|:-------:|:------:|:------------:|--------------------------------|")

emb_names = ["GreekBERT", "RoBERTa", "FastText", "mDeBERTa", "FastChar", "BPEmb"]

# Add Baseline 
print(f"| **0 (Baseline)** | -- | -- | Όλα τα embeddings [1,1,1,1] |")

for h in history:
    # Build the action string
    # We'll just show the embeddings that have at least one '1'
    selected = []
    for i, act in enumerate(h["action"]):
        if '1' in act:
            selected.append(f"{emb_names[i]} {act}")
    
    act_str = ", ".join(selected) if selected else "Κανένα"
    print(f"| {h['episode']} | {h['epochs']} | {h['best_f1']:.2f}% | {act_str} |")

