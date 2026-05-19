"""
Label mapping — based on actual post-deduplication training split.

Post-dedup reality:
  Train total : 36,322
  Train Fake  :    906  → weight = 36322 / (2 * 906)   = 20.04
  Train Cred  : 35,416  → weight = 36322 / (2 * 35416) = 0.513
  Real ratio  : 39.1:1  (worse than EDA showed — duplicates inflated fake count)
"""

RAW_TO_INT = {
    0.0: 0, 1.0: 1,
    "fake": 0, "authentic": 1, "real": 1, "credible": 1,
    "pants-fire": 0, "false": 0,
    "barely-true": 2, "half-true": 2,
    "mostly-true": 1, "true": 1,
}

INT_TO_LABEL = {0: "Fake", 1: "Credible", 2: "Unverified"}
INT_TO_COLOR = {0: "#b83030", 1: "#1a6b40", 2: "#a05c0a"}
NUM_LABELS   = 2

# Recomputed from actual training split (post-dedup)
CLASS_WEIGHTS = {
    0: round(36322 / (2 * 906),   4),   # 20.0386 — Fake
    1: round(36322 / (2 * 35416), 4),   # 0.5126  — Credible
}

def raw_to_int(raw_label) -> int:
    if isinstance(raw_label, float):
        raw_label = int(raw_label) if raw_label in (0.0, 1.0) else raw_label
    key = raw_label if isinstance(raw_label, (int, float)) \
          else str(raw_label).lower().strip()
    if key not in RAW_TO_INT:
        raise ValueError(f"Unknown label: {raw_label!r}")
    return RAW_TO_INT[key]

def int_to_label(idx: int) -> str:
    return INT_TO_LABEL[idx]

def get_class_weight_list() -> list:
    return [CLASS_WEIGHTS[i] for i in range(NUM_LABELS)]

if __name__ == "__main__":
    print("Updated class weights (post-dedup training split):")
    for k, v in CLASS_WEIGHTS.items():
        print(f"  {INT_TO_LABEL[k]:<12} weight: {v}")
    print(f"\nImbalance ratio: {CLASS_WEIGHTS[0]/CLASS_WEIGHTS[1]:.1f}:1")
    print("Class weight list:", get_class_weight_list())
