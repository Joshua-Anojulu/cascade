"""How often does a pairwise view understate a multi-drug exposure change?

The headline number comes from enumerating real combinations in the FDA role
table, not from hand-picked cases.

The model depends on a combination only through its classes: which enzyme, the
multiset of inhibitor classes acting on it, and the substrate's class. So the
whole space can be covered exactly. The script enumerates every class multiset,
evaluates the model once per multiset, and weights the result by how many real
drug combinations in the table map onto it. Nothing is sampled and nothing is
truncated, so "every combination of 3 listed inhibitors against every listed
substrate" means every one.

Two things are counted:

  understated   the full model's severity band is higher than the best band any
                single inhibitor in the list reaches with that substrate. This
                needs no assumption about what a checker chooses to display.

  invisible     every pair sits below a checker's reporting threshold while the
                combination sits at or above it. Pairwise checkers commonly do
                not surface weak-inhibitor pairs, so the threshold is a stated
                assumption and the count is reported for each setting.

Run: python evaluate.py [--regimens N] [--size N] [--seed N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from math import comb

from core.model import SEVERITY_ORDER, analyze, cascades, predict_aucr, severity
from core.roles import load_roles

INHIBITOR_CLASSES = ("strong", "moderate", "weak")
SUBSTRATE_CLASSES = ("sensitive", "moderate")
THRESHOLDS = ("low", "moderate")


def class_counts(roles):
    """Per enzyme: how many drugs hold each inhibitor class, and each substrate class."""
    inhibitors = defaultdict(Counter)
    substrates = defaultdict(Counter)
    for role in roles:
        if role.role == "inhibitor" and role.strength in INHIBITOR_CLASSES:
            inhibitors[role.enzyme][role.strength] += 1
        elif role.role == "substrate" and role.strength in SUBSTRATE_CLASSES:
            substrates[role.enzyme][role.strength] += 1
    return inhibitors, substrates


def ways(available: Counter, multiset: tuple[str, ...]) -> int:
    """How many ways to pick this multiset of inhibitor classes from one enzyme."""
    total = 1
    for cls, needed in Counter(multiset).items():
        total *= comb(available.get(cls, 0), needed)
        if total == 0:
            return 0
    return total


def exhaustive(roles, sizes=(2, 3)):
    inhibitors, substrates = class_counts(roles)
    rows = []
    for size in sizes:
        for multiset in itertools.combinations_with_replacement(INHIBITOR_CLASSES, size):
            for substrate_class in SUBSTRATE_CLASSES:
                full = predict_aucr(list(multiset), [], substrate_class)
                full_band = severity(full["point"])
                singles = {
                    cls: severity(predict_aucr([cls], [], substrate_class)["point"])
                    for cls in set(multiset)
                }
                best_pair = max(singles.values(), key=lambda band: SEVERITY_ORDER[band])
                for enzyme, available in inhibitors.items():
                    count = ways(available, multiset)
                    if not count:
                        continue
                    for substrate_count in [substrates[enzyme].get(substrate_class, 0)]:
                        if not substrate_count:
                            continue
                        # A drug cannot inhibit and be the substrate in the same finding;
                        # overlap is corrected below using the real drug names.
                        rows.append({
                            "enzyme": enzyme, "size": size, "classes": list(multiset),
                            "substrate_class": substrate_class,
                            "combinations": count * substrate_count,
                            "full_band": full_band, "full_aucr": full["point"],
                            "best_pair_band": best_pair,
                            "understated": SEVERITY_ORDER[full_band] > SEVERITY_ORDER[best_pair],
                            "invisible_at": [t for t in THRESHOLDS
                                             if SEVERITY_ORDER[full_band] >= SEVERITY_ORDER[t]
                                             and SEVERITY_ORDER[best_pair] < SEVERITY_ORDER[t]],
                        })
    return rows


def overlap_correction(roles, sizes=(2, 3)):
    """Drugs that are both an inhibitor and a substrate of the same enzyme.

    Those combinations are not real (a drug is never its own interacting
    partner), so they are subtracted from the weighted totals.
    """
    inhibitors, substrates = defaultdict(set), defaultdict(set)
    for role in roles:
        if role.role == "inhibitor" and role.strength in INHIBITOR_CLASSES:
            inhibitors[(role.enzyme, role.strength)].add(role.drug)
        elif role.role == "substrate" and role.strength in SUBSTRATE_CLASSES:
            substrates[(role.enzyme, role.strength)].add(role.drug)
    return inhibitors, substrates


def exact_counts(roles, sizes=(2, 3)):
    """Weighted totals with self-pairs removed, by enumerating drug names per enzyme."""
    per_enzyme = defaultdict(lambda: {"inhibitor": {}, "substrate": {}})
    for role in roles:
        if role.role == "inhibitor" and role.strength in INHIBITOR_CLASSES:
            per_enzyme[role.enzyme]["inhibitor"][role.drug] = role.strength
        elif role.role == "substrate" and role.strength in SUBSTRATE_CLASSES:
            per_enzyme[role.enzyme]["substrate"][role.drug] = role.strength

    cache: dict = {}
    def evaluate(multiset, substrate_class):
        key = (multiset, substrate_class)
        if key not in cache:
            full = predict_aucr(list(multiset), [], substrate_class)
            full_band = severity(full["point"])
            best_pair = max(
                (severity(predict_aucr([cls], [], substrate_class)["point"]) for cls in set(multiset)),
                key=lambda band: SEVERITY_ORDER[band],
            )
            cache[key] = {
                "full_band": full_band, "full_aucr": full["point"], "best_pair_band": best_pair,
                "understated": SEVERITY_ORDER[full_band] > SEVERITY_ORDER[best_pair],
                "invisible_at": [t for t in THRESHOLDS
                                 if SEVERITY_ORDER[full_band] >= SEVERITY_ORDER[t]
                                 and SEVERITY_ORDER[best_pair] < SEVERITY_ORDER[t]],
            }
        return cache[key]

    results = {size: Counter() for size in sizes}
    examples: dict = {}
    for enzyme, group in sorted(per_enzyme.items()):
        inhibitor_names = sorted(group["inhibitor"])
        substrate_names = sorted(group["substrate"])
        for size in sizes:
            for combo in itertools.combinations(inhibitor_names, size):
                multiset = tuple(sorted((group["inhibitor"][d] for d in combo), reverse=True))
                for substrate in substrate_names:
                    if substrate in combo:
                        continue
                    verdict = evaluate(multiset, group["substrate"][substrate])
                    bucket = results[size]
                    bucket["total"] += 1
                    if verdict["understated"]:
                        bucket["understated"] += 1
                        bucket[f"from_{verdict['best_pair_band']}_to_{verdict['full_band']}"] += 1
                        key = (size, verdict["full_band"], verdict["best_pair_band"])
                        if key not in examples:
                            examples[key] = {
                                "enzyme": enzyme, "inhibitors": list(combo), "classes": list(multiset),
                                "substrate": substrate, "substrate_class": group["substrate"][substrate],
                                "full_aucr": verdict["full_aucr"], "full_band": verdict["full_band"],
                                "pairwise_band": verdict["best_pair_band"],
                            }
                    for threshold in verdict["invisible_at"]:
                        bucket[f"invisible_at_{threshold}"] += 1
    return results, examples


def regimens(roles, count=2000, size=7, seed=20261108):
    drugs = sorted({role.drug for role in roles})
    rng = random.Random(seed)
    hits, severities, contributors = 0, Counter(), Counter()
    for _ in range(count):
        sample = rng.sample(drugs, size)
        found = cascades(sample, roles)
        if found:
            hits += 1
            severities[found[0]["finding"].severity] += 1
            contributors[len(found[0]["finding"].contributors)] += 1
    return {"regimens": count, "size": size, "with_understated_finding": hits,
            "share": hits / count if count else 0.0,
            "top_severity": dict(severities), "contributors": dict(sorted(contributors.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimens", type=int, default=2000)
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--json", default="results.json")
    args = parser.parse_args()

    roles = load_roles()
    counts, examples = exact_counts(roles)
    sample = regimens(roles, args.regimens, args.size, args.seed)

    print(f"FDA role table: {len({r.drug for r in roles})} drugs, {len(roles)} published roles\n")
    print("1. Every combination in the table (exhaustive, nothing sampled)")
    summary = {}
    for size, bucket in counts.items():
        total, understated = bucket["total"], bucket["understated"]
        print(f"   {size} inhibitors + 1 substrate: {total:>7} combinations, "
              f"{understated:>6} understated by a pairwise view ({understated / total * 100:.1f}%)")
        for threshold in THRESHOLDS:
            invisible = bucket[f"invisible_at_{threshold}"]
            print(f"      invisible to a checker reporting '{threshold}' and above: "
                  f"{invisible} ({invisible / total * 100:.1f}%)")
        moves = {k: v for k, v in bucket.items() if k.startswith("from_")}
        print(f"      band moves: {moves}")
        summary[size] = dict(bucket)

    print(f"\n2. Random {args.size}-drug regimens drawn from the table "
          f"(n={sample['regimens']}, seed {args.seed})")
    print(f"   {sample['with_understated_finding']} of {sample['regimens']} "
          f"({sample['share'] * 100:.1f}%) contain at least one understated finding")
    print(f"   band of the top finding: {sample['top_severity']}")
    print(f"   contributing drugs in the top finding: {sample['contributors']}")

    print("\n3. One worked example of each band move")
    for key in sorted(examples, key=lambda k: (-k[0], k[1])):
        row = examples[key]
        print(f"   {row['enzyme']}: {' + '.join(row['inhibitors'])} ({', '.join(row['classes'])}) "
              f"with {row['substrate']} ({row['substrate_class']})")
        print(f"      full model x{row['full_aucr']:.1f} = {row['full_band']}, "
              f"best single pair = {row['pairwise_band']}")

    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump({"exhaustive": summary, "regimens": sample,
                   "examples": {str(k): v for k, v in examples.items()}}, handle, indent=1)
    print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
