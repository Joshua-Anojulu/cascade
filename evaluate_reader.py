"""Measure the medication-list reader on data it was never tuned on.

    python evaluate_reader.py
    python evaluate_reader.py --json reader-results.json

Standard library only. Four tests:

1. Unseen misspellings. Every dictionary name misspelled again with a new random
   seed and a harsher mix (more two- and three-letter errors) than training used.
2. Look-alike names. Pairs from the ISMP List of Confused Drug Names (2024), each
   name typed exactly and with typos. The count that matters is how often a name
   is read as its look-alike partner.
3. Ordinary English. The 38,000 commonest English words, typed alone. None
   should be read as a medicine.
4. Blind lists. data/blind_med_lists.json: 80 medication lists in five styles,
   written with answer keys by a separate agent that never saw the reader. Run
   once, after the model was frozen. Compared with Cascade's earlier resolver
   (exact names plus the 84 aliases in synonyms.csv), given the same splitting
   and dose stripping, so the comparison isolates what the model adds.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.reader import STOP, key, load_lexicon, load_model, read_list  # noqa: E402
from core.roles import resolve  # noqa: E402
from core.typos import variants  # noqa: E402

DATA = ROOT / "data"
SEED = 99

# From the ISMP List of Confused Drug Names (Institute for Safe Medication
# Practices, 2024). Only pairs where at least one name is in Cascade's dictionary.
ISMP_PAIRS = [
    ("Accupril", "Aciphex"), ("Aciphex", "Aricept"), ("Actonel", "Actos"), ("Adderall", "Inderal"),
    ("Allegra", "Viagra"), ("alprazolam", "clonazepam"), ("atomoxetine", "atorvastatin"),
    ("Avandia", "Coumadin"), ("Avandia", "Prandin"), ("Brilinta", "Brintellix"), ("Cardura", "Coumadin"),
    ("Celebrex", "Celexa"), ("Celebrex", "Cerebyx"), ("Celexa", "Cerebyx"), ("Celexa", "Zyprexa"),
    ("cetirizine", "sertraline"), ("clobazam", "clonazepam"), ("clonazepam", "clonidine"),
    ("clonazepam", "clozapine"), ("clonidine", "clozapine"), ("clonidine", "Klonopin"),
    ("Colace", "Cozaar"), ("Cozaar", "Zocor"), ("Desyrel", "Seroquel"), ("Dexilant", "duloxetine"),
    ("diazepam", "diltiazem"), ("duloxetine", "fluoxetine"), ("duloxetine", "paroxetine"),
    ("fluoxetine", "paroxetine"), ("fluoxetine", "Loxitane"), ("hydralazine", "hydroxyzine"),
    ("hydralazine", "hydrochlorothiazide"), ("labetalol", "Lamictal"), ("Lamictal", "Lamisil"),
    ("Lantus", "Latuda"), ("Lasix", "Luvox"), ("Lipitor", "Zyrtec"), ("Lipitor", "Loniten"),
    ("lithium", "Ultram"), ("Nexium", "Nexavar"), ("Paxil", "Plavix"), ("Paxil", "Taxol"),
    ("Paxil", "Doxil"), ("Plavix", "Pradaxa"), ("Prilosec", "Prozac"), ("Prograf", "Prozac"),
    ("Prozac", "Provera"), ("Restoril", "Risperdal"), ("Risperdal", "ropinirole"),
    ("risperidone", "ropinirole"), ("Seroquel", "Sinequan"), ("Seroquel", "Serzone"),
    ("Sinequan", "Singulair"), ("sertraline", "Soriatane"), ("Tegretol", "Trental"),
    ("Tegretol", "Tequin"), ("tiagabine", "tizanidine"), ("Topamax", "Toprol"),
    ("tramadol", "trazodone"), ("Ultracet", "Duricef"), ("Xanax", "Zantac"), ("Xanax", "Tenex"),
    ("Xanax", "Fanapt"), ("Zocor", "Zyrtec"), ("Zyprexa", "Zyrtec"), ("Zyprexa", "Zestril"),
    ("Zyrtec", "Zantac"),
]


def meaning_of(entry) -> tuple:
    return tuple(entry.drugs) or tuple("other:" + o for o in entry.others)


def verdict_meaning(v: dict):
    return meaning_of(v["entry"]) if v["status"] in ("matched", "not_covered") else None


def test_misspellings(model, lexicon) -> dict:
    rng = random.Random(SEED)
    is_word = lambda w: w in lexicon  # noqa: E731
    c = Counter()
    for entry in model.entries:
        truth = meaning_of(entry)
        for typed in variants(entry.key, rng, 5 if entry.drugs else 2, is_word=is_word,
                              weights_long=(40, 40, 20), weights_short=(60, 40, 0)):
            v = model.decide(typed)
            group = "table" if entry.drugs else "other"
            if v["status"] in ("matched", "not_covered"):
                c[group + ("_right" if verdict_meaning(v) == truth else "_wrong")] += 1
            elif v["status"] == "ambiguous":
                c[group + "_asked"] += 1
            else:
                c[group + "_unread"] += 1
    out = {}
    for group in ("table", "other"):
        n = sum(c[group + s] for s in ("_right", "_wrong", "_asked", "_unread"))
        out[group] = {"typed": n, **{s: round(c[group + "_" + s] / n, 4) for s in ("right", "wrong", "asked", "unread")}}
    return out


def test_lookalikes(model, lexicon) -> dict:
    by_key = {e.key: e for e in model.entries}
    rng = random.Random(SEED + 1)
    is_word = lambda w: w in lexicon  # noqa: E731
    rows, c, swaps = [], Counter(), []
    for a, b in ISMP_PAIRS:
        for typed_name, partner in ((a, b), (b, a)):
            partner_entry = by_key.get(key(partner))
            own_entry = by_key.get(key(typed_name))
            if partner_entry is None:
                continue                      # nothing to be confused with
            forbidden = meaning_of(partner_entry)
            if own_entry is not None and meaning_of(own_entry) == forbidden:
                continue                      # same drug under two names
            attempts = [("exact", key(typed_name))] + [
                ("typo", t) for t in variants(key(typed_name), rng, 3, is_word=is_word)]
            for kind, typed in attempts:
                v = model.decide(typed)
                got = verdict_meaning(v)
                if got == forbidden:
                    c[kind + "_swapped"] += 1
                    swaps.append(f"{typed} -> {v['entry'].display} (partner of {typed_name})")
                elif own_entry is not None and got == meaning_of(own_entry):
                    c[kind + "_right"] += 1
                elif v["status"] == "ambiguous":
                    c[kind + "_asked"] += 1
                elif got is None:
                    c[kind + "_unread"] += 1
                else:
                    c[kind + "_other"] += 1
                c[kind] += 1
    return {"directed_pairs_tested": c["exact"], "typos_tested": c["typo"],
            "exact": {k: c["exact_" + k] for k in ("right", "asked", "unread", "other", "swapped")},
            "typo": {k: c["typo_" + k] for k in ("right", "asked", "unread", "other", "swapped")},
            "swaps": swaps}


def test_english(model) -> dict:
    lexicon = sorted(load_lexicon())
    words = [w for w in lexicon if len(w) >= 3 and w not in STOP]
    c = Counter(model.decide(w)["status"] for w in words if w not in model.keys)
    return {"words": len(words), "read_as_medicine": c["matched"] + c["not_covered"], "asked_about": c["ambiguous"]}


def baseline_read(text: str) -> list:
    """Cascade's resolver before the model: exact names and synonyms.csv, one per piece."""
    import re
    drugs = []
    for piece in re.split(r"\n|;|,|\||\s(?:and|&|plus)\s|\+", text):
        r = resolve(key(piece))
        if r["drug"] and r["drug"] not in drugs:
            drugs.append(r["drug"])
    return drugs


def test_blind(model, path: Path) -> dict:
    lists = json.load(open(path, encoding="utf-8"))
    results = {}
    for name, reader in (("model", lambda t: read_list(t, model)["drugs"]), ("baseline", baseline_read)):
        tp = fp = fn = exact = trap_hits = ambiguous_guessed = 0
        styles: dict = {}
        errors = []
        for item in lists:
            gold = set(item["table_drugs"])
            ambiguous = {c for a in item.get("ambiguous", []) for c in a["candidates"]}
            read = set(reader(item["text"]))
            got = read - (ambiguous - gold)
            guessed = read & (ambiguous - gold)
            ambiguous_guessed += len(guessed)
            tp += len(got & gold)
            fp += len(got - gold)
            fn += len(gold - got)
            ok = got == gold
            exact += ok
            s = styles.setdefault(item["style"], [0, 0])
            s[0] += ok
            s[1] += 1
            for trap in item.get("traps", []):
                if trap["not"] in got and trap["not"] not in gold:
                    trap_hits += 1
            if not ok and name == "model":
                errors.append({"id": item["id"], "missed": sorted(gold - got), "extra": sorted(got - gold)})
        results[name] = {
            "lists": len(lists),
            "lists_read_exactly": exact,
            "precision": round(tp / max(1, tp + fp), 4),
            "recall": round(tp / max(1, tp + fn), 4),
            "wrong_drugs_added": fp,
            "drugs_missed": fn,
            "look_alike_traps_fallen_for": trap_hits,
            "ambiguous_mentions_guessed": ambiguous_guessed,
            "by_style": {k: f"{v[0]}/{v[1]}" for k, v in sorted(styles.items())},
        }
        if name == "model":
            results[name]["errors"] = errors
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write the results here as JSON")
    args = parser.parse_args()
    model = load_model()
    lexicon = load_lexicon()
    results = {
        "model": {k: model.meta[k] for k in ("name", "version", "trained", "dictionary_names", "rxnorm_version")},
        "unseen_misspellings": test_misspellings(model, lexicon),
        "look_alikes_ismp": test_lookalikes(model, lexicon),
        "ordinary_english": test_english(model),
    }
    blind = DATA / "blind_med_lists.json"
    if blind.exists():
        results["blind_lists"] = test_blind(model, blind)
    print(json.dumps(results, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
