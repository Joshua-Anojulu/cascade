"""Measure the medication-list reader on data it was never tuned on.

    python evaluate_reader.py
    python evaluate_reader.py --json reader-results.json
    python evaluate_reader.py --exclude-seen      # needs requirements-train.txt

Standard library only, except --exclude-seen. Four tests:

1. New-seed misspellings. Every dictionary name misspelled again with a new
   random seed and a harsher mix (more two- and three-letter errors) than
   training used. A new seed can still produce a string training also saw;
   --exclude-seen regenerates the training strings and scores only the rest.
2. Look-alike names. Pairs from the ISMP List of Confused Drug Names (2024), each
   name typed exactly and with typos. The count that matters is how often a name
   is read as its look-alike partner.
3. Ordinary English. Every word of three or more letters in the reader's
   English word list (from wordfreq's 50,000 commonest, minus drug names),
   typed alone. None should be read as a medicine.
4. Blind lists. data/blind_med_lists.json: 80 medication lists in five styles,
   written with answer keys by a separate agent that never saw the reader. Run
   once, after the model was frozen. Compared with Cascade's earlier resolver
   (exact names plus the aliases in synonyms.csv) two ways: one lookup per
   whole piece of the list, and one lookup per window using the reader's own
   windows, so the second comparison isolates what the model adds.
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

from core.reader import RULES_VERSION, STOP, _segments, _tokens, key, load_lexicon, load_model, read_list  # noqa: E402
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


def strings_built_on() -> set:
    """Every string train_reader.py trains or validates on, regenerated from its seeds.

    Needs requirements-train.txt. A fresh seed does not guarantee a fresh string:
    short names have few plausible misspellings, so some test strings also occur
    in training. Passing --exclude-seen scores only the ones that never do.
    """
    import train_reader as tr
    entries, _, _ = tr.build_dictionary()
    tr.LEXICON.clear()
    tr.LEXICON.update(tr.build_lexicon({e["key"] for e in entries}))
    held, held_train, held_valid = tr.hold_out(entries, random.Random(tr.SEED + 7))
    seen = [e for i, e in enumerate(entries) if i not in held]
    out = {t for t, _ in tr.mentions(seen, random.Random(tr.SEED + 1), 12, 5)}
    out |= {t for t, _ in tr.mentions(seen, random.Random(tr.SEED + 2), 4, 2)}
    out |= {row[0] for row in tr.unknown_rows(entries, held_train, random.Random(tr.SEED + 8))}
    out |= {row[0] for row in tr.unknown_rows(entries, held_valid, random.Random(tr.SEED + 9))}
    return out


def test_misspellings(model, lexicon, exclude: set | None = None) -> dict:
    rng = random.Random(SEED)
    is_word = lambda w: w in lexicon  # noqa: E731
    c = Counter()
    for entry in model.entries:
        truth = meaning_of(entry)
        for typed in variants(entry.key, rng, 5 if entry.drugs else 2, is_word=is_word,
                              weights_long=(40, 40, 20), weights_short=(60, 40, 0)):
            if exclude is not None and typed in exclude:
                continue
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
    """Cascade's resolver before the model, one lookup per whole piece of the list.

    Exact names and synonyms.csv; a piece is a line or a comma-, semicolon- or
    "and"-separated part, with doses and form words stripped. "Zocor 40 at night"
    is one piece, so it is not found.
    """
    import re
    drugs = []
    for piece in re.split(r"\n|;|,|\||\s(?:and|&|plus)\s|\+", text):
        r = resolve(key(piece))
        if r["drug"] and r["drug"] not in drugs:
            drugs.append(r["drug"])
    return drugs


def windowed_exact_read(text: str) -> list:
    """The reader's own segments and 1- to 4-word windows, but exact lookups only.

    Same resolver as baseline_read (exact names and synonyms.csv), longest
    window first, no overlaps. This is the fairer comparison: it shows what the
    windowing alone recovers, so the gap to the model is what the model adds.
    """
    drugs = []
    for lo, hi in _segments(text):
        tokens = list(_tokens(text, lo, hi))
        used = set()
        for size in range(4, 0, -1):
            for i in range(len(tokens) - size + 1):
                if used.intersection(range(i, i + size)):
                    continue
                r = resolve(key(text[tokens[i][0]:tokens[i + size - 1][1]]))
                if r["drug"]:
                    used.update(range(i, i + size))
                    if r["drug"] not in drugs:
                        drugs.append(r["drug"])
    return drugs


def test_blind(model, path: Path) -> dict:
    lists = json.load(open(path, encoding="utf-8"))
    results = {}
    readers = (("model", lambda t: read_list(t, model)["drugs"]),
               ("exact_lookup_whole_pieces", baseline_read),
               ("exact_lookup_same_windows", windowed_exact_read))
    for name, reader in readers:
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
            # Strict: a list-writer marked some mentions as genuinely ambiguous.
            # Picking one of their candidates counts as an error here, not a pass.
            "precision_strict": round(tp / max(1, tp + fp + ambiguous_guessed), 4),
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
    parser.add_argument("--exclude-seen", action="store_true",
                        help="also score only the misspellings that never occur in training or "
                             "validation (needs requirements-train.txt)")
    args = parser.parse_args()
    model = load_model()
    lexicon = load_lexicon()
    results = {
        "model": {k: model.meta[k] for k in ("name", "version", "trained", "dictionary_names", "rxnorm_version")},
        "rules": RULES_VERSION,
        "new_seed_misspellings": test_misspellings(model, lexicon),
    }
    if args.exclude_seen:
        results["new_seed_misspellings_never_in_training"] = test_misspellings(model, lexicon, strings_built_on())
    results["look_alikes_ismp"] = test_lookalikes(model, lexicon)
    results["ordinary_english"] = test_english(model)
    blind = DATA / "blind_med_lists.json"
    if blind.exists():
        results["blind_lists"] = test_blind(model, blind)
    print(json.dumps(results, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
