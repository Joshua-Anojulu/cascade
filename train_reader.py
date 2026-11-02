"""Build the medication-list reader: dictionary, synthetic misspellings, network.

    python train_reader.py

Inputs, all in data/:
  cyp_roles.csv         the FDA table; its drugs are the only ones Cascade can score
  synonyms.csv          the project's existing brand aliases
  rxnorm_extract.json   branded drug names from RxNorm via the RxNav API, retrieved
                        2026-11-02, for every table drug and for the ClinCalc Top 300
                        of 2024 (the most prescribed US drugs)
  hand_aliases.csv      a short list of spellings RxNorm does not give, each with a reason

Writes data/reader_model.json (everything core/reader.py needs) and
data/reader_dictionary.csv (the same dictionary, readable).

Training uses scikit-learn, numpy and wordfreq (requirements-train.txt). Reading
lists with the result needs only the standard library. Every random choice is
seeded, so the same inputs give the same model.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import core.reader as reader  # noqa: E402
from core.reader import FEATURES, STOP, Model, key, ngrams, osa_distance  # noqa: E402
from core.typos import variants as typo_variants  # noqa: E402

DATA = ROOT / "data"
SEED = 20261102
CANDIDATES = 8
TARGET_WRONG_RATE = 0.005   # at most 1 in 200 accepted readings may name the wrong drug

# --- the dictionary ----------------------------------------------------------------

TABLE = sorted({row["drug"] for row in csv.DictReader(open(DATA / "cyp_roles.csv", encoding="utf-8"))})
TABLE_SET = set(TABLE)

# Ingredient names that map onto a table row under the project's existing conventions.
CONVENTIONS = {"venlafaxine": "r-venlafaxine", "esomeprazole": "omeprazole",
               "mephenytoin": "s-mephenytoin", "peginterferon alfa-2a": "peginterferon alpha-2a"}
COMBINATIONS = [
    ("lopinavir and ritonavir", {"lopinavir", "ritonavir"}),
    ("lumacaftor and ivacaftor", {"lumacaftor", "ivacaftor"}),
    ("saquinavir and ritonavir", {"saquinavir", "ritonavir"}),
    ("tipranavir and ritonavir", {"tipranavir", "ritonavir"}),
    ("indinavir and ritonavir", {"indinavir", "ritonavir"}),
    ("elvitegravir and ritonavir", {"elvitegravir", "ritonavir"}),
    ("paritaprevir and ritonavir and (ombitasvir and/or dasabuvir)", {"paritaprevir", "ritonavir"}),
]
SALTS = ("hydrochloride hydrobromide sodium potassium calcium acetate maleate mesylate besylate "
         "succinate tartrate bitartrate citrate sulfate phosphate fumarate lactobionate "
         "ethylsuccinate stearate nitrate polistirex gluconate monohydrate macrocrystals "
         "anhydrous hyclate dihydrate propionate").split()
JUNK_PREFIX = re.compile(r"^(sprinkle|hr|ml|pulmonary hypertension|hyponatremia)\s+")
SKIP_OTHER = {"lactate", "sodium salts", "magnesium salts", "calcium"}
# Table rows that are habits and foods rather than drug names: matched only as written.
LIFESTYLE = {"tobacco (smoking)", "grapefruit juice", "oral contraceptives"}
# Veterinary and aquarium products RxNorm lists under human ingredients. Nobody
# will type them into a human medication list, and several are near English words.
VETERINARY = {k.lower() for k in """Otomax Mometamax Mometavet Motazol Posatex Simplera Duotic Osurnia
Klentz Florfinasone Euthasol Euthaphen Fidoquel Optimmune Atopica Cyclavance Sporimune Modulis
Felycin Mal-A-Ket Ketochlor Keto-C T8 Ketodan Malacetic Malaseb MiconaHex+Triz Surolan Equisul
Uniprim Thyro-Tabs ThyroKare Maracyn Suffusion""".split()}
VETERINARY_INGREDIENTS = {"florfenicol", "orbifloxacin", "pentobarbital"}
COMMON_WORD = 3.5   # Zipf frequency: roughly once per 300,000 words of English
LEXICON_SIZE = 50000  # the commonest English words: never fuzzy-matched at reading time
LEXICON: set = set()


def english_word(word: str) -> bool:
    return zipf_frequency(word, "en") >= COMMON_WORD


def keep_brand(brand: str, ingredients, table: bool) -> bool:
    """Drop products nobody types into a human list, and names that are ordinary English."""
    first = brand.split()[0].lower()
    if first in VETERINARY or brand.lower() in VETERINARY:
        return False
    if any(i.lower().split()[0] in VETERINARY_INGREDIENTS for i in ingredients if i.strip()):
        return False
    words = key(brand).split()
    if len(words) > 1 and any(english_word(w) for w in words[1:]):
        return False          # "Carmol Scalp Treatment", "Bromfed DM": the head word stays if listed alone
    if not table and len(words) == 1 and english_word(words[0]):
        return False          # an outside-the-table brand that is an English word ("Supreme")
    return True


def clean_ingredient(name: str) -> str:
    text = JUNK_PREFIX.sub("", name.lower().strip()).split(",")[0].strip()
    if text.startswith("tenofovir"):
        return "tenofovir"
    words = text.split()
    while len(words) > 1 and words[-1] in SALTS and len(" ".join(words[:-1])) >= 5:
        words = words[:-1]
    return " ".join(words)


def map_ingredients(ingredients) -> tuple:
    names = {clean_ingredient(i) for i in ingredients if i.strip()}
    drugs, others, used = set(), set(), set()
    for combo, needed in COMBINATIONS:
        if needed <= names:
            drugs.add(combo)
            used |= needed
    for name in names - used:
        target = CONVENTIONS.get(name, name)
        (drugs if target in TABLE_SET else others).add(target if target in TABLE_SET else name)
    return tuple(sorted(drugs)), tuple(sorted(others))


PACKAGING = re.compile(r"(\s*\(|\s+\d|\s+(tri-fold|pack|kit|carton|reformulated|sample|titration|"
                       r"initiation|starting|maintenance|dose|combination)\b).*$", re.IGNORECASE)


def clean_brand(brand: str) -> str:
    text = re.sub(r"\s+", " ", brand).strip()
    text = PACKAGING.sub("", text).strip(" -") or text
    return text.title() if text.isupper() and len(text) > 4 else text


def build_dictionary() -> tuple:
    entries: dict = {}
    conflicts = []

    def add(display, kind, drugs, others, source, exact=False):
        k = key(display)
        if not k or (not drugs and not others):
            return
        if kind != "generic" and len(k.replace(" ", "")) < 4:
            return
        meaning = (tuple(sorted(drugs)), tuple(sorted(others)))
        if k in entries:
            old = entries[k]
            if (tuple(old["drugs"]), tuple(old["others"])) != meaning:
                if not old["drugs"] and drugs:          # a table reading wins over "outside the table"
                    old.update(drugs=list(meaning[0]), others=list(meaning[1]))
                conflicts.append((k, old["source"], source))
            return
        entries[k] = {"key": k, "display": display, "kind": kind, "drugs": list(meaning[0]),
                      "others": list(meaning[1]), "source": source,
                      "exact": exact or all(english_word(w) for w in k.split())}

    for drug in TABLE:
        add(drug, "generic", (drug,), (), "FDA table", exact=drug in LIFESTYLE)
    for row in csv.DictReader(open(DATA / "hand_aliases.csv", encoding="utf-8")):
        drugs = tuple(d for d in row["drugs"].split(";") if d)
        others = tuple(o for o in row["others"].split(";") if o)
        assert all(d in TABLE_SET for d in drugs), row
        add(row["alias"], row["kind"], drugs, others, "hand: " + row["why"], exact=row["match"] == "exact")
    for row in csv.DictReader(open(DATA / "synonyms.csv", encoding="utf-8")):
        brand = row["alias"] != "rifampicin"
        add(row["alias"].title() if brand else row["alias"], "brand" if brand else "alias", (row["drug"],), (),
            "synonyms.csv")
    extract = json.load(open(DATA / "rxnorm_extract.json", encoding="utf-8"))
    for query, brands in extract["table"]:
        for brand, ingredients in brands:
            if not keep_brand(clean_brand(brand), ingredients, table=True):
                continue
            drugs, others = map_ingredients(ingredients)
            add(clean_brand(brand), "brand", drugs, others, "RxNorm brand")
    for query, brands in extract["other_top300"]:
        name = clean_ingredient(query)
        if name in SKIP_OTHER:
            continue
        drugs, others = map_ingredients([query])
        add(name, "generic", drugs, others, "ClinCalc Top 300")
        for brand, ingredients in brands:
            drugs, others = map_ingredients(ingredients)
            if not keep_brand(clean_brand(brand), ingredients, table=bool(drugs)):
                continue
            add(clean_brand(brand), "brand", drugs, others, "RxNorm brand")
    # Every other ingredient named in those products is a medicine too, if not an
    # ordinary English word: knowing "ezetimibe" keeps it from being read as something else.
    for _, brands in extract["table"] + extract["other_top300"]:
        for _, ingredients in brands:
            for name in map_ingredients(ingredients)[1]:
                if not all(english_word(w) for w in name.split()) and len(name) >= 4:
                    add(name, "generic", (), (name,), "RxNorm ingredient")
    return [entries[k] for k in sorted(entries)], conflicts, extract


def make_blob(entries, network=None, scaler=None, thresholds=None, meta=None, english=None) -> dict:
    df = Counter()
    for e in entries:
        df.update(set(ngrams(e["key"])))
    n = len(entries)
    idf = {g: round(math.log((1 + n) / (1 + c)) + 1.0, 5) for g, c in sorted(df.items())}
    return {
        "meta": meta or {"name": "cascade-reader"},
        "thresholds": thresholds or {"accept": 0.5, "accept_other": 0.5, "margin": 0.0, "ask": 0.3},
        "idf": idf,
        "default_idf": round(math.log(1 + n) + 1.0, 5),
        "english": english or {"weights": {}, "bias": 0.0},
        "scaler": scaler or {"mean": [0.0] * len(FEATURES), "scale": [1.0] * len(FEATURES)},
        "network": network or [{"W": [[0.0] * len(FEATURES)], "b": [0.0]}],
        "candidates": CANDIDATES,
        "entries": entries,
    }


# --- misspellings ---------------------------------------------------------------------

def variants(text: str, rng: random.Random, count: int) -> list:
    """Misspellings of a name (core/typos.py), minus any that land on a common English word."""
    return typo_variants(text, rng, count, is_word=lambda w: w in LEXICON or english_word(w))


def meaning(entry: dict) -> tuple:
    return tuple(entry["drugs"]) or ("other:" + "+".join(entry["others"]),)


def entry_meaning(entry) -> tuple:
    return tuple(entry.drugs) or ("other:" + "+".join(entry.others),)


def mentions(entries, rng, table_copies, other_copies) -> list:
    """(typed text, true meaning) pairs: each name once exactly, then misspelled."""
    rows = []
    for e in entries:
        rows.append((e["key"], meaning(e)))
        for v in variants(e["key"], rng, table_copies if e["drugs"] else other_copies):
            rows.append((v, meaning(e)))
    return rows


def build_lexicon(keys: set) -> set:
    """The commonest English words, minus anything that is itself a dictionary name."""
    return {w for w in top_n_list("en", LEXICON_SIZE)
            if w.isalpha() and w.isascii() and len(w) >= 2 and w not in keys}


def english_words(keys: set, rng: random.Random) -> list:
    """Rarer English words and names, beyond the lexicon: the ones the network must judge."""
    words = [w for w in top_n_list("en", 120000)
             if w.isalpha() and w.isascii() and len(w) >= 3 and w not in STOP
             and w not in keys and w not in LEXICON]
    rng.shuffle(words)
    return words


def hard_negatives(model: Model, words: list) -> list:
    """English words that sit close to some dictionary name: the ones that teach the boundary."""
    hard = []
    for w in words:
        ranked = model.retrieve(w)[:3]
        if not ranked:
            continue
        if ranked[0][1] >= 0.45 or any(osa_distance(w, model.entries[i].key) <= 2 for i, _ in ranked):
            hard.append(w)
    return hard


def english_phrases(words: list, rng: random.Random, count: int) -> list:
    """Two-word windows of ordinary text with at least one rarer word in them."""
    common = sorted(LEXICON)
    out = []
    while len(out) < count:
        pair = [rng.choice(words), rng.choice(common)]
        rng.shuffle(pair)
        out.append(" ".join(pair))
    return out


# --- the English character model -------------------------------------------------------

def train_english(english: list, entries: list, rng: random.Random) -> tuple:
    """Logistic regression on character n-grams: ordinary English (1) against drug names (0).

    L1 regularisation keeps only the n-grams that carry signal, so the weights
    shipped in the model file stay small.
    """
    names = [e["key"] for e in entries]
    noisy = [v for e in entries for v in variants(e["key"], rng, 3)]
    texts = english + names + noisy
    labels = [1] * len(english) + [0] * (len(names) + len(noisy))
    vec = DictVectorizer()
    X = vec.fit_transform([{g: 1 for g in ngrams(t)} for t in texts]).tocsr()
    X.indices, X.indptr = X.indices.astype(np.int32), X.indptr.astype(np.int32)  # liblinear needs 32-bit
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000,
                             random_state=SEED).fit(X, labels)
    weights = {g: round(float(w), 4) for g, w in zip(vec.get_feature_names_out(), clf.coef_[0])
               if abs(w) >= 1e-4}
    return {"weights": dict(sorted(weights.items())), "bias": round(float(clf.intercept_[0]), 4)}, clf, vec


# --- features, training, thresholds ----------------------------------------------------

# Every training or validation row is (typed text, true meaning, kind), where kind is
# "known" (a dictionary name, maybe misspelled), "unknown" (a real medicine name held
# out of the dictionary the network sees) or "english" (ordinary words; meaning None).

def featurise(model: Model, rows) -> tuple:
    X, y, groups = [], [], []
    for g, (text, truth, _) in enumerate(rows):
        ranked = model.retrieve(text)
        if not ranked:
            continue
        feats = model.features_for(text, ranked)
        for (i, _), f in zip(ranked, feats):
            X.append(f)
            y.append(1 if truth is not None and entry_meaning(model.entries[i]) == truth else 0)
            groups.append(g)
    return np.array(X, dtype=float), np.array(y), np.array(groups)


def mention_table(model: Model, rows, probs, groups) -> dict:
    """Per row: best probability and meaning, and the best rival that means something else."""
    by_group: dict = {}
    for p, g in zip(probs, groups):
        by_group.setdefault(int(g), []).append(float(p))
    cols = {k: [] for k in ("kind", "p1", "pr", "best_table", "best_right", "ask_ok", "has_rival", "english",
                            "blocked")}
    for g, (text, truth, kind) in enumerate(rows):
        ps = by_group.get(g)
        if ps:
            ranked = model.retrieve(text)
            gated = [(0.0 if model.entries[i].exact and text != model.entries[i].key else p, model.entries[i])
                     for p, (i, _) in zip(ps, ranked)]
            scored = sorted(gated, key=lambda item: (-item[0], item[1].key))
            p1, best = scored[0]
            rival = next(((p, e) for p, e in scored[1:] if entry_meaning(e) != entry_meaning(best)), (0.0, None))
        else:
            p1, best, rival = 0.0, None, (0.0, None)
        m1 = entry_meaning(best) if best else None
        mr = entry_meaning(rival[1]) if rival[1] else None
        cols["kind"].append(kind)
        cols["p1"].append(p1)
        cols["pr"].append(rival[0])
        cols["best_table"].append(bool(best and best.drugs))
        cols["best_right"].append(truth is not None and m1 == truth)
        cols["ask_ok"].append(truth is not None and truth in (m1, mr))
        cols["has_rival"].append(rival[1] is not None)
        cols["english"].append(model.englishness(text))
        cols["blocked"].append(model.plain_english(text))
    return {k: np.array(v) for k, v in cols.items()}


def outcomes(t: dict, accept: float, accept_other: float, margin: float, ask: float,
             ask_english: float = 1.0) -> dict:
    """Vectorised replay of Model.decide over a validation set."""
    threshold = np.where(t["best_table"], accept, accept_other)
    open_ = ~t["blocked"]
    sure = open_ & (t["p1"] >= threshold) & (t["p1"] - t["pr"] >= margin)
    asks = open_ & ~sure & (t["p1"] >= ask) & t["has_rival"] & (t["pr"] >= ask / 2) & (t["english"] <= ask_english)
    accepted = sure | (open_ & ~sure & ~asks & (t["p1"] >= threshold))
    c = {}
    for kind in ("known", "unknown", "english"):
        k = t["kind"] == kind
        c[kind] = int(k.sum())
        c[kind + "_right"] = int((accepted & k & t["best_right"]).sum())
        c[kind + "_wrong_table"] = int((accepted & k & ~t["best_right"] & t["best_table"]).sum())
        c[kind + "_wrong_other"] = int((accepted & k & ~t["best_right"] & ~t["best_table"]).sum())
        c[kind + "_ask_ok"] = int((asks & k & t["ask_ok"]).sum())
        c[kind + "_ask_bad"] = int((asks & k & ~t["ask_ok"]).sum())
        c[kind + "_none"] = int((~accepted & ~asks & k).sum())
    return c


def summary(c: dict) -> dict:
    def share(kind, *parts):
        return round(sum(c[kind + "_" + p] for p in parts) / max(1, c[kind]), 5)
    known_accepted = c["known_right"] + c["known_wrong_table"] + c["known_wrong_other"]
    return {
        "typed_names": c["known"],
        "read_correctly": share("known", "right"),
        "read_as_wrong_drug": share("known", "wrong_table", "wrong_other"),
        "wrong_share_of_accepted": round((c["known_wrong_table"] + c["known_wrong_other"]) / max(1, known_accepted), 5),
        "asked": share("known", "ask_ok", "ask_bad"),
        "left_unread": share("known", "none"),
        "unknown_names": c["unknown"],
        "unknown_read_as_table_drug": share("unknown", "wrong_table"),
        "unknown_read_as_other_medicine": share("unknown", "wrong_other"),
        "unknown_left_unread_or_asked": share("unknown", "none", "ask_ok", "ask_bad"),
        "english_windows": c["english"],
        "english_read_as_table_drug": share("english", "wrong_table"),
        "english_read_as_other_medicine": share("english", "wrong_other"),
        "english_asked_about": share("english", "ask_ok", "ask_bad"),
    }


# Limits the thresholds must respect on validation; within them, correct readings are maximised.
LIMITS = {"wrong_share_of_accepted": TARGET_WRONG_RATE,
          "unknown_read_as_table_drug": 0.01,
          "english_read_as_table_drug": 0.002,
          "english_read_as_other_medicine": 0.003,
          "english_asked_about": 0.005}


def choose_thresholds(t: dict) -> tuple:
    best = None
    grid = [x / 100 for x in range(40, 100)]
    for accept in grid:
        for accept_other in grid[::2]:
            for margin in (0.0, 0.1, 0.2, 0.3):
                for ask in (0.3, 0.4, 0.5, 0.6):
                    if ask >= min(accept, accept_other):
                        continue
                    for ask_english in (0.2, 0.35, 0.5, 1.0):
                        c = outcomes(t, accept, accept_other, margin, ask, ask_english)
                        s = summary(c)
                        if any(s[k] > v for k, v in LIMITS.items()):
                            continue
                        score = (c["known_right"], -c["unknown_wrong_table"], -c["known_ask_bad"], -accept)
                        if best is None or score > best[0]:
                            best = (score, {"accept": accept, "accept_other": accept_other, "margin": margin,
                                            "ask": ask, "ask_english": ask_english}, c)
    if best is None:
        return None, None
    return best[1], best[2]


def hold_out(entries: list, rng: random.Random) -> tuple:
    """Real names hidden from the network's dictionary, to stand in for medicines it will not know.

    A quarter of the names outside the table, and a sixth of the table brands
    (never a table generic), are removed. Typed exactly or misspelled, they
    teach the network that a drug-like name near a known one is not that one.
    """
    pool = [i for i, e in enumerate(entries)
            if not e["drugs"] or (e["kind"] == "brand" and e["source"] == "RxNorm brand")]
    other = [i for i in pool if not entries[i]["drugs"]]
    brands = [i for i in pool if entries[i]["drugs"]]
    held = set(rng.sample(other, len(other) // 4)) | set(rng.sample(brands, len(brands) // 6))
    held_list = sorted(held)
    rng.shuffle(held_list)
    cut = int(len(held_list) * 0.7)
    return held, held_list[:cut], held_list[cut:]


def unknown_rows(entries, idx, rng) -> list:
    rows = []
    for i in idx:
        e = entries[i]
        rows.append((e["key"], meaning(e), "unknown"))
        rows += [(v, meaning(e), "unknown") for v in variants(e["key"], rng, 2)]
    return rows


def main() -> None:
    entries, conflicts, extract = build_dictionary()
    keys = {e["key"] for e in entries}
    print(f"dictionary: {len(entries)} names, {sum(1 for e in entries if e['drugs'])} read as table drugs, "
          f"{len(conflicts)} duplicate keys merged")

    LEXICON.clear()
    LEXICON.update(build_lexicon(keys))
    with gzip.GzipFile(DATA / "english_words.txt.gz", "wb", mtime=0) as handle:
        handle.write(("\n".join(sorted(LEXICON)) + "\n").encode("utf-8"))
    reader.load_lexicon.cache_clear()
    print(f"English word list: {len(LEXICON)} words")

    words = english_words(keys, random.Random(SEED))
    half = len(words) // 2
    english_train, network_words = words[:half], words[half:]
    common_sample = random.Random(SEED + 4).sample(sorted(LEXICON), 20000)
    english, eng_clf, eng_vec = train_english(
        common_sample + english_train + english_phrases(english_train, random.Random(SEED + 5), 4000),
        entries, random.Random(SEED + 6))
    print(f"English character model: {len(english['weights'])} n-gram weights")

    held, held_train, held_valid = hold_out(entries, random.Random(SEED + 7))
    seen = [e for i, e in enumerate(entries) if i not in held]
    probe = Model(make_blob(seen, english=english))
    print(f"held out of the network's dictionary: {len(held)} real names "
          f"({len(held_train)} for training, {len(held_valid)} for validation)")

    neg_rng = random.Random(SEED + 3)
    pool = network_words[:-4000]
    hard = hard_negatives(probe, pool)
    hard_set = set(hard)
    easy = [w for w in pool if w not in hard_set][:10000]
    neg_train = hard + easy + english_phrases(pool, neg_rng, 3000)
    neg_valid = network_words[-4000:] + english_phrases(network_words[-4000:], neg_rng, 1000)
    print(f"English negatives for the network: {len(hard)} near a drug name, {len(easy)} random")

    train_rows = [(t, m, "known") for t, m in mentions(seen, random.Random(SEED + 1), 12, 5)]
    train_rows += unknown_rows(entries, held_train, random.Random(SEED + 8))
    train_rows += [(w, None, "english") for w in neg_train]
    valid_rows = [(t, m, "known") for t, m in mentions(seen, random.Random(SEED + 2), 4, 2)]
    valid_rows += unknown_rows(entries, held_valid, random.Random(SEED + 9))
    valid_rows += [(w, None, "english") for w in neg_valid]
    print(f"training rows: {len(train_rows)}, validation rows: {len(valid_rows)}")

    X, y, _ = featurise(probe, train_rows)
    Xv, yv, gv = featurise(probe, valid_rows)
    scaler = StandardScaler().fit(X)
    net = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=1e-4, max_iter=400,
                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                        random_state=SEED).fit(scaler.transform(X), y)
    linear = LogisticRegression(max_iter=2000).fit(scaler.transform(X), y)
    print(f"pairs: {len(y)} train ({int(y.sum())} correct), {len(yv)} validation")

    tables = {name: mention_table(probe, valid_rows, clf.predict_proba(scaler.transform(Xv))[:, 1], gv)
              for name, clf in (("network", net), ("linear", linear))}
    thresholds, counts = choose_thresholds(tables["network"])
    if thresholds is None:
        raise SystemExit("no thresholds meet LIMITS on validation; loosen them or improve the model")
    network_summary = summary(counts)
    print("network on validation:", thresholds, network_summary)
    lin_thresholds, lin_counts = choose_thresholds(tables["linear"])
    if lin_thresholds is None:     # compare at the network's thresholds instead
        lin_thresholds, lin_counts = "none meet the limits; shown at the network's", outcomes(tables["linear"], **thresholds)
    print("linear baseline on validation:", lin_thresholds, summary(lin_counts))

    layers = [{"W": [[round(float(v), 6) for v in row] for row in W.T], "b": [round(float(v), 6) for v in b]}
              for W, b in zip(net.coefs_, net.intercepts_)]
    meta = {
        "name": "cascade-reader",
        "version": "1.0",
        "trained": "2026-11-02",
        "seed": SEED,
        "features": list(FEATURES),
        "network": "one hidden layer of 32 ReLU units, logistic output (scikit-learn MLPClassifier)",
        "english_model": "L1 logistic regression on character 2- to 4-grams",
        "english_word_list": f"{len(LEXICON)} commonest English words from wordfreq; never fuzzy-matched",
        "dictionary_names": len(entries),
        "table_names": sum(1 for e in entries if e["drugs"]),
        "rxnorm_version": extract["rxnorm_version"],
        "held_out_names": len(held),
        "training_rows": len(train_rows),
        "training_pairs": int(len(y)),
        "limits": LIMITS,
        "validation": network_summary,
        "validation_linear_baseline": summary(lin_counts),
        "linear_baseline_thresholds": lin_thresholds,
    }
    blob = make_blob(entries, network=layers,
                     scaler={"mean": [round(float(v), 6) for v in scaler.mean_],
                             "scale": [round(float(v), 6) for v in scaler.scale_]},
                     thresholds=thresholds, meta=meta, english=english)
    with open(DATA / "reader_model.json", "w", encoding="utf-8") as handle:
        json.dump(blob, handle, separators=(",", ":"))
    with open(DATA / "reader_dictionary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "kind", "table_drugs", "other_ingredients", "source"])
        for e in entries:
            writer.writerow([e["display"], e["kind"], ";".join(e["drugs"]), ";".join(e["others"]), e["source"]])
    print("wrote data/reader_model.json and data/reader_dictionary.csv")


if __name__ == "__main__":
    main()
