"""Read a free-text medication list into the drugs Cascade can score.

People do not type "simvastatin". They type "Zocor 40 at night", "simvistatin",
"my dad's cholesterol pill (zocor) and the biaxin he just got". This module
turns that text into names from the FDA table, using a small trained model that
runs here, with no API key and nothing sent anywhere.

How it reads a list:

1. Split the text into segments (lines, commas, "and") and every segment into
   word windows of one to four words.
2. Retrieve, for each window, the closest names in a dictionary of about 2,400
   drug names: every drug in the FDA table, its US brand names from RxNorm,
   the 300 most prescribed US drugs and their brands. Closeness is cosine
   similarity of character n-gram TF-IDF vectors, so "simvistatin" still lands
   near "simvastatin".
3. Score each window-candidate pair with a trained neural network (one hidden
   layer, weights in data/reader_model.json, trained by train_reader.py on
   synthetic misspellings). It answers one question: is this the drug the
   person meant? Its inputs are string-similarity features and, importantly,
   how much better this candidate fits than the best candidate that means a
   different drug. That margin is how it learns to refuse on look-alike names
   (Celexa and Celebrex) instead of guessing.
4. Keep the best non-overlapping windows. A confident match becomes a drug; a
   match to a known medication outside the table is reported as not covered;
   two plausible drugs become a question for the user; anything weaker is
   left unread and visible, never silently dropped.

The network is the only learned part. Everything it needs is in the JSON file,
and inference uses only the standard library, so the evaluation numbers in
the README come from exactly the code that serves the page.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
RULES_VERSION = "1.1"   # 1.0 ran the blind test; 1.1 fixed two rule bugs it exposed (see README)
MODEL_PATH = DATA / "reader_model.json"
LEXICON_PATH = DATA / "english_words.txt.gz"   # the 50,000 commonest English words, from wordfreq

# --- normalisation ------------------------------------------------------------

# Words that describe how a drug is made or taken rather than which drug it is.
# They are dropped from both sides before matching, so "METOPROLOL SUCC ER 50MG
# TAB" and "metoprolol" compare as the same thing.
STRIP = frozenset("""
er xl xr sr cr la dr ec od odt cd xt sa ir hs
tab tabs tablet tablets cap caps capsule capsules oral po solution soln susp suspension
inj injection cream ointment gel patch spray inhaler inh drops chewable chew liquid
syrup elixir powder packet vial pen kit topical vaginal ophthalmic otic nasal
hcl hydrochloride succ succinate tart tartrate mesylate maleate besylate citrate
sulfate phosphate fumarate hbr hydrobromide acetate ethylsuccinate stearate
lactobionate polistirex nitrate bromide sodium na monohydrate
mg mcg g gm ml l iu unit units meq
""".split())

# A window never starts on one of these. They are the glue of a typed list:
# doses, schedules, people, and the words around them.
STOP = frozenset("""
a an the and or but with without plus also just only then than to of in on at by for from into per
as is are was were be been being has have had do does did not no yes
i me my mine we our us you your he him his she her they them their it its
take takes taking took taken start started starting stop stopped new old recently
mom mum dad mother father grandma grandpa wife husband son daughter patient pt
daily day days nightly night nights morning mornings evening evenings bedtime noon
week weeks weekly month monthly year once twice three four times every each other
bid tid qid qd qhs qam qpm prn hs am pm q d h x
mg mcg ml tab tabs tablet tablets cap caps capsule capsules pill pills dose doses
needed when before after meals meal food water mouth by sig dispense refill refills
for blood pressure sugar cholesterol pain heart sleep anxiety depression infection
""".split())

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.\-]*")
DOSE = re.compile(r"^\d+([.,]\d+)?[a-z%/]*$")
HARD_BREAK = re.compile(r"\n|\r|;|,|\||\u2022|\u00b7|\u25aa|\u25cf|\u25e6|\t")
SEGMENT_BREAK = re.compile(
    r"\n|\r|;|,|\||•|·|▪|●|◦|\t|"
    r"(?<=\w)\s*\+\s*(?=\w)|"
    r"\s(?:and|&|plus|also|w/|then)\s",
    re.IGNORECASE,
)


def norm(text: str) -> str:
    """Lowercase ASCII, apostrophes removed, anything else non-alphanumeric to space."""
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    text = text.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def key(text: str) -> str:
    """The comparison form of a name or a typed mention.

    Normalised, with dose numbers and STRIP words removed, unless removing them
    would leave nothing (so a drug literally named "sodium fluoride" survives).
    The same function is applied to dictionary names and to what people type.
    """
    words = norm(text).split()
    kept = [w for w in words if not DOSE.match(w) and w not in STRIP]
    if not kept:
        kept = [w for w in words if not DOSE.match(w)]
    return " ".join(kept)


# --- string similarity, written out so training and serving share one definition ---

def jaro_winkler(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    window = max(0, max(la, lb) // 2 - 1)
    flags_a, flags_b = [False] * la, [False] * lb
    matches = 0
    for i, ch in enumerate(a):
        lo, hi = max(0, i - window), min(lb, i + window + 1)
        for j in range(lo, hi):
            if not flags_b[j] and b[j] == ch:
                flags_a[i] = flags_b[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    transpositions, j = 0, 0
    for i in range(la):
        if flags_a[i]:
            while not flags_b[j]:
                j += 1
            if a[i] != b[j]:
                transpositions += 1
            j += 1
    m = float(matches)
    jaro = (m / la + m / lb + (m - transpositions / 2.0) / m) / 3.0
    prefix = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1.0 - jaro)


def osa_distance(a: str, b: str) -> int:
    """Edit distance counting insertions, deletions, substitutions and adjacent swaps."""
    la, lb = len(a), len(b)
    prev2 = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[lb]


_PHONETIC = [("ph", "f"), ("ck", "k"), ("qu", "kw"), ("q", "k"), ("c", "k"), ("x", "ks"),
             ("z", "s"), ("y", "i"), ("th", "t"), ("gh", "g"), ("wh", "w")]


def phonetic(text: str) -> str:
    """A crude sound-alike key: common spelling swaps, vowels dropped after the first letter."""
    s = text.replace(" ", "")
    for old, new in _PHONETIC:
        s = s.replace(old, new)
    if not s:
        return s
    tail = re.sub(r"[aeiou]", "", s[1:])
    out = s[0] + tail
    return re.sub(r"(.)\1+", r"\1", out)


def ngrams(text: str) -> dict:
    padded = f" {text} "
    counts: dict = {}
    for n in (2, 3, 4):
        for i in range(len(padded) - n + 1):
            g = padded[i:i + n]
            counts[g] = counts.get(g, 0) + 1
    return counts


FEATURES = ("cosine", "jaro_winkler", "edit_similarity", "edit_distance", "prefix", "first_letter",
            "last_two", "length_ratio", "sound_alike", "exact", "cosine_margin", "jw_margin",
            "length", "is_brand", "same_word_count", "englishness")


def pair_features(mention: str, cand: "Entry", cos: float, rival_cos: float, rival_jw: float,
                  englishness: float) -> list:
    """The network's inputs for one typed mention against one dictionary name.

    rival_cos and rival_jw are the best scores among candidates that mean a
    different drug; the margins against them carry the look-alike signal.
    englishness is how much the mention looks like ordinary English, from a
    separate character model, so "program" is not read as Prograf.
    """
    name = cand.key
    jw = jaro_winkler(mention, name)
    dist = osa_distance(mention, name)
    longest = max(len(mention), len(name)) or 1
    prefix = 0
    for x, y in zip(mention, name):
        if x != y:
            break
        prefix += 1
    return [
        cos,
        jw,
        1.0 - dist / longest,
        min(dist, 6) / 6.0,
        min(prefix, 6) / 6.0,
        1.0 if mention[:1] == name[:1] else 0.0,
        1.0 if mention[-2:] == name[-2:] else 0.0,
        min(len(mention), len(name)) / longest,
        1.0 if phonetic(mention) == phonetic(name) else 0.0,
        1.0 if mention == name else 0.0,
        cos - rival_cos,
        jw - rival_jw,
        min(len(mention), 20) / 20.0,
        1.0 if cand.kind == "brand" else 0.0,
        1.0 if mention.count(" ") == name.count(" ") else 0.0,
        englishness,
    ]


# --- the dictionary and the model -------------------------------------------

@lru_cache(maxsize=1)
def load_lexicon(path: str | None = None) -> frozenset:
    source = Path(path) if path else LEXICON_PATH
    if not source.exists():
        return frozenset()
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        return frozenset(line.strip() for line in handle if line.strip())


@dataclass(frozen=True)
class Entry:
    key: str            # comparison form
    display: str        # how to show the name
    kind: str           # generic, brand or alias
    drugs: tuple        # FDA-table drugs it contains (empty for medicines outside the table)
    others: tuple       # ingredients outside the table
    source: str
    exact: bool = False  # an ordinary English word ("smoker", "grapefruit"): matched only as typed


class Model:
    def __init__(self, blob: dict):
        self.meta = blob["meta"]
        self.thresholds = blob["thresholds"]
        self.idf = blob["idf"]
        self.default_idf = blob["default_idf"]
        self.mean = blob["scaler"]["mean"]
        self.scale = blob["scaler"]["scale"]
        self.layers = blob["network"]          # list of {"W": [[...]], "b": [...]}
        self.entries = [Entry(e["key"], e["display"], e["kind"], tuple(e["drugs"]),
                              tuple(e["others"]), e["source"], bool(e.get("exact")))
                        for e in blob["entries"]]
        self.k = blob.get("candidates", 8)
        self.lexicon = load_lexicon()
        self.keys = frozenset(e["key"] for e in blob["entries"])
        english = blob.get("english", {"weights": {}, "bias": 0.0})
        self.english_weights = english["weights"]
        self.english_bias = english["bias"]
        self._vectors = []
        self._index: dict = {}
        for i, entry in enumerate(self.entries):
            vec = self._vector(entry.key)
            self._vectors.append(vec)
            for g in vec:
                self._index.setdefault(g, []).append(i)

    def _vector(self, text: str) -> dict:
        counts = ngrams(text)
        vec = {g: c * self.idf.get(g, self.default_idf) for g, c in counts.items()}
        length = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {g: v / length for g, v in vec.items()}

    def retrieve(self, mention: str) -> list:
        """Top candidates by cosine, at most one per distinct dictionary key."""
        vec = self._vector(mention)
        scores: dict = {}
        for g, v in vec.items():
            for i in self._index.get(g, ()):
                scores[i] = scores.get(i, 0.0) + v * self._vectors[i][g]
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.entries[item[0]].key))
        return ranked[: self.k]

    def englishness(self, text: str) -> float:
        """Probability that a string is ordinary English rather than a drug name.

        A logistic model over which character 2- to 4-grams the string contains,
        trained on common English words against drug names and their misspellings.
        """
        z = self.english_bias + sum(self.english_weights.get(g, 0.0) for g in ngrams(text))
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def features_for(self, mention: str, ranked: list) -> list:
        """Feature rows for every retrieved candidate, with rival margins filled in."""
        rows = []
        english = self.englishness(mention)
        jws = [jaro_winkler(mention, self.entries[i].key) for i, _ in ranked]
        for pos, (i, cos) in enumerate(ranked):
            meaning = self.entries[i].drugs or self.entries[i].others
            rival_cos, rival_jw = 0.0, 0.0
            for other_pos, (j, other_cos) in enumerate(ranked):
                if other_pos == pos:
                    continue
                if (self.entries[j].drugs or self.entries[j].others) != meaning:
                    rival_cos = max(rival_cos, other_cos)
                    rival_jw = max(rival_jw, jws[other_pos])
            rows.append(pair_features(mention, self.entries[i], cos, rival_cos, rival_jw, english))
        return rows

    def probability(self, row: list) -> float:
        x = [(v - m) / s for v, m, s in zip(row, self.mean, self.scale)]
        for depth, layer in enumerate(self.layers):
            out = []
            for weights, bias in zip(layer["W"], layer["b"]):
                out.append(sum(w * v for w, v in zip(weights, x)) + bias)
            x = out if depth == len(self.layers) - 1 else [max(0.0, v) for v in out]
        z = max(-30.0, min(30.0, x[0]))
        return 1.0 / (1.0 + math.exp(-z))

    def score(self, mention: str) -> list:
        """All retrieved candidates for one mention, best first, as (probability, entry)."""
        if not mention:
            return []
        ranked = self.retrieve(mention)
        if not ranked:
            return []
        rows = self.features_for(mention, ranked)
        scored = []
        for row, (i, _) in zip(rows, ranked):
            entry = self.entries[i]
            # Fuzzy matching is for drug names. A name that is itself an English
            # word is only ever matched exactly, so "stoker" never becomes "smoker".
            p = 0.0 if entry.exact and mention != entry.key else self.probability(row)
            scored.append((p, entry))
        scored.sort(key=lambda item: (-item[0], item[1].key))
        return scored

    def plain_english(self, mention: str) -> bool:
        """Ordinary English that is not itself a dictionary name.

        A real word is not a typo. "pipeline" is one letter from piperine and
        "program" one from Prograf, and no string similarity can tell them apart;
        a word list can. Such words are never matched, only exact names are.
        """
        return mention not in self.keys and all(w in self.lexicon or len(w) == 1 for w in mention.split())

    def decide(self, mention: str) -> dict:
        """The model's verdict on one mention, before windows compete with each other."""
        if self.plain_english(mention):
            return {"status": "none", "p": 0.0}
        scored = self.score(mention)
        if not scored:
            return {"status": "none", "p": 0.0}
        p1, best = scored[0]
        meaning = best.drugs or best.others
        rival = next(((p, e) for p, e in scored[1:] if (e.drugs or e.others) != meaning), (0.0, None))
        t = self.thresholds
        # A reading that would change the analysis (a table drug) and one that only
        # adds a "not covered" note have separately chosen thresholds.
        accept = t["accept"] if best.drugs else t.get("accept_other", t["accept"])
        if p1 >= accept and p1 - rival[0] >= t["margin"]:
            status = "matched" if best.drugs else "not_covered"
            return {"status": status, "p": p1, "entry": best, "rival": rival}
        if (p1 >= t["ask"] and rival[1] is not None and rival[0] >= t["ask"] / 2
                and self.englishness(mention) <= t.get("ask_english", 1.0)):
            # Only ask about text that looks like a drug name, not about ordinary words.
            return {"status": "ambiguous", "p": p1, "entry": best, "rival": rival}
        if p1 >= accept:
            status = "matched" if best.drugs else "not_covered"
            return {"status": status, "p": p1, "entry": best, "rival": rival}
        return {"status": "none", "p": p1, "entry": best, "rival": rival}


@lru_cache(maxsize=1)
def load_model(path: str | None = None) -> Model:
    with open(Path(path) if path else MODEL_PATH, encoding="utf-8") as handle:
        return Model(json.load(handle))


# --- reading a whole list ---------------------------------------------------------

# Combinations the FDA table lists as one entry. A combination is present on a
# line when it is read directly (Kaletra, "lopinavir") or when all of its
# required parts are; its parts on that line are then not counted again.
COMBOS = {
    "lopinavir and ritonavir": ({"lopinavir", "ritonavir"}, {"lopinavir", "ritonavir"}),
    "lumacaftor and ivacaftor": ({"lumacaftor", "ivacaftor"}, {"lumacaftor", "ivacaftor"}),
    "saquinavir and ritonavir": ({"saquinavir", "ritonavir"}, {"saquinavir", "ritonavir"}),
    "tipranavir and ritonavir": ({"tipranavir", "ritonavir"}, {"tipranavir", "ritonavir"}),
    "indinavir and ritonavir": ({"indinavir", "ritonavir"}, {"indinavir", "ritonavir"}),
    "elvitegravir and ritonavir": ({"elvitegravir", "ritonavir"}, {"elvitegravir", "ritonavir"}),
    "paritaprevir and ritonavir and (ombitasvir and/or dasabuvir)": (
        {"paritaprevir", "ritonavir"}, {"paritaprevir", "ritonavir", "ombitasvir", "dasabuvir"}),
}


def _names(verdict: dict) -> set:
    if verdict["status"] not in ("matched", "not_covered"):
        return set()
    return set(verdict["entry"].drugs) | set(verdict["entry"].others)


def _merge_combinations(text: str, picks: list) -> list:
    """Fold the parts of a table combination written together into that combination.

    "Together" means between hard breaks (lines, commas, semicolons), so
    "lopinavir and ritonavir" is one combination but "Kaletra; Norvir" is two items.
    """
    breaks = [m.start() for m in HARD_BREAK.finditer(text)]
    clause_of = lambda pos: sum(1 for b in breaks if b < pos)  # noqa: E731
    lines: dict = {}
    for i, (_, start, _) in enumerate(picks):
        lines.setdefault(clause_of(start), []).append(i)
    drop, replace = set(), {}
    for idxs in lines.values():
        present = set().union(*(_names(picks[i][0]) for i in idxs))
        for combo, (required, parts) in COMBOS.items():
            direct = [i for i in idxs
                      if picks[i][0]["status"] == "matched" and combo in picks[i][0]["entry"].drugs]
            if not direct and not required <= present:
                continue
            members = [i for i in idxs if _names(picks[i][0]) and _names(picks[i][0]) <= parts | {combo}]
            if not members:
                continue
            keep = direct[0] if direct else members[0]
            drop.update(i for i in members if i != keep)
            if not direct:
                replace[keep] = (combo, min(picks[i][1] for i in members), max(picks[i][2] for i in members))
    out = []
    for i, (verdict, start, end) in enumerate(picks):
        if i in drop:
            continue
        if i in replace:
            combo, start, end = replace[i]
            merged = Entry(combo, combo, "generic", (combo,), (), "combination rule")
            verdict = dict(verdict, status="matched", entry=merged)
        out.append((verdict, start, end))
    return out


def _segments(text: str):
    """(start, end) character spans of the pieces of a list."""
    start = 0
    for match in SEGMENT_BREAK.finditer(text):
        if match.start() > start:
            yield start, match.start()
        start = match.end()
    if start < len(text):
        yield start, len(text)


def _tokens(text: str, lo: int, hi: int):
    for match in TOKEN.finditer(text, lo, hi):
        word = match.group(0).rstrip(".-'’")
        if word:
            yield match.start(), match.start() + len(word), word


def read_list(text: str, model: Model | None = None, max_words: int = 4) -> dict:
    """Read a pasted medication list. Returns what was read, what was not, and why."""
    model = model or load_model()
    text = str(text or "")[:5000]
    picks = []
    for lo, hi in _segments(text):
        tokens = list(_tokens(text, lo, hi))
        options = []
        for i, (start, _, word) in enumerate(tokens):
            first = norm(word)
            if not first or first in STOP or DOSE.match(first) or not first[0].isalpha():
                continue
            for size in range(1, max_words + 1):
                if i + size > len(tokens):
                    break
                end = tokens[i + size - 1][1]
                mention = key(text[start:end])
                if len(mention.replace(" ", "")) < 3:
                    continue
                if size > 1 and mention.split()[-1] in STOP and mention not in model.keys:
                    continue      # "Zyvox 600 mg BID" is a drug and a schedule, not a drug named "zyvox bid"
                verdict = model.decide(mention)
                if verdict["status"] == "none":
                    continue
                options.append((verdict, i, i + size, start, end))
        rank = {"matched": 2, "not_covered": 2, "ambiguous": 1}
        options.sort(key=lambda o: (-rank[o[0]["status"]], -o[0]["p"], -(o[2] - o[1]), o[1]))
        used = set()
        for verdict, a, b, start, end in options:
            if used.intersection(range(a, b)):
                continue
            used.update(range(a, b))
            picks.append((verdict, start, end))
    picks.sort(key=lambda p: p[1])

    picks = _merge_combinations(text, picks)

    items, drugs, not_covered, asks = [], [], [], []
    for verdict, start, end in picks:
        entry, rival = verdict["entry"], verdict.get("rival", (0.0, None))
        item = {
            "text": text[start:end],
            "start": start,
            "end": end,
            "status": verdict["status"],
            "confidence": round(verdict["p"], 3),
            "read_as": entry.display,
            "kind": entry.kind,
            "drugs": list(entry.drugs),
            "others": list(entry.others),
        }
        if verdict["status"] == "matched":
            for drug in entry.drugs:
                if drug not in drugs:
                    drugs.append(drug)
        elif verdict["status"] == "not_covered":
            for name in entry.others:
                if name not in not_covered:
                    not_covered.append(name)
        else:
            choices = [entry] + ([rival[1]] if rival[1] is not None else [])
            item["choices"] = [{"read_as": c.display, "drugs": list(c.drugs), "others": list(c.others)}
                               for c in choices]
            asks.append(item["text"])
        items.append(item)
    return {
        "items": items,
        "drugs": drugs,
        "not_covered": not_covered,
        "ambiguous": asks,
        "model": {"name": model.meta.get("name"), "version": model.meta.get("version"), "rules": RULES_VERSION},
    }
