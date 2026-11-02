"""Realistic misspellings of drug names, for training and for testing the reader.

Typos are made the way people make them: a dropped, doubled or swapped letter, a
neighbouring key, a wrong vowel, a spelling that sounds the same ("ph" for "f",
"ine" for "ene"), or a word cut short. The first letter is left alone, as it
nearly always is. A misspelling that happens to be a real English word is
discarded: it is no longer evidence for the drug.
"""
from __future__ import annotations

import random
import re

KEYBOARD = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etdf", "t": "ryfg", "y": "tugh", "u": "yihj",
    "i": "uojk", "o": "ipkl", "p": "ol", "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop", "z": "asx", "x": "zsdc",
    "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}
VOWELS = "aeiou"
SOUNDS = [("ph", "f"), ("f", "ph"), ("c", "k"), ("k", "c"), ("s", "z"), ("z", "s"), ("y", "i"),
          ("i", "y"), ("x", "z"), ("ine", "ene"), ("ine", "in"), ("ide", "id"), ("ol", "al"),
          ("am", "em"), ("ck", "k"), ("qu", "kw"), ("ei", "ie"), ("ie", "ei"), ("tion", "sion")]


def misspell(text: str, rng: random.Random, edits: int) -> str:
    s = text
    for _ in range(edits):
        if len(s) < 4:
            break
        op = rng.choices(["delete", "insert", "keyboard", "vowel", "swap", "double", "undouble",
                          "sound", "truncate", "space"],
                         weights=[16, 10, 12, 16, 10, 7, 7, 14, 4, 4])[0]
        i = rng.randrange(1, len(s))           # people rarely get the first letter wrong
        ch = s[i]
        if op == "delete" and ch != " ":
            s = s[:i] + s[i + 1:]
        elif op == "insert":
            s = s[:i] + rng.choice(KEYBOARD.get(ch, "aeiou")) + s[i:]
        elif op == "keyboard" and ch in KEYBOARD:
            s = s[:i] + rng.choice(KEYBOARD[ch]) + s[i + 1:]
        elif op == "vowel" and ch in VOWELS:
            s = s[:i] + rng.choice(VOWELS.replace(ch, "")) + s[i + 1:]
        elif op == "swap" and i < len(s) - 1 and " " not in s[i:i + 2]:
            s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        elif op == "double" and ch.isalpha():
            s = s[:i] + ch + s[i:]
        elif op == "undouble":
            m = re.search(r"([a-z])\1", s)
            if m:
                s = s[:m.start()] + s[m.start() + 1:]
        elif op == "sound":
            options = [(a, b) for a, b in SOUNDS if a in s[1:]]
            if options:
                a, b = rng.choice(options)
                at = s.find(a, 1)
                s = s[:at] + b + s[at + len(a):]
        elif op == "truncate" and len(s) > 7:
            s = s[: len(s) - rng.choice([1, 2])]
        elif op == "space":
            s = s.replace(" ", "", 1) if " " in s else (s[:i] + " " + s[i:] if 3 < i < len(s) - 3 else s)
    return re.sub(r"\s+", " ", s).strip()


def variants(text: str, rng: random.Random, count: int, is_word=lambda w: False,
             weights_long=(55, 33, 12), weights_short=(75, 25, 0)) -> list:
    """Up to `count` distinct misspellings of `text`, none of them a real word."""
    out, tries = [], 0
    while len(out) < count and tries < count * 8:
        tries += 1
        edits = rng.choices([1, 2, 3], weights=weights_long if len(text) >= 9 else weights_short)[0]
        v = misspell(text, rng, edits)
        if v == text or len(v.replace(" ", "")) < 3 or v in out:
            continue
        if all(is_word(w) for w in v.split()):
            continue
        out.append(v)
    return out
