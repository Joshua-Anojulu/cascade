"""Load the FDA role table and resolve what a user typed into a listed drug."""
from __future__ import annotations

import csv
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from .model import Role

DATA = Path(__file__).resolve().parent.parent / "data"


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@lru_cache(maxsize=1)
def load_roles(path: str | None = None) -> tuple[Role, ...]:
    source = Path(path) if path else DATA / "cyp_roles.csv"
    with open(source, newline="", encoding="utf-8") as handle:
        return tuple(
            Role(row["drug"], row["enzyme"], row["role"], row["strength"])
            for row in csv.DictReader(handle)
        )


@lru_cache(maxsize=1)
def load_synonyms(path: str | None = None) -> dict[str, str]:
    """Brand names and spellings that map to a drug in the FDA table."""
    source = Path(path) if path else DATA / "synonyms.csv"
    if not source.exists():
        return {}
    with open(source, newline="", encoding="utf-8") as handle:
        return {normalise(row["alias"]): row["drug"].lower() for row in csv.DictReader(handle)}


@lru_cache(maxsize=1)
def listed_drugs() -> tuple[str, ...]:
    return tuple(sorted({role.drug for role in load_roles()}))


def resolve(text: str) -> dict:
    """Map one typed name to a listed drug, or say plainly that it is unlisted.

    Exact and synonym matches only. A drug the table does not list is reported
    as unlisted rather than guessed at, because a wrong match here would put a
    role on a drug that never had it.
    """
    query = normalise(text)
    if not query:
        return {"input": text, "status": "empty", "drug": None}
    exact = {normalise(drug): drug for drug in listed_drugs()}
    if query in exact:
        return {"input": text, "status": "listed", "drug": exact[query]}
    synonyms = load_synonyms()
    if query in synonyms:
        return {"input": text, "status": "synonym", "drug": synonyms[query], "matched": query}
    close = [drug for key, drug in exact.items() if query in key or key in query]
    close += [drug for key, drug in synonyms.items() if query in key and drug not in close]
    return {
        "input": text,
        "status": "unlisted",
        "drug": None,
        "suggestions": sorted(dict.fromkeys(close))[:5],
    }


def resolve_all(names) -> dict:
    results = [resolve(name) for name in names]
    return {
        "drugs": [r["drug"] for r in results if r["drug"]],
        "resolved": results,
        "unlisted": [r["input"] for r in results if not r["drug"] and r["status"] != "empty"],
    }
