"""Tests for the medication-list reader (core/reader.py, data/reader_model.json).

The reader's job has two halves and both are tested: read what people actually
write, and never turn a real word, a look-alike or an unknown name into a drug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reader import FEATURES, key, load_model, read_list  # noqa: E402
from core.roles import listed_drugs  # noqa: E402

MODEL = load_model()


def drugs(text: str) -> list:
    return read_list(text, MODEL)["drugs"]


def verdict(text: str) -> dict:
    return MODEL.decide(key(text))


# --- reading what people write ---------------------------------------------------

@pytest.mark.parametrize("typed, drug", [
    ("Zocor", "simvastatin"), ("Biaxin", "clarithromycin"), ("Cardizem CD", "diltiazem"),
    ("Toprol XL", "metoprolol"), ("Effexor XR", "r-venlafaxine"), ("Paxlovid", "ritonavir"),
    ("Vytorin", "simvastatin"), ("Coumadin", "warfarin"),
])
def test_brand_names_read_as_their_table_drug(typed, drug):
    assert drugs(typed) == [drug]


@pytest.mark.parametrize("typed, drug", [
    ("simvistatin", "simvastatin"), ("amiodorone", "amiodarone"), ("clarithromycine", "clarithromycin"),
    ("metorpolol", "metoprolol"), ("propanolol", "propranolol"), ("phenobarbitol", "phenobarbital"),
])
def test_misspellings_are_read(typed, drug):
    assert drugs(typed) == [drug]


def test_pharmacy_printout_lines_lose_their_doses_and_forms():
    text = "SIMVASTATIN 40MG TAB\nMETOPROLOL SUCC ER 50MG TAB\nWARFARIN SODIUM 5MG TAB"
    assert drugs(text) == ["simvastatin", "metoprolol", "warfarin"]


def test_a_narrative_is_read_and_its_ordinary_words_are_not():
    text = "my dad takes zocor at night and just got biaxin for a sinus infection, he also smokes"
    assert drugs(text) == ["simvastatin", "clarithromycin", "tobacco (smoking)"]


def test_the_messy_example_on_the_page_reads_to_the_demo_regimen():
    text = ("Zocor 40mg at bedtime\nCardizem CD 180 daily\nerythromycin 500 four times a day\n"
            "grapefruit juice with breakfast\nToprol XL 50\nPrilosec 20\nCoumadin 5mg Mon/Wed/Fri\nlisinopril 10")
    result = read_list(text, MODEL)
    assert result["drugs"] == ["simvastatin", "diltiazem", "erythromycin", "grapefruit juice",
                               "metoprolol", "omeprazole", "warfarin"]
    assert result["not_covered"] == ["lisinopril"]


def test_every_reading_is_a_drug_the_analysis_can_score():
    text = "Zocor, Kaletra, St Johns Wort, birth control pill, Nexium, Effexor"
    assert set(drugs(text)) <= set(listed_drugs())


# --- refusing to guess -------------------------------------------------------------

@pytest.mark.parametrize("typed, forbidden", [
    ("Celexa", "celecoxib"), ("Lamictal", "terbinafine"), ("trazodone", "tramadol"),
    ("Sinequan", "quetiapine"), ("clonidine", "clozapine"), ("Zyrtec", "simvastatin"),
])
def test_look_alike_names_are_not_read_as_their_partner(typed, forbidden):
    assert forbidden not in drugs(typed)


@pytest.mark.parametrize("word", ["program", "pipeline", "lipid", "vision", "detroit", "stoker",
                                  "smoky", "consensus", "trailer", "melanin", "pegasus"])
def test_ordinary_english_is_never_read_as_a_medicine(word):
    assert verdict(word)["status"] == "none"


def test_habit_words_are_matched_only_exactly():
    assert drugs("smoker") == ["tobacco (smoking)"]
    assert drugs("grapefruit") == ["grapefruit juice"]
    assert drugs("smoked salmon") == []


def test_a_medicine_outside_the_table_is_named_not_scored():
    result = read_list("lisinopril 10, Eliquis 5 bid", MODEL)
    assert result["drugs"] == []
    assert result["not_covered"] == ["lisinopril", "apixaban"]


def test_empty_text_reads_nothing():
    assert read_list("", MODEL)["items"] == []
    assert read_list("   \n , ;", MODEL)["items"] == []


# --- combinations -------------------------------------------------------------------

def test_a_table_combination_is_one_drug_not_its_parts():
    assert drugs("Kaletra") == ["lopinavir and ritonavir"]
    assert drugs("lopinavir and ritonavir") == ["lopinavir and ritonavir"]
    assert drugs("Kaletra (lopinavir/ritonavir) 400/100") == ["lopinavir and ritonavir"]


def test_separate_items_stay_separate():
    assert drugs("Kaletra 2 tabs bid; Norvir 100 daily") == ["lopinavir and ritonavir", "ritonavir"]


# --- the model file -------------------------------------------------------------------

def test_model_file_is_complete_and_consistent():
    meta = MODEL.meta
    assert meta["features"] == list(FEATURES)
    assert len(MODEL.mean) == len(MODEL.scale) == len(FEATURES)
    assert len(MODEL.layers[0]["W"][0]) == len(FEATURES)
    assert set(MODEL.thresholds) >= {"accept", "accept_other", "margin", "ask", "ask_english"}
    assert MODEL.lexicon, "the English word list must ship with the model"


def test_probabilities_are_probabilities():
    for text in ("simvastatin", "simvistatin", "zzzzqx", "celexa"):
        for p, _ in MODEL.score(key(text)):
            assert 0.0 <= p <= 1.0


def test_exact_table_names_are_always_read():
    missed = [d for d in listed_drugs() if d not in drugs(d)]
    assert missed == []
