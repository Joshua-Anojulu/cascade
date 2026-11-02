"""Tests for the exposure model, the role table, and the pairwise baseline.

The first test is the important one: it checks the model against the FDA class
definitions it is built from. A single inhibitor of a given class, acting on a
sensitive substrate, must read back inside that same class. If that ever breaks,
every number the project reports is wrong.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model import (  # noqa: E402
    INHIBITOR_FOLD,
    SEVERITY_ORDER,
    STRONG_CAP,
    analyze,
    cascades,
    inhibitor_r,
    pairwise_findings,
    predict_aucr,
    severity,
    substrate_fm,
)
from core.roles import load_roles, resolve, resolve_all  # noqa: E402

ROLES = load_roles()


# --- calibration against the definitions ------------------------------------

@pytest.mark.parametrize("strength", ["weak", "moderate", "strong"])
def test_single_inhibitor_reads_back_in_its_own_class(strength):
    """The central estimate for one inhibitor lands in that inhibitor's FDA band."""
    result = predict_aucr([strength], [], "sensitive")
    low, high = INHIBITOR_FOLD[strength]
    assert low <= result["point"] < (high if strength != "strong" else math.inf)
    assert severity(result["point"]) == {"weak": "low", "moderate": "moderate", "strong": "high"}[strength]


def test_interval_contains_the_defining_case():
    """A sensitive substrate with a boundary strong inhibitor is exactly 5-fold."""
    result = predict_aucr(["strong"], [], "sensitive")
    assert result["low"] <= 5.0 <= result["high"]
    assert result["high"] == pytest.approx(STRONG_CAP)


def test_r_is_fold_change_minus_one():
    for strength, (low, high) in INHIBITOR_FOLD.items():
        assert inhibitor_r(strength) == (low - 1.0, high - 1.0)


def test_fm_intervals_are_derived_not_assumed():
    sensitive, moderate = substrate_fm("sensitive"), substrate_fm("moderate")
    assert sensitive[0] == pytest.approx(0.8 / 0.95, rel=1e-9)
    assert moderate[0] == pytest.approx(0.5 / 0.95, rel=1e-9)
    assert sensitive[1] == moderate[1] == 1.0
    assert sensitive[0] > moderate[0], "a sensitive substrate must carry more flux"


# --- the cascade itself -----------------------------------------------------

def test_inhibitors_combine_and_the_band_moves():
    one = predict_aucr(["moderate"], [], "sensitive")
    three = predict_aucr(["moderate"] * 3, [], "sensitive")
    assert severity(one["point"]) == "moderate"
    assert severity(three["point"]) == "high"
    assert three["low"] > one["low"] and three["point"] > one["point"]


def test_competitive_combination_is_below_independent_action():
    """The multiplicative shortcut overstates; the model must not use it."""
    result = predict_aucr(["moderate"] * 3, [], "sensitive")
    assert result["high"] < result["independent_high"]
    single = predict_aucr(["moderate"], [], "sensitive")
    assert single["high"] == single["independent_high"], "one inhibitor, no combination rule"


def test_induction_opposes_inhibition():
    inhibited = predict_aucr(["moderate"], [], "sensitive")
    netted = predict_aucr(["moderate"], ["strong"], "sensitive")
    assert netted["point"] < inhibited["point"]
    assert predict_aucr([], ["strong"], "sensitive")["point"] < 1.0


def test_worked_demo_case_is_pinned():
    """Three moderate CYP3A inhibitors move simvastatin from moderate to high."""
    drugs = ["simvastatin", "diltiazem", "erythromycin", "grapefruit juice",
             "metoprolol", "omeprazole", "warfarin"]
    rows = cascades(drugs, ROLES)
    assert len(rows) == 1
    finding = rows[0]["finding"]
    assert (finding.substrate, finding.enzyme) == ("simvastatin", "CYP3A")
    assert finding.severity == "high"
    assert rows[0]["pairwise_severity"] == "moderate"
    assert finding.aucr_point == pytest.approx(5.94, abs=0.05)
    assert finding.aucr_low == pytest.approx(2.71, abs=0.05)
    assert finding.aucr_high == pytest.approx(13.0, abs=0.05)
    assert {c.drug for c in finding.inhibitors} == {"diltiazem", "erythromycin", "grapefruit juice"}


def test_removing_one_inhibitor_removes_the_cascade():
    """The finding is caused by the third drug, not by the pair."""
    drugs = ["simvastatin", "diltiazem", "erythromycin", "metoprolol", "omeprazole", "warfarin"]
    assert cascades(drugs, ROLES) == []
    assert analyze(drugs, ROLES)[0].severity == "moderate"


def test_pairwise_baseline_never_exceeds_the_full_model():
    drugs = ["simvastatin", "diltiazem", "erythromycin", "grapefruit juice", "warfarin",
             "fluconazole", "omeprazole"]
    full = {(f.enzyme, f.substrate): f for f in analyze(drugs, ROLES)}
    for pair in pairwise_findings(drugs, ROLES):
        assert pair.aucr_point <= full[(pair.enzyme, pair.substrate)].aucr_point + 1e-9


def test_a_drug_does_not_interact_with_itself():
    assert analyze(["fluvoxamine"], ROLES) == []
    assert analyze(["simvastatin", "simvastatin"], ROLES) == []


def test_substrate_with_no_published_class_gets_no_number():
    """Only substrates the FDA table classes are predicted; nothing is defaulted."""
    unclassed = [r for r in ROLES if r.role == "substrate" and r.strength not in ("sensitive", "moderate")]
    assert unclassed == [], "every substrate row should carry a class"
    assert analyze(["diltiazem", "erythromycin"], ROLES) == [], "inhibitors alone predict nothing"


def test_severity_bands_follow_the_fda_thresholds():
    assert severity(1.1) == "none"
    assert severity(1.25) == "low"
    assert severity(2.0) == "moderate"
    assert severity(5.0) == "high"
    assert SEVERITY_ORDER["high"] > SEVERITY_ORDER["moderate"] > SEVERITY_ORDER["low"]


# --- the role table and name resolution -------------------------------------

def test_role_table_is_well_formed():
    assert len({r.drug for r in ROLES}) > 150
    for role in ROLES:
        assert role.role in {"inhibitor", "inducer", "substrate"}
        assert role.enzyme.startswith("CYP")
        assert role.drug == role.drug.lower().strip()
        if role.role == "substrate":
            assert role.strength in {"sensitive", "moderate"}
        else:
            assert role.strength in {"strong", "moderate", "weak"}
    assert len({(r.drug, r.enzyme, r.role) for r in ROLES}) == len(ROLES), "no duplicate rows"


def test_known_rows_match_the_fda_table():
    index = {(r.drug, r.enzyme, r.role): r.strength for r in ROLES}
    assert index[("clarithromycin", "CYP3A", "inhibitor")] == "strong"
    assert index[("diltiazem", "CYP3A", "inhibitor")] == "moderate"
    assert index[("simvastatin", "CYP3A", "substrate")] == "sensitive"
    assert index[("rifampin", "CYP3A", "inducer")] == "strong"
    assert index[("paroxetine", "CYP2D6", "inhibitor")] == "strong"
    assert index[("warfarin", "CYP2C9", "substrate")] == "moderate"


def test_resolution_is_exact_and_refuses_to_guess():
    assert resolve("Simvastatin")["drug"] == "simvastatin"
    assert resolve("  DILTIAZEM ")["drug"] == "diltiazem"
    unlisted = resolve("ibuprofen")
    assert unlisted["status"] == "unlisted" and unlisted["drug"] is None
    assert resolve("")["status"] == "empty"
    batch = resolve_all(["simvastatin", "ibuprofen", "diltiazem"])
    assert batch["drugs"] == ["simvastatin", "diltiazem"]
    assert batch["unlisted"] == ["ibuprofen"]


def test_synonyms_point_at_real_drugs():
    from core.roles import load_synonyms
    listed = {r.drug for r in ROLES}
    for alias, drug in load_synonyms().items():
        assert drug in listed, f"synonym {alias} points at unlisted {drug}"
        assert resolve(alias)["drug"] == drug
