"""Static mechanistic model for cumulative CYP-mediated exposure changes.

Educational prototype. Not clinical decision support. See README.

Everything here is a pure function of a drug list and the FDA role table. No
network, no database, no global state, so the whole model is testable.

The point of the model
----------------------
Most interaction checkers evaluate drug pairs. A patient on seven drugs has 21
pairs, and three of those drugs can each partly inhibit the same enzyme.
Inhibition of one enzyme by several drugs does not live in any pair, so a
pairwise check understates it, and a checker that reports only moderate and
stronger interactions can clear every pair while the whole list reads moderate.

Where the numbers come from
---------------------------
The FDA defines its inhibitor, inducer and substrate classes by observed AUC
changes, so each class is an interval, not a point:

  strong inhibitor    AUC of a sensitive substrate x5 or more
  moderate inhibitor  x2 to x5
  weak inhibitor      x1.25 to x2
  strong inducer      AUC down 80% or more
  moderate inducer    down 50% to 80%
  weak inducer        down 20% to 50%
  sensitive substrate AUC x5 or more with a strong inhibitor
  moderately sensitive  x2 to x5 with a strong inhibitor

This module keeps those intervals and propagates them, so every prediction is a
range with a stated basis. Nothing is a tuned parameter, and no value was chosen
to make an example look better.

The algebra
-----------
For one enzyme, write A for its remaining fractional activity. With fraction
metabolised fm through that enzyme, the exposure ratio of the substrate is

    AUCR = 1 / (fm * A + (1 - fm))

A reversible inhibitor contributes r = [I] / Ki. From the class definition,
a single inhibitor acting alone on a sensitive substrate (fm = 1) gives
AUCR = 1 + r, so r is simply the fold increase minus one.

Several reversible inhibitors competing for the same enzyme add in r:

    A = 1 / (1 + sum(r))                       competitive, the primary model

If instead each inhibitor is assumed to act independently of the others, the
surviving activities multiply:

    A = prod(1 / (1 + r))                      independent, an upper bound

The competitive form is the standard combination for reversible inhibitors of
one enzyme and is what this model reports. The independent form is reported
beside it because it is what one gets by combining fractional inhibitions as
1 - prod(1 - theta), a shortcut that treats the inhibitors as acting on separate
pools of enzyme and therefore overstates the effect. Showing both makes the
assumption visible instead of hiding it inside one number.

Induction multiplies the enzyme's activity. Combined induction has no settled
rule; this model multiplies the factors and says so.

Reading one number off an interval
----------------------------------
Exposure ratios are multiplicative, so the central estimate is the geometric
mean of the interval, and that is what the severity band is taken from. This is
not a free choice: for a single inhibitor acting on a sensitive substrate, the
geometric mean lands inside that inhibitor's own FDA class every time (a weak
inhibitor reads 1.25 to 2, a moderate 2 to 5, a strong 5 or more). The model is
therefore calibrated against the definitions it is built from, and
tests/test_model.py checks exactly that for all three classes. The interval
itself is always reported beside the point estimate.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# Fold change bounds straight from the FDA class definitions. The open-ended
# strong classes are capped so the arithmetic stays finite; the cap is reported
# with every result that depends on it.
STRONG_CAP = 20.0

INHIBITOR_FOLD = {
    "strong": (5.0, STRONG_CAP),
    "moderate": (2.0, 5.0),
    "weak": (1.25, 2.0),
}
INDUCER_FOLD = {  # factor by which the inducer multiplies enzyme activity
    "strong": (5.0, STRONG_CAP),
    "moderate": (2.0, 5.0),
    "weak": (1.25, 2.0),
}
SUBSTRATE_AUCR = {  # AUC fold increase that defines the class, with a strong inhibitor
    "sensitive": (5.0, None),
    "moderate": (2.0, 5.0),
}

# Severity bands reuse the FDA inhibitor thresholds, so "high" here means the
# exposure change a single strong inhibitor is defined to cause.
SEVERITY_BANDS = ((5.0, "high"), (2.0, "moderate"), (1.25, "low"))
SEVERITY_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3}


def severity(aucr: float) -> str:
    """Band an exposure ratio using the FDA inhibitor thresholds."""
    for threshold, label in SEVERITY_BANDS:
        if aucr >= threshold:
            return label
    return "none"


def inhibitor_r(strength: str) -> tuple[float, float]:
    """Interval for [I]/Ki implied by an inhibitor class: r = fold change - 1."""
    low, high = INHIBITOR_FOLD[strength]
    return low - 1.0, high - 1.0


def substrate_fm(strength: str) -> tuple[float, float]:
    """Interval for the fraction metabolised, derived from the substrate class.

    A substrate's class is defined by its AUC ratio with *a* strong inhibitor,
    and a strong inhibitor is anything from 5-fold upward. Inverting
    AUCR = 1 / (1 - fm * theta) over both ranges gives the fm interval that is
    consistent with the class. Nothing here is fitted or assumed beyond the
    cap on the open-ended strong class.
    """
    theta_min = 1.0 - 1.0 / INHIBITOR_FOLD["strong"][0]
    theta_max = 1.0 - 1.0 / INHIBITOR_FOLD["strong"][1]
    low, high = SUBSTRATE_AUCR[strength]
    fm_low = (1.0 - 1.0 / low) / theta_max
    fm_high = 1.0 if high is None else min(1.0, (1.0 - 1.0 / high) / theta_min)
    return min(fm_low, 1.0), min(fm_high, 1.0)


@dataclass(frozen=True)
class Role:
    """One drug's published role on one enzyme."""
    drug: str
    enzyme: str
    role: str       # inhibitor | inducer | substrate
    strength: str   # strong | moderate | weak | sensitive


@dataclass(frozen=True)
class Contributor:
    drug: str
    role: str
    strength: str


@dataclass(frozen=True)
class Finding:
    """A predicted exposure change for one substrate through one enzyme."""
    enzyme: str
    substrate: str
    substrate_class: str
    aucr_low: float
    aucr_point: float                  # geometric mean of the interval
    aucr_high: float
    aucr_independent_high: float
    severity: str                      # banded from aucr_point
    severity_low: str                  # banded from aucr_low
    severity_high: str                 # banded from aucr_high
    inhibitors: tuple[Contributor, ...]
    inducers: tuple[Contributor, ...]
    fm_low: float
    fm_high: float
    capped: bool                       # a strong class hit the finite cap
    assumptions: tuple[str, ...] = field(default=())

    @property
    def contributors(self) -> tuple[str, ...]:
        return tuple(c.drug for c in self.inhibitors + self.inducers)


def _aucr(fm: float, activity: float) -> float:
    denominator = fm * activity + (1.0 - fm)
    return float("inf") if denominator <= 0 else 1.0 / denominator


def _activity_competitive(rs: Sequence[float], inductions: Sequence[float]) -> float:
    induced = 1.0
    for factor in inductions:
        induced *= factor
    return induced / (1.0 + sum(rs))


def _activity_independent(rs: Sequence[float], inductions: Sequence[float]) -> float:
    induced = 1.0
    for factor in inductions:
        induced *= factor
    surviving = 1.0
    for r in rs:
        surviving *= 1.0 / (1.0 + r)
    return induced * surviving


def predict_aucr(
    inhibitor_strengths: Sequence[str],
    inducer_strengths: Sequence[str],
    substrate_strength: str,
) -> dict:
    """Exposure ratio interval for one substrate through one enzyme.

    Every input is an interval, so the result is evaluated at each corner of the
    input box and the extremes are kept. The corners are exact because AUCR is
    monotone in each input over the ranges used.
    """
    fm_bounds = substrate_fm(substrate_strength)
    r_bounds = [inhibitor_r(s) for s in inhibitor_strengths]
    m_bounds = [INDUCER_FOLD[s] for s in inducer_strengths]

    competitive, independent = [], []
    for fm in fm_bounds:
        for rs in itertools.product(*r_bounds) if r_bounds else [()]:
            for ms in itertools.product(*m_bounds) if m_bounds else [()]:
                competitive.append(_aucr(fm, _activity_competitive(rs, ms)))
                independent.append(_aucr(fm, _activity_independent(rs, ms)))
    capped = any(s == "strong" for s in list(inhibitor_strengths) + list(inducer_strengths))
    low, high = min(competitive), max(competitive)
    return {
        "low": low,
        "point": math.sqrt(low * high) if math.isfinite(high) else float("inf"),
        "high": high,
        "independent_high": max(independent),
        "fm_low": fm_bounds[0],
        "fm_high": fm_bounds[1],
        "capped": capped,
    }


def _index(roles: Iterable[Role]) -> dict:
    table: dict = {}
    for role in roles:
        table.setdefault(role.drug.lower(), []).append(role)
    return table


def analyze(drugs: Sequence[str], roles: Iterable[Role]) -> list[Finding]:
    """All predicted exposure changes for a drug list, strongest first.

    A finding needs a substrate with a published class and at least one
    inhibitor or inducer of the same enzyme. A drug never interacts with itself.
    """
    table = _index(roles)
    present = [d.lower() for d in dict.fromkeys(drug.lower() for drug in drugs)]
    findings: list[Finding] = []
    for substrate in present:
        for entry in table.get(substrate, []):
            if entry.role != "substrate" or entry.strength not in SUBSTRATE_AUCR:
                continue
            inhibitors, inducers = [], []
            for other in present:
                if other == substrate:
                    continue
                for role in table.get(other, []):
                    if role.enzyme != entry.enzyme:
                        continue
                    if role.role == "inhibitor":
                        inhibitors.append(Contributor(other, "inhibitor", role.strength))
                    elif role.role == "inducer":
                        inducers.append(Contributor(other, "inducer", role.strength))
            if not inhibitors and not inducers:
                continue
            prediction = predict_aucr(
                [c.strength for c in inhibitors], [c.strength for c in inducers], entry.strength
            )
            assumptions = []
            if len(inhibitors) > 1:
                assumptions.append(
                    "Several inhibitors of one enzyme are combined competitively; "
                    "the independent-action figure is shown as an upper bound."
                )
            if len(inducers) > 1:
                assumptions.append(
                    "Combined induction has no settled rule. Induction factors are multiplied."
                )
            if inhibitors and inducers:
                assumptions.append(
                    "Inhibition and induction of the same enzyme are netted on one activity "
                    "term. Their time courses differ, so the net figure is indicative only."
                )
            if prediction["capped"]:
                assumptions.append(
                    f"An open-ended strong class is capped at {STRONG_CAP:g}-fold for the upper bound."
                )
            findings.append(
                Finding(
                    enzyme=entry.enzyme,
                    substrate=substrate,
                    substrate_class=entry.strength,
                    aucr_low=prediction["low"],
                    aucr_point=prediction["point"],
                    aucr_high=prediction["high"],
                    aucr_independent_high=prediction["independent_high"],
                    severity=severity(prediction["point"]),
                    severity_low=severity(prediction["low"]),
                    severity_high=severity(prediction["high"]),
                    inhibitors=tuple(sorted(inhibitors, key=lambda c: c.drug)),
                    inducers=tuple(sorted(inducers, key=lambda c: c.drug)),
                    fm_low=prediction["fm_low"],
                    fm_high=prediction["fm_high"],
                    capped=prediction["capped"],
                    assumptions=tuple(assumptions),
                )
            )
    findings.sort(key=lambda f: (-f.aucr_point, f.substrate, f.enzyme))
    return findings


def pairwise_findings(drugs: Sequence[str], roles: Iterable[Role]) -> list[Finding]:
    """The same model restricted to one pair at a time: the baseline.

    This is deliberately not a third-party checker. Running the identical model
    over pairs isolates one variable, the number of drugs considered at once, so
    any difference is the cascade and not a difference in underlying data.
    """
    best: dict = {}
    for first, second in itertools.combinations(dict.fromkeys(d.lower() for d in drugs), 2):
        for finding in analyze([first, second], roles):
            key = (finding.enzyme, finding.substrate)
            if key not in best or finding.aucr_point > best[key].aucr_point:
                best[key] = finding
    return sorted(best.values(), key=lambda f: (-f.aucr_point, f.substrate, f.enzyme))


def cascades(drugs: Sequence[str], roles: Iterable[Role]) -> list[dict]:
    """Findings the pairwise view understates, which is the whole claim.

    For each (enzyme, substrate) the full model is compared with the best any
    single pair achieves. A cascade is reported only when the severity band
    actually moves, so a larger number alone is not called a finding.
    """
    full = {(f.enzyme, f.substrate): f for f in analyze(drugs, roles)}
    pairs = {(f.enzyme, f.substrate): f for f in pairwise_findings(drugs, roles)}
    out = []
    for key, finding in full.items():
        pair = pairs.get(key)
        pair_severity = pair.severity if pair else "none"
        if SEVERITY_ORDER[finding.severity] > SEVERITY_ORDER[pair_severity]:
            out.append(
                {
                    "finding": finding,
                    "pairwise_best": pair,
                    "pairwise_severity": pair_severity,
                    "pairwise_aucr_point": pair.aucr_point if pair else 1.0,
                    "severity_steps": SEVERITY_ORDER[finding.severity] - SEVERITY_ORDER[pair_severity],
                }
            )
    out.sort(key=lambda row: (-row["severity_steps"], -row["finding"].aucr_point))
    return out
