# Cascade

Cascade is an educational model of the drug interaction a pairwise check cannot see: several drugs each partly inhibiting the same metabolic enzyme.

**Educational prototype. Not clinical decision support.** Every prediction describes published drug classes, not a patient. Nothing here should be used to make or change treatment.

## The problem, with a number

Interaction checkers compare drugs two at a time. A patient on seven drugs generates 21 pairs. Each pair can be cleared while three of those drugs each partly inhibit the same enzyme, because the combined effect does not exist in any pair.

Taking the FDA's own index table of 205 drugs and enumerating every combination in it:

| Combination | Count | Understated by a pairwise view |
| --- | --- | --- |
| 2 inhibitors of one enzyme + 1 substrate | 84,302 | 8,057 (9.6%) |
| 3 inhibitors of one enzyme + 1 substrate | 1,341,070 | 78,753 (5.9%) |

"Understated" means the full model's severity band is higher than the band any single pair of those same drugs reaches. Of the three-inhibitor cases, 8,981 move from moderate to high: the combination lands in the exposure band the FDA uses to define a strong inhibitor while no pair exceeds moderate. A further 69,772 move from low to moderate, and those are invisible to any checker that does not surface weak-inhibitor pairs.

Drawing random 7-drug lists from the same table, 3.1% contain at least one understated finding (62 of 2,000, seed 20261108).

Reproduce all of it with `python evaluate.py`. It runs in about two seconds and enumerates the space exhaustively; nothing is sampled or hand-picked.

## The worked example

```
simvastatin, diltiazem, erythromycin, grapefruit juice, metoprolol, omeprazole, warfarin
```

Diltiazem, erythromycin and grapefruit juice are each a moderate CYP3A inhibitor. Simvastatin is a sensitive CYP3A substrate.

- Best single pair: 2.9x predicted exposure, **moderate**.
- All three together: 5.9x predicted exposure, **high**.

Drop any one of the three inhibitors and the finding disappears, which is the point: the effect belongs to the set, not to any pair. `tests/test_model.py` pins both results.

## Where the numbers come from

Nothing in this model is a tuned parameter. The FDA defines its interaction classes by observed exposure changes, and every class is a range:

| Class | Definition |
| --- | --- |
| Strong inhibitor | AUC of a sensitive substrate x5 or more |
| Moderate inhibitor | x2 to x5 |
| Weak inhibitor | x1.25 to x2 |
| Strong / moderate / weak inducer | AUC down 80%+ / 50 to 80% / 20 to 50% |
| Sensitive substrate | AUC x5 or more with a strong inhibitor |
| Moderately sensitive substrate | x2 to x5 with a strong inhibitor |

Those ranges are carried through the arithmetic, so every prediction is an interval. For one enzyme with remaining fractional activity A and fraction metabolised fm:

```
AUCR = 1 / (fm * A + (1 - fm))
```

A single reversible inhibitor contributes `r = [I]/Ki`, and from the class definition `r` is exactly the fold change minus one. Several reversible inhibitors of the same enzyme add in `r`:

```
A = 1 / (1 + sum(r))
```

The fraction metabolised is derived rather than assumed. Inverting the substrate class definition over the range of inhibitors that qualify as strong gives fm between 0.84 and 1.0 for a sensitive substrate, and 0.53 to 1.0 for a moderately sensitive one. There is no "default fm of 0.5" anywhere in this project; a substrate with no published class gets no number at all.

**The model is calibrated against the definitions it is built from.** For a single inhibitor acting on a sensitive substrate, the geometric mean of the predicted interval lands inside that inhibitor's own FDA class every time: a weak inhibitor reads 1.25 to 2, a moderate 2 to 5, a strong 5 or more. That is a property of the arithmetic, not a fit, and `test_single_inhibitor_reads_back_in_its_own_class` checks all three classes. It is also why the severity bands reuse the FDA thresholds: "high" means the exposure change a single strong inhibitor is defined to cause.

### Two corrections worth stating out loud

**The combination rule.** Fractional inhibitions are often combined as `1 - prod(1 - theta)`. That treats each inhibitor as acting on a separate pool of enzyme and overstates the result. For three moderate inhibitors it predicts up to 125x where competitive combination predicts 13x. Cascade uses the competitive form and reports the independent figure beside it as an upper bound, so the assumption is visible instead of buried.

**What the baseline actually is.** The pairwise comparison is not a third-party checker. It is this same model run over one pair at a time. That isolates a single variable, the number of drugs considered at once, so any difference is the cascade and not a difference in underlying data. Comparing against a commercial product would confound the two.

## Run it

Needs Python 3.11 or newer.

```bash
pip install -r requirements.txt
pytest -q                                   # 19 tests, including the calibration check
python evaluate.py                          # the headline numbers, about 2 seconds
uvicorn api.main:app --port 8000            # then open http://127.0.0.1:8000
```

The interface puts the pairwise view and the full-list view side by side with the same drugs in both. "Load the example regimen" sets up the worked case.

## Layout

```
data/cyp_roles.csv     285 published roles for 205 drugs, from the FDA table
data/synonyms.csv      brand names that map onto those drugs
core/model.py          the model: intervals, combination rules, cascade detection. No I/O
core/roles.py          loading the table and resolving typed names
api/main.py            three endpoints
web/index.html         one page, no build step
evaluate.py            the exhaustive enumeration behind the headline numbers
Cascade_one_page.pdf   the one-page summary
tests/test_model.py    calibration, the worked case, the table, name resolution
```

`core/model.py` performs no network or file access. It takes a drug list and a role table and returns findings, which is what makes the calibration test possible.

## What it does not do

- **It predicts exposure, not harm.** A 5.9x exposure change is not a clinical outcome. Whether that matters depends on the drug's therapeutic window, the dose, the patient's renal and hepatic function, their genotype, and timing, none of which this model has.
- **One enzyme at a time, reversible inhibition only.** No time-dependent inhibition, no transporters, no pharmacodynamic stacking (QT, serotonergic, bleeding, sedation). Detector B in the original plan is deliberately absent: there is no source for it with the authority of the FDA table, and inventing one would undermine everything else here.
- **Only the FDA index list.** 205 drugs, chosen by the FDA as clinical index substrates and perpetrators. Common drugs are missing; the FDA says plainly its table "are a guide and not considered a comprehensive list". An unlisted drug is reported as unlisted, never silently dropped or guessed at.
- **Intervals are wide, and that is honest.** Class-level data cannot produce a precise number. The point estimate is the geometric mean of the interval and the interval is always shown beside it.
- **Combined induction has no settled rule.** Induction factors are multiplied and the assumption is printed on every affected finding.
- **It is not validated against clinical outcomes.** The evaluation measures the model against the pairwise version of itself over published data. It does not show that any predicted cascade caused harm in a real patient.

## AI disclosure

- **Joshua Anojulu**: project lead. Chose the problem and the event, and directs the build.
- **Claude (Anthropic)**: extracted the FDA role table into `data/cyp_roles.csv`, derived the fm intervals from the class definitions, wrote `core/`, `api/`, `web/`, `evaluate.py` and the tests, and wrote this README. The decision to use competitive rather than multiplicative combination, and to report intervals rather than point estimates, came from working through the FDA definitions rather than from the original project plan, which specified the multiplicative form and a default fm of 0.5.

Earlier project specification and architecture planning: Claude. No part of the model was copied from an existing interaction checker.

## Credits

Drug roles and class definitions: U.S. Food and Drug Administration, [Drug Development and Drug Interactions: Table of Substrates, Inhibitors and Inducers](https://www.fda.gov/drugs/drug-interactions-labeling/healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-transporter-systems), Table 1 and Table 2, retrieved 31 October 2026. The static mechanistic approach follows the modelling described in the FDA's in vitro drug interaction guidance; the implementation here is original.
