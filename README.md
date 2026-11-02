# Cascade

Cascade is an educational model of the drug interaction a pairwise check cannot see: several drugs each partly inhibiting the same metabolic enzyme. It reads a medication list the way people actually write it, with a small trained model that runs on the same server and needs no API key.

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

## Reading the list people actually have

Nobody types "simvastatin". People paste a pharmacy printout or write "Zocor 40 at night, simvistatin, the Biaxin from urgent care". An exact-name lookup reads almost none of that. Given 80 realistic lists, Cascade's original resolver (exact names plus 84 brand aliases) found 15% of the table drugs in them.

So Cascade has a reader, `core/reader.py`. It runs on the same server as the rest; the text goes nowhere else.

1. **Windows.** The text is split at lines, commas and "and", then into windows of one to four words. Doses, forms and release codes ("40MG", "TAB", "XL") are stripped.
2. **Candidates.** Each window retrieves its eight closest names from a dictionary of 1,967 by character n-gram TF-IDF cosine, so "simvistatin" still lands near "simvastatin". The dictionary holds the 205 FDA table drugs and 387 brand names that contain them (RxNorm, via the RxNav API, 2 November 2026). It also holds the 300 most prescribed US drugs (ClinCalc, 2024) with their brands and the other ingredients in all those products, so a medicine outside the table is recognised as outside it, not mistaken for one inside it.
3. **The network.** A neural network with one hidden layer of 32 units scores every window-candidate pair: is this the drug the person meant? It sees 16 features: string similarity of several kinds, how much better this candidate fits than the best candidate that means a different drug, and how much the window looks like ordinary English (from a separate character model). The margin against a rival is how it learns to stop at look-alikes such as Celexa and Celebrex instead of guessing.
4. **Three rules around it**, each for a failure no string score can fix:
   - **Common English words are never fuzzy-matched.** The 47,882 commonest words (wordfreq) are matched only if they are a drug name exactly. "pipeline" is one letter from piperine and "program" one from Prograf. A word list tells a real word from a typo; similarity cannot.
   - **Habit words match only exactly.** "smoker", "grapefruit" and "birth control pill" are English words themselves.
   - **A combination the FDA lists as one row** (lopinavir and ritonavir) absorbs its parts when they are written together.
5. **Outcomes.** A confident reading becomes a drug. A known medicine outside the table is named and marked not covered. Two plausible drugs become a question. Anything weaker stays unread and is left unhighlighted on the page, never silently dropped.

**Training** (`train_reader.py`) builds 29,757 training rows:
- every dictionary name, exactly and misspelled the ways people misspell: dropped, doubled or swapped letters, neighbouring keys, wrong vowels, sound-alike spellings ("ph" for "f"), words cut short;
- 15,000 English words and phrases, 2,440 of them chosen because they sit near a drug name;
- 385 real medicine names hidden from the network's dictionary, so it learns that a drug-like name near a known one is not that one.

Thresholds are set on a separate validation set. They maximise correct readings while keeping wrong readings under 0.5% of those accepted, unknown names read as a table drug under 1%, and English under similar limits. Every random choice is seeded, and retraining reproduces the model file byte for byte.

### How well it reads

| Test | Result |
| --- | --- |
| **Blind lists**: 80 lists with answer keys, written by a separate agent that never saw the reader; run once, after the model was frozen | **67 of 80 read exactly. Table drugs: precision 99.1%, recall 95.2%.** The original resolver, given the same splitting: 6 of 80, recall 15.2% |
| Blind lists by style | narrative 12/12, clinician 12/13, patient 21/26, caregiver 16/21, pharmacy printout 6/8 |
| Look-alike pairs from ISMP's List of Confused Drug Names: 114 names typed exactly, 342 misspelled | Never read as its listed partner. 5 misspellings were read as a third medicine, none of them a table drug |
| Unseen misspellings: every table name, new seed, more two- and three-letter errors than training | 87.2% read, 0.13% read as the wrong drug, 12.6% left unread |
| Validation, the training mix | 93.7% read correctly, 0.05% wrong. A linear model on the same features: 88.9% |

The ISMP test was used during development. An earlier build read names it did not know as their nearest neighbour (Serzone as Lorzone), and holding real names out of training came from that finding. So that test is not blind; the 80 lists are.

**What the blind lists caught** (model v1.0, frozen before the run):

- **Two wrong drugs, both look-alike traps.** QUININE was read as quinidine, because quinine is not in the dictionary. "Zantac 360" was read as ranitidine: Zantac 360 has been famotidine since 2021, but RxNorm still files the Zantac name under ranitidine.
- **Eleven missed drugs:**
  - an abbreviation and a supplement brand it never learned (TMP-SMX, BioPerine);
  - two first-letter misspellings (Zarelto, Symbalta), because the typo model assumes the first letter is right;
  - three misspelled habit words, matched only exactly by design (grapefuit juice, birth controll pill, St Johns wart);
  - "Singular" for Singulair, which is a real word;
  - Multac for Multaq;
  - an oral contraceptive written as its two ingredients;
  - "fenytoin", where it asked instead of reading.
- **All 5 mentions the answer key marks as genuinely ambiguous got a reading instead of a question** ("buspropion": bupropion or buspirone?). The question path fires too rarely.
- **Medicines outside the table:** 137 of 198 were named as the key names them. About 7 readings named a different medicine (Trintellix as Trilipix, valganciclovir as valacyclovir, vitamin D3 as vitamin B6). Those change a "not in the table" note, never the analysis.

Reading the errors exposed two bugs in the rules, not the network. A window could end on a schedule word ("Zyvox 600 mg BID" became Bidil), and a one-letter token slipped past the English check ("fentanyl 50 mcg/h" became Centany). Both are fixed in v1.1. Re-running the same lists gives the same table-drug results (67 of 80, 99.1% and 95.2%) and three fewer wrong names outside the table. That second run is not blind.

## Where the numbers come from

Nothing in the exposure model is a tuned parameter; the list reader above is the only trained part, and it never touches a number in a finding. The FDA defines its interaction classes by observed exposure changes, and every class is a range:

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
pytest -q                                   # 62 tests: calibration, the worked case, the reader
python evaluate.py                          # the headline numbers, about 2 seconds
python evaluate_reader.py                   # the reader's four tests, about 10 seconds
uvicorn api.main:app --port 8000            # then open http://127.0.0.1:8000
```

The interface takes a pasted list, shows what it read and how, and puts the pairwise view and the full-list view side by side with the same drugs in both. "Use a messy example" pastes the worked case as a person would write it.

To retrain the reader from its sources (about two and a half minutes, identical output):

```bash
pip install -r requirements-train.txt
python train_reader.py
```

## Layout

```
data/cyp_roles.csv            285 published roles for 205 drugs, from the FDA table
data/synonyms.csv             brand names that map onto those drugs
data/rxnorm_extract.json      branded drug names from RxNorm, as retrieved, for the reader
data/hand_aliases.csv         49 spellings RxNorm does not give, each with its reason
data/reader_model.json        the trained reader: dictionary, n-gram weights, network, thresholds
data/reader_dictionary.csv    the same dictionary, readable
data/english_words.txt.gz     the common-English word list the reader never fuzzy-matches
data/blind_med_lists.json     the 80 blind test lists and their answer keys
core/model.py                 the model: intervals, combination rules, cascade detection. No I/O
core/roles.py                 loading the table and resolving typed names
core/reader.py                reading a pasted list; standard library only
core/typos.py                 realistic misspellings, for training and testing
api/main.py                   four endpoints
web/index.html                one page, no build step
evaluate.py                   the exhaustive enumeration behind the headline numbers
evaluate_reader.py            the reader's tests; standard library only
train_reader.py               builds the reader from its sources
Cascade_one_page.pdf          the one-page summary
tests/                        calibration, the worked case, the table, the reader
```

`core/model.py` performs no network or file access. It takes a drug list and a role table and returns findings, which is what makes the calibration test possible.

## What it does not do

- **It predicts exposure, not harm.** A 5.9x exposure change is not a clinical outcome. Whether that matters depends on the drug's therapeutic window, the dose, the patient's renal and hepatic function, their genotype, and timing, none of which this model has.
- **One enzyme at a time, reversible inhibition only.** No time-dependent inhibition, no transporters, no pharmacodynamic stacking (QT, serotonergic, bleeding, sedation). Detector B in the original plan is deliberately absent: there is no source for it with the authority of the FDA table, and inventing one would undermine everything else here.
- **Only the FDA index list.** 205 drugs, chosen by the FDA as clinical index substrates and perpetrators. Common drugs are missing; the FDA says plainly its table "are a guide and not considered a comprehensive list". An unlisted drug is reported as unlisted, never silently dropped or guessed at.
- **Intervals are wide, and that is honest.** Class-level data cannot produce a precise number. The point estimate is the geometric mean of the interval and the interval is always shown beside it.
- **Combined induction has no settled rule.** Induction factors are multiplied and the assumption is printed on every affected finding.
- **The reader can be wrong, so every reading is shown.** On the blind lists it added two wrong drugs in 80 lists and missed eleven. It is weakest on abbreviations, first-letter misspellings and names outside its dictionary; it asks a question less often than it should. Check the highlighted readings before trusting the analysis.
- **The reader's look-alike and English safeguards are rules plus a network, not understanding.** It does not know what a sentence means; "stopped Zocor last month" still reads Zocor.
- **It is not validated against clinical outcomes.** The evaluation measures the model against the pairwise version of itself over published data. It does not show that any predicted cascade caused harm in a real patient.

## AI disclosure

- **Joshua Anojulu**: project lead. Chose the problem and the event, and directs the build.
- **Claude (Anthropic)**: extracted the FDA role table into `data/cyp_roles.csv`, derived the fm intervals from the class definitions, wrote `core/`, `api/`, `web/`, `evaluate.py` and the tests, and wrote this README. The decision to use competitive rather than multiplicative combination, and to report intervals rather than point estimates, came from working through the FDA definitions rather than from the original project plan, which specified the multiplicative form and a default fm of 0.5.

- **Claude (Anthropic)** also built the list reader: gathered the RxNorm and ClinCalc names (through Joshua's browser, from the public RxNav API and the ClinCalc table), wrote `core/reader.py`, `core/typos.py`, `train_reader.py`, `evaluate_reader.py` and their tests, and trained the network. The English word list, the exact-only habit words and the held-out-name training each came from a measured failure of the version before, described above.
- **A separate Claude agent** wrote the 80 blind test lists and their answer keys from the FDA drug names alone, without access to the reader or this repository.

Earlier project specification and architecture planning: Claude. No part of the model was copied from an existing interaction checker.

## Credits

Drug roles and class definitions: U.S. Food and Drug Administration, [Drug Development and Drug Interactions: Table of Substrates, Inhibitors and Inducers](https://www.fda.gov/drugs/drug-interactions-labeling/healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-transporter-systems), Table 1 and Table 2, retrieved 31 October 2026. The static mechanistic approach follows the modelling described in the FDA's in vitro drug interaction guidance; the implementation here is original.

Brand and ingredient names: RxNorm, U.S. National Library of Medicine, through the RxNav REST API, version of 2 November 2026. Most prescribed drugs: ClinCalc DrugStats, The Top 300 of 2024 (from the Medical Expenditure Panel Survey). Look-alike name pairs for testing: Institute for Safe Medication Practices, ISMP List of Confused Drug Names, 2024. English word frequencies: wordfreq by Robyn Speer; `data/english_words.txt.gz` is derived from wordfreq's data and is shared under its CC BY-SA 4.0 licence.
