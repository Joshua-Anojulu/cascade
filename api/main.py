"""Cascade API. Three endpoints, no database, no accounts.

Educational prototype. Not clinical decision support.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model import STRONG_CAP, analyze, cascades, pairwise_findings  # noqa: E402
from core.roles import listed_drugs, load_roles, resolve_all  # noqa: E402

WEB = Path(__file__).resolve().parent.parent / "web"
DISCLAIMER = (
    "Educational prototype, not clinical decision support. Predictions come from a static "
    "model built on the FDA's published interaction classes and apply to no individual "
    "patient. Do not use to make or change treatment decisions."
)

app = FastAPI(title="Cascade", description=DISCLAIMER, version="1.0.0")


class Request(BaseModel):
    drugs: list[str] = Field(default_factory=list, max_length=40)


def _contributor(c) -> dict:
    return {"drug": c.drug, "role": c.role, "strength": c.strength}


def _finding(f) -> dict:
    return {
        "enzyme": f.enzyme,
        "substrate": f.substrate,
        "substrate_class": f.substrate_class,
        "aucr": {"low": round(f.aucr_low, 2), "point": round(f.aucr_point, 2),
                 "high": round(f.aucr_high, 2),
                 "independent_high": round(f.aucr_independent_high, 2)},
        "severity": f.severity,
        "severity_range": [f.severity_low, f.severity_high],
        "fm": [round(f.fm_low, 3), round(f.fm_high, 3)],
        "inhibitors": [_contributor(c) for c in f.inhibitors],
        "inducers": [_contributor(c) for c in f.inducers],
        "capped": f.capped,
        "assumptions": list(f.assumptions),
    }


@app.get("/api/drugs")
def drugs() -> dict:
    """Every drug the FDA table lists, for the type-ahead."""
    return {"drugs": list(listed_drugs()), "count": len(listed_drugs())}


@app.post("/api/analyze")
def analyse(request: Request) -> dict:
    """Full model, pairwise baseline, and the findings the pairwise view understates."""
    if len(request.drugs) > 40:
        raise HTTPException(status_code=422, detail="At most 40 drugs.")
    resolved = resolve_all(request.drugs)
    roles = load_roles()
    names = resolved["drugs"]
    rows = cascades(names, roles)
    return {
        "resolved": resolved["resolved"],
        "unlisted": resolved["unlisted"],
        "findings": [_finding(f) for f in analyze(names, roles)],
        "pairwise": [_finding(f) for f in pairwise_findings(names, roles)],
        "cascades": [
            {
                "finding": _finding(row["finding"]),
                "pairwise_severity": row["pairwise_severity"],
                "pairwise_aucr": round(row["pairwise_aucr_point"], 2),
                "severity_steps": row["severity_steps"],
            }
            for row in rows
        ],
        "strong_cap": STRONG_CAP,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/roles/{drug}")
def roles_for(drug: str) -> dict:
    """What the FDA table publishes about one drug, so any finding can be traced."""
    match = resolve_all([drug])
    if not match["drugs"]:
        raise HTTPException(status_code=404, detail=f"{drug} is not in the FDA table.")
    name = match["drugs"][0]
    return {
        "drug": name,
        "roles": [
            {"enzyme": r.enzyme, "role": r.role, "strength": r.strength}
            for r in load_roles() if r.drug == name
        ],
        "source": "FDA, Drug Development and Drug Interactions, Table 1",
        "source_url": ("https://www.fda.gov/drugs/drug-interactions-labeling/"
                       "healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-"
                       "and-transporter-systems"),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
