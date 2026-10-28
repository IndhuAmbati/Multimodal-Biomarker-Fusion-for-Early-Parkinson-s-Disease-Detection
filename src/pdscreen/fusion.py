"""Fusion of the voice model (Model 1) and the gait-video model (Model 2) into one screening result.

Principles
- **Each model is trusted in proportion to the skill that can be credited to Parkinson's.** The weight of a model is
  ``max(AUC - 0.5, 0.01)`` (the AUC-0.5 rule from the project's implementation guide), computed from the model's
  person-wise validation AUC. For the gait-video model the AUC is *credited* only for what it achieves beyond a
  model that sees no gait at all (video properties, camera framing, task), because the audit showed those models reach
  the same score; the credited AUC is then 0.5 and the weight falls to the 0.01 floor.
- **Probabilities are put on one scale first.** Each model has its own decision threshold (voice 0.62, gait 0.5), so each
  probability is shifted in log-odds until its own threshold sits at 0.5 before the weighted average is taken.
- **A model that cannot be trusted for this input is left out.** For a particular video the gait result is excluded if a
  camera-framing-only model reaches the same verdict, and a model that failed or has no input is simply absent.
- **A single unreliable model never produces a verdict.** With only the gait-video model the result is "Inconclusive".
- **Confidence reflects agreement.** Informative models that all agree raise confidence one level; informative models
  that split lower it; a single model is capped at "Moderate".

The combination has not been validated on people who have both a voice recording and a gait video, because the two
models were trained on different people. The result always says so.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

FLOOR = 0.01          # weight of a model whose credited AUC is at or below chance
INFORMATIVE = 0.02    # a model must be weighted above this to carry a verdict on its own
LEVELS = ["Low", "Moderate", "High"]


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def align(probability: float, threshold: float) -> float:
    """Shift a probability in log-odds so the model's own decision threshold maps to 0.5."""
    return _sigmoid(_logit(probability) - _logit(threshold))


def credited_auc(model_auc: float | None, shortcut_auc: float | None = None) -> float:
    """The AUC that can be credited to the disease: what the model reaches beyond a model with no disease information."""
    if model_auc is None:
        return 0.5
    if shortcut_auc is None:
        return float(model_auc)
    return 0.5 + max(0.0, float(model_auc) - float(shortcut_auc))


def reliability_weight(auc: float | None) -> float:
    return max((0.5 if auc is None else float(auc)) - 0.5, FLOOR)


def confidence_level(score: float) -> str:
    distance = abs(score - 0.5)
    return "High" if distance >= 0.25 else "Moderate" if distance >= 0.10 else "Low"


def _shift(level: str, steps: int) -> str:
    return LEVELS[min(max(LEVELS.index(level) + steps, 0), len(LEVELS) - 1)]


@dataclass
class ModelInput:
    """What one model contributes for the current input."""
    key: str                                  # "voice" or "gait"
    label: str                                # name shown to the user
    probability: float | None                 # P(Parkinson's); None if the model did not run or failed
    threshold: float = 0.5                    # the model's own decision threshold
    credited_auc: float | None = None         # validated skill that can be credited to Parkinson's
    usable: bool = True                       # False if this particular result should not be trusted
    exclusion_reason: str | None = None
    error: str | None = None


@dataclass
class Contribution:
    key: str
    label: str
    probability: float
    threshold: float
    aligned: float
    verdict: str                              # "Parkinson's pattern" / "No Parkinson's pattern"
    weight: float
    share: float                              # fraction of the combined score this model carries
    included: bool
    note: str


@dataclass
class FusionResult:
    status: str                               # "none", "single", "fused" or "inconclusive"
    label: str                                # "PD likely", "Healthy", "Inconclusive" or "No result"
    score: float | None
    confidence: str | None
    agreement: str                            # "agree", "disagree", "single" or "n/a"
    headline: str
    contributions: list[Contribution] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


def _verdict(aligned: float) -> str:
    return "Parkinson's pattern" if aligned >= 0.5 else "No Parkinson's pattern"


def fuse(*inputs: ModelInput | None) -> FusionResult:
    """Combine the models that have input (voice, gait video); ``None`` means that model had no input."""
    models = [m for m in inputs if m is not None]
    present = [m for m in models if m.probability is not None]
    reasons: list[str] = []
    caveats: list[str] = []
    for m in models:
        if m.error:
            reasons.append(f"The {m.label} could not run: {m.error}")

    if not present:
        return FusionResult("none", "No result", None, None, "n/a",
                            "Add a voice recording and/or a gait video to get a screening result.", [], reasons, caveats)

    weights = {m.key: (reliability_weight(m.credited_auc) if m.usable else 0.0) for m in present}
    for m in present:
        if not m.usable:
            reasons.append(f"The {m.label} was left out of the combined score: {m.exclusion_reason}")
    informative = [m for m in present if weights[m.key] > INFORMATIVE]
    total = sum(weights.values())

    contributions: list[Contribution] = []
    for m in present:
        aligned = align(m.probability, m.threshold)
        share = weights[m.key] / total if total > 0 else 0.0
        if not m.usable:
            note = "Left out of the combined score for this input."
        elif weights[m.key] <= INFORMATIVE:
            note = ("Its accuracy cannot be credited to Parkinson's, so it carries almost no weight.")
        else:
            note = "Trusted according to its validated accuracy."
        contributions.append(Contribution(m.key, m.label, float(m.probability), m.threshold, aligned, _verdict(aligned),
                                          weights[m.key], share, m.usable, note))

    if not informative:
        names = ", ".join(f"the {m.label}" for m in present)
        reasons.append(f"{names[0].upper() + names[1:]} cannot be credited with detecting Parkinson's, so on its own it gives no verdict.")
        return FusionResult("inconclusive", "Inconclusive", None, None, "n/a",
                            "Not enough reliable evidence for a result. Add a voice recording.", contributions, reasons,
                            ["Do not read the raw score of a model whose accuracy cannot be credited to Parkinson's."])

    score = sum(c.aligned * c.weight for c in contributions) / total
    label = "PD likely" if score >= 0.5 else "Healthy"
    confidence = confidence_level(score)
    used = [c for c in contributions if c.weight > 0]
    informative_used = [c for c in used if c.weight > INFORMATIVE]

    if len(used) == 1:
        status, agreement = "single", "single"
        if confidence == "High":
            confidence = "Moderate"
        reasons.append(f"The result is based on the {used[0].label} alone, so confidence is capped at Moderate.")
    else:
        status = "fused"
        sides = {c.aligned >= 0.5 for c in used}
        agreement = "agree" if len(sides) == 1 else "disagree"
        minor = [c for c in used if c.weight <= INFORMATIVE]
        if len(informative_used) >= 2:
            many = "Two independent models" if len(informative_used) == 2 else f"All {len(informative_used)} informative models"
            sides_informative = {c.aligned >= 0.5 for c in informative_used}
            if len(sides_informative) == 1:
                confidence = _shift(confidence, +1)
                reasons.append(f"{many} agree, which raises confidence one level.")
            else:
                confidence = _shift(confidence, -1)
                reasons.append("The informative models disagree, which lowers confidence one level.")
            for c in minor:
                reasons.append(f"The {c.label} carries only {c.share:.0%} of the combined score, because its accuracy cannot be "
                               "credited to Parkinson's.")
        else:
            lead = max(used, key=lambda c: c.weight)
            others = [c for c in used if c is not lead]
            listed = " and ".join(f"the {c.label} {c.share:.0%}" for c in others)
            reasons.append(f"The {lead.label} carries {lead.share:.0%} of the combined score and {listed}, because the accuracy of "
                           "the other model(s) cannot be credited to Parkinson's. The models "
                           + ("agree." if agreement == "agree" else "disagree, which is noted but barely moves the result."))
        caveats.append("The two models were trained on different people, so this combination has not been validated on "
                       "people who have both a voice recording and a gait video.")
    if label == "Healthy":
        caveats.append("'Healthy' means no strong Parkinson's pattern was found. It does not rule out Parkinson's disease.")
    headline = (f"{label}: combined score {score:.0%} (decision threshold 50%), confidence {confidence}."
                if status == "fused" else f"{label}: score {score:.0%} from the {used[0].label}, confidence {confidence}.")
    return FusionResult(status, label, float(score), confidence, agreement, headline, contributions, reasons, caveats)


def _from_audit(path: Path | None, name: str) -> dict:
    """Credited AUC from a gait audit file: the model's AUC minus the best AUC reached by models that see no gait."""
    if path is None or not Path(path).exists():
        return {"credited_auc": 0.5, "model_auc": None, "shortcut_auc": None,
                "basis": f"No audit file found, so the {name} is treated as unvalidated."}
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    shortcut = max(v["cv_auc"] for v in audit["no_gait_baselines"].values())
    return {"credited_auc": credited_auc(audit["gait_model"]["cv_auc"], shortcut), "model_auc": audit["gait_model"]["cv_auc"],
            "shortcut_auc": shortcut, "basis": audit["finding"]}


def reliability_from_files(voice_deployment: dict | None, gait_audit_path: Path | None) -> dict[str, dict]:
    """Credited AUC and the evidence behind it, for the fusion weights (keys: voice, gait)."""
    cv = ((voice_deployment or {}).get("evaluation") or {}).get("cv") or {}
    voice_auc = cv.get("roc_auc")
    return {"voice": {"credited_auc": credited_auc(voice_auc), "model_auc": voice_auc,
                      "basis": "person-wise cross-validation on 64 people (held-out test AUC is similar)"},
            "gait": _from_audit(gait_audit_path, "gait-video model")}
