import json

import pytest

from pdscreen import fusion
from pdscreen.fusion import ModelInput


def voice(p, auc=0.65, **kw):
    return ModelInput("voice", "Voice model", p, threshold=0.62, credited_auc=auc, **kw)


def gait(p, auc=0.5, **kw):
    return ModelInput("gait", "Gait-video model", p, threshold=0.5, credited_auc=auc, **kw)


def test_align_puts_each_models_threshold_at_one_half_and_keeps_order():
    assert fusion.align(0.62, 0.62) == pytest.approx(0.5)
    assert fusion.align(0.5, 0.5) == pytest.approx(0.5)
    values = [fusion.align(p, 0.62) for p in (0.2, 0.5, 0.62, 0.8)]
    assert values == sorted(values) and values[0] < 0.5 < values[-1]


def test_credited_auc_only_counts_skill_beyond_a_model_that_sees_no_gait():
    assert fusion.credited_auc(1.0, 1.0) == pytest.approx(0.5)      # same score without any gait: nothing to credit
    assert fusion.credited_auc(0.9, 0.6) == pytest.approx(0.8)
    assert fusion.credited_auc(0.65, None) == pytest.approx(0.65)
    assert fusion.credited_auc(0.55, 0.7) == pytest.approx(0.5)     # never below chance
    assert fusion.credited_auc(None) == 0.5


def test_reliability_weight_follows_auc_minus_half_with_a_floor():
    assert fusion.reliability_weight(0.65) == pytest.approx(0.15)
    assert fusion.reliability_weight(0.5) == fusion.FLOOR
    assert fusion.reliability_weight(0.4) == fusion.FLOOR and fusion.reliability_weight(None) == fusion.FLOOR


def test_no_inputs_gives_no_result():
    result = fusion.fuse(None, None)
    assert result.status == "none" and result.label == "No result" and result.score is None
    # A model that failed counts as missing and its error is reported.
    result = fusion.fuse(ModelInput("voice", "Voice model", None, error="bad audio"), None)
    assert result.status == "none" and any("bad audio" in r for r in result.reasons)


def test_voice_only_uses_the_voices_own_threshold_and_caps_confidence():
    below, above = fusion.fuse(voice(0.55), None), fusion.fuse(voice(0.80), None)
    assert below.label == "Healthy" and above.label == "PD likely"
    assert above.status == "single" and above.agreement == "single" and above.confidence in {"Low", "Moderate"}
    assert above.contributions[0].share == pytest.approx(1.0)


def test_gait_alone_is_inconclusive_because_its_accuracy_cannot_be_credited():
    result = fusion.fuse(None, gait(0.97))
    assert result.status == "inconclusive" and result.label == "Inconclusive" and result.score is None
    assert result.contributions[0].verdict == "Parkinson's pattern"          # shown, but not turned into a verdict


def test_untrusted_gait_barely_moves_the_voice_result_even_when_it_disagrees():
    base = fusion.fuse(voice(0.30), None)
    fused = fusion.fuse(voice(0.30), gait(0.99))
    assert fused.status == "fused" and fused.label == base.label == "Healthy"
    shares = {c.key: c.share for c in fused.contributions}
    assert shares["voice"] > 0.9 and shares["gait"] < 0.1 and sum(shares.values()) == pytest.approx(1.0)
    assert fused.agreement == "disagree" and any("barely moves" in r for r in fused.reasons)
    assert abs(fused.score - base.score) < 0.08


def test_gait_flagged_for_this_input_is_excluded_entirely():
    result = fusion.fuse(voice(0.80), gait(0.97, usable=False, exclusion_reason="camera framing alone gives the same verdict"))
    assert result.status == "single"
    shares = {c.key: c.share for c in result.contributions}
    assert shares == {"voice": pytest.approx(1.0), "gait": pytest.approx(0.0)}
    assert any("left out" in r and "camera framing" in r for r in result.reasons)


def test_two_informative_models_raise_confidence_when_they_agree_and_lower_it_when_they_disagree():
    # A hypothetical future gait model with real, credited skill.
    agree = fusion.fuse(voice(0.80), gait(0.85, auc=0.80))
    split = fusion.fuse(voice(0.80), gait(0.15, auc=0.80))
    alone = fusion.fuse(voice(0.80), None)
    order = ["Low", "Moderate", "High"]
    assert agree.agreement == "agree" and split.agreement == "disagree"
    assert order.index(agree.confidence) >= order.index(alone.confidence) or alone.confidence == "Moderate"
    assert order.index(split.confidence) < order.index(agree.confidence)
    assert 0.3 < {c.key: c.share for c in agree.contributions}["gait"] < 0.8


def test_the_combined_score_is_the_weighted_mean_of_the_aligned_probabilities():
    result = fusion.fuse(voice(0.70), gait(0.40, auc=0.65))
    expected = (fusion.align(0.70, 0.62) * 0.15 + fusion.align(0.40, 0.5) * 0.15) / 0.30
    assert result.score == pytest.approx(expected)


def test_a_fused_result_always_carries_the_validation_caveat():
    assert any("not been validated" in c for c in fusion.fuse(voice(0.7), gait(0.9)).caveats)
    assert any("does not rule out" in c for c in fusion.fuse(voice(0.3), None).caveats)


def test_reliability_is_read_from_the_deployment_and_audit_files(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"gait_model": {"cv_auc": 1.0}, "finding": "same score without gait",
                                 "no_gait_baselines": {"a": {"cv_auc": 0.9}, "b": {"cv_auc": 1.0}}}))
    rel = fusion.reliability_from_files({"evaluation": {"cv": {"roc_auc": 0.65}}}, audit)
    assert rel["voice"]["credited_auc"] == pytest.approx(0.65) and rel["gait"]["credited_auc"] == pytest.approx(0.5)
    missing = fusion.reliability_from_files(None, tmp_path / "nope.json")
    assert missing["gait"]["credited_auc"] == 0.5 and missing["voice"]["credited_auc"] == 0.5
