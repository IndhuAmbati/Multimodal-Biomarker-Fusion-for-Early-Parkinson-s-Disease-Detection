"""How does the CURRENT deployed feature model react to pitch changes and a microphone channel?

Uses the original pipeline (features from the whole recording, no windows) on the 17 held-out people, with the
same deterministic perturbations the robust model is later tested on.

    python scripts/baseline_robustness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdscreen import audio, augment, config, features  # noqa: E402


def main() -> None:
    bundle = joblib.load(config.MODELS / "feature_model_plain.joblib")
    table = pd.read_csv(config.DATA_PROCESSED / "figshare_features.csv")
    test = table[table.test]
    rows = []
    for r in test.itertuples():
        raw, rate = audio.load_audio(r.path)
        for version in augment.VERSIONS:
            x, rt = augment.make_version(raw, rate, version, r.subject_id)
            y, _ = audio.preprocess(x, rt)
            values = features.extract_features(y)
            p = float(bundle["model"].predict_proba(np.array([[values[c] for c in bundle["features"]]]))[:, 1][0])
            rows.append({"subject_id": r.subject_id, "label": r.label, "version": version,
                         "family": augment.family(version), "p_pd": p})
    frame = pd.DataFrame(rows)
    summary = {}
    for fam, g in frame.groupby("family"):
        hc, pdx = g[g.label == 0].p_pd, g[g.label == 1].p_pd
        summary[fam] = {"mean_p_pd_healthy": float(hc.mean()), "mean_p_pd_pd": float(pdx.mean()),
                        "specificity": float((hc < 0.5).mean()), "sensitivity": float((pdx >= 0.5).mean()),
                        "healthy_flagged_pd": float((hc >= 0.5).mean())}
    (config.REPORTS / "baseline_robustness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
