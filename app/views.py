"""Rendering helpers for the PD Screening page: styles, cards, the fused-result hero and the detail views.

Nothing here runs a model. The page script (streamlit_app.py) runs the models and the fusion, then hands the results
to these functions. Tables and charts avoid pandas on purpose (see pdscreen.compat).
"""

from __future__ import annotations

import html

import numpy as np
import streamlit as st

from pdscreen import config, features, voice_id
from pdscreen.fusion import FusionResult

CSS = """
/* ---------- result colours (shared by every result block) ---------- */
.result { border-radius: 14px; padding: 1.1rem 1.4rem; margin: .4rem 0 1rem; }
.result h2 { margin: 0 0 .3rem; font-size: 1.7rem; }
.result p { margin: 0; }
.pd { background: #fdecea; border: 1px solid #f5b5ad; color: #7d1d13; }
.hc { background: #e8f6ee; border: 1px solid #a6d8b9; color: #14532d; }
.neutral { background: #eef2f7; border: 1px solid #b8c6d8; color: #1c3350; }
.disclaimer { background: #fff7e0; border: 1px solid #f0d58c; border-radius: 10px; padding: .6rem 1rem;
              color: #5c4500; font-weight: 600; }

/* ---------- page header ---------- */
.page-sub { font-size: 1.02rem; opacity: .8; margin: -.4rem 0 .9rem; }
.step { display: flex; align-items: center; gap: .6rem; margin: 1.3rem 0 .6rem; font-size: 1.25rem; font-weight: 700; }
.step .num { width: 1.9rem; height: 1.9rem; border-radius: 50%; background: #1f5d7a; color: #fff; font-size: 1rem;
             display: inline-flex; align-items: center; justify-content: center; }

/* ---------- cards: two side by side, wrapping to one column on narrow screens ---------- */
.st-key-inputs [data-testid="stHorizontalBlock"], .st-key-models [data-testid="stHorizontalBlock"],
.st-key-cards [data-testid="stHorizontalBlock"] { flex-wrap: wrap; align-items: stretch; row-gap: 1rem; }
.st-key-inputs [data-testid="stColumn"], .st-key-models [data-testid="stColumn"] { flex: 1 1 320px !important; min-width: min(320px, 100%) !important; }
.st-key-cards [data-testid="stColumn"] { flex: 1 1 235px !important; min-width: min(235px, 100%) !important; }
.st-key-inputs [data-testid="stColumn"] > [data-testid="stVerticalBlock"], .st-key-models [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
.st-key-cards [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height: 100%; }
.st-key-inputs [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"],
.st-key-models [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"],
.st-key-cards [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] { flex: 1 1 auto; }
.st-key-card_voice, .st-key-card_gait, .st-key-model_voice, .st-key-model_gait, .st-key-card_identify { padding: 1.1rem 1.1rem 1rem; }
.card-head { display: flex; gap: .75rem; align-items: flex-start; margin-bottom: .8rem; }
.card-badge { flex: 0 0 2rem; width: 2rem; height: 2rem; border-radius: 50%; background: #1f5d7a; color: #fff;
              display: flex; align-items: center; justify-content: center; font-weight: 700; }
.card-title { font-size: 1.2rem; font-weight: 700; line-height: 1.25; }
.card-sub { font-size: .88rem; opacity: .75; line-height: 1.35; margin-top: .15rem; }
.state { font-size: .86rem; margin-top: .5rem; padding: .35rem .6rem; border-radius: 8px; display: inline-block; }
.state.ok { background: #e8f6ee; color: #14532d; } .state.wait { background: #eef2f7; color: #1c3350; }
.state.err { background: #fdecea; color: #7d1d13; }

/* ---------- the fused result (main focus) ---------- */
.hero { border-radius: 20px; padding: 1.5rem 1.7rem; margin: .3rem 0 1.2rem; border: 1px solid; box-shadow: 0 2px 10px rgba(0,0,0,.12); }
.hero.pd { background: linear-gradient(135deg, #fdecea, #fad3cd); border-color: #f0a69d; color: #6f1b14; }
.hero.hc { background: linear-gradient(135deg, #e8f6ee, #d0ebdb); border-color: #95d0ab; color: #124d2a; }
.hero.neutral { background: linear-gradient(135deg, #eef2f7, #e0e8f2); border-color: #b8c6d8; color: #1c3350; }
.hero-top { display: flex; flex-wrap: wrap; gap: .4rem .5rem; align-items: center; }
.hero-kicker { font-size: .82rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; opacity: .8; margin-right: .5rem; }
.pill { display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .76rem; font-weight: 700; border: 1px solid currentColor; }
.pill.off { opacity: .5; border-style: dashed; }
.hero-label { font-size: clamp(2rem, 6vw, 3rem); font-weight: 800; line-height: 1.08; margin: .55rem 0 .2rem; }
.hero-sub { font-size: clamp(.95rem, 2.5vw, 1.12rem); font-weight: 600; opacity: .92; }
.scorebar { margin: 1.1rem 0 .2rem; }
.track { position: relative; height: .7rem; border-radius: 999px; background: rgba(0,0,0,.13); }
.fill { height: 100%; border-radius: 999px; background: currentColor; opacity: .5; }
.tick { position: absolute; top: -.3rem; bottom: -.3rem; width: 2px; background: currentColor; left: 50%; opacity: .7; }
.dot { position: absolute; top: 50%; width: 1.15rem; height: 1.15rem; border-radius: 50%; background: currentColor;
       border: 3px solid #fff; transform: translate(-50%, -50%); box-shadow: 0 1px 4px rgba(0,0,0,.35); }
.scale { display: flex; justify-content: space-between; font-size: .74rem; opacity: .75; margin-top: .35rem; }
.hero-points { margin: .9rem 0 0; padding-left: 1.15rem; font-size: .95rem; line-height: 1.55; }
.hero-points li { margin-bottom: .2rem; }

/* ---------- the two model result cards ---------- */
.mhead { display: flex; gap: .7rem; align-items: flex-start; margin-bottom: .7rem; }
.mchip { margin-left: auto; font-size: .78rem; font-weight: 800; padding: .18rem .65rem; border-radius: 999px; white-space: nowrap; }
.mchip.pd { background: #fdecea; color: #7d1d13; border: 1px solid #f5b5ad; }
.mchip.hc { background: #e8f6ee; color: #14532d; border: 1px solid #a6d8b9; }
.mchip.neutral { background: #eef2f7; color: #1c3350; border: 1px solid #b8c6d8; }
.mstats { display: flex; flex-wrap: wrap; gap: .5rem 1.4rem; margin: .2rem 0 .6rem; }
.mstat b { display: block; font-size: 1.45rem; line-height: 1.15; }
.mstat span { font-size: .76rem; opacity: .75; }
.share { display: flex; height: .5rem; border-radius: 999px; overflow: hidden; background: rgba(128,128,128,.25); margin: .2rem 0 .15rem; }
.share > div { background: #1f5d7a; }
.small { font-size: .82rem; opacity: .8; }
.kv-wrap { overflow-x: auto; max-width: 100%; }
table.kv { width: 100%; border-collapse: collapse; font-size: .84rem; margin: .4rem 0; }
table.kv th { text-align: left; font-weight: 700; opacity: .75; padding: .25rem .4rem; border-bottom: 1px solid rgba(128,128,128,.35); }
table.kv td { padding: .3rem .4rem; border-bottom: 1px solid rgba(128,128,128,.18); }
.empty { text-align: center; padding: 1.4rem .6rem; opacity: .75; font-size: .95rem; }

/* ---------- identify-unknown-voice result (unchanged feature) ---------- */
.id-result { border-radius: 12px; padding: .9rem 1rem; margin: .7rem 0 .4rem; }
.id-label { font-size: 1.35rem; font-weight: 700; }
.id-sub { font-size: .92rem; margin: .1rem 0 .55rem; }
.id-bar { display: flex; height: .6rem; border-radius: 999px; overflow: hidden; background: rgba(0,0,0,.12); }
.id-male { background: #2f6fcf; } .id-female { background: #b0479f; }
.male { background: #e8f0fb; border: 1px solid #a9c4ec; color: #173b73; }
.female { background: #f8e9f6; border: 1px solid #e0b3da; color: #5a1b55; }
.uncertain { background: #fff7e0; border: 1px solid #f0d58c; color: #5c4500; }

@media (max-width: 640px) {
  .hero { padding: 1.1rem 1.1rem; border-radius: 16px; }
  .mstat b { font-size: 1.25rem; }
}
"""

KEY_VOICE_FEATURES = [("f0_std_st", "Pitch variation (semitones)"), ("jitter_local", "Pitch instability (jitter)"),
                      ("shimmer_local", "Loudness instability (shimmer)"), ("hnr_db", "Breathiness (HNR, dB)")]


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def markdown_table(rows: list[dict]) -> str:
    """A plain markdown table (st.dataframe needs pandas, which some Windows setups block)."""
    if not rows:
        return ""
    head = list(rows[0])
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join("" if r[h] is None else str(r[h]) for h in head) + " |" for r in rows]
    return "\n".join(lines)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<div class='kv-wrap'><table class='kv'><tr>{head}</tr>{body}</table></div>"


def card_header(step: str, title: str, subtitle: str) -> None:
    st.markdown(f"<div class='card-head'><div class='card-badge'>{esc(step)}</div><div>"
                f"<div class='card-title'>{esc(title)}</div><div class='card-sub'>{esc(subtitle)}</div></div></div>",
                unsafe_allow_html=True)


def step_heading(number: str, text: str) -> None:
    st.markdown(f"<div class='step'><span class='num'>{esc(number)}</span>{esc(text)}</div>", unsafe_allow_html=True)


def state_chip(kind: str, text: str) -> None:
    st.markdown(f"<span class='state {kind}'>{esc(text)}</span>", unsafe_allow_html=True)


# ------------------------------------------------------------------ the fused result

def hero(result: FusionResult, voice_state: str, gait_state: str) -> None:
    """The combined result, the main visual focus. *_state is 'used', 'excluded' or 'off'."""
    css = {"PD likely": "pd", "Healthy": "hc"}.get(result.label, "neutral")
    names = {"used": "✓", "excluded": "left out", "off": "no input"}
    pills = "".join(f"<span class='pill {'off' if s == 'off' else ''}'>{esc(n)}: {names[s]}</span>"
                    for n, s in (("Voice", voice_state), ("Gait video", gait_state)))
    label = {"No result": "Waiting for input"}.get(result.label, result.label)
    if result.score is not None:
        sub = f"{'Combined score' if result.status == 'fused' else 'Score'} {result.score:.0%} &nbsp;·&nbsp; Confidence {esc(result.confidence)}"
        bar = ("<div class='scorebar'><div class='track'><div class='fill' style='width:%.1f%%'></div><div class='tick'></div>"
               "<div class='dot' style='left:%.1f%%'></div></div><div class='scale'><span>0%%</span>"
               "<span>decision threshold 50%%</span><span>100%%</span></div></div>") % (result.score * 100, result.score * 100)
    else:
        sub, bar = esc(result.headline), ""
    points = [*result.reasons[:4], *result.caveats[:2]]
    items = "".join(f"<li>{esc(p)}</li>" for p in points)
    st.markdown(f"<div class='hero {css}'><div class='hero-top'><span class='hero-kicker'>Combined PD screening result</span>"
                f"{pills}</div><div class='hero-label'>{esc(label)}</div><div class='hero-sub'>{sub}</div>{bar}"
                f"<ul class='hero-points'>{items}</ul></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='disclaimer'>⚠️ {esc(config.DISCLAIMER)} This is not a diagnosis.</div>", unsafe_allow_html=True)


def contribution(result: FusionResult, key: str):
    return next((c for c in result.contributions if c.key == key), None)


def _stats(items: list[tuple[str, str]]) -> str:
    return "<div class='mstats'>" + "".join(f"<div class='mstat'><b>{esc(v)}</b><span>{esc(k)}</span></div>" for k, v in items) + "</div>"


def _share(share: float) -> str:
    return (f"<div class='share'><div style='width:{share * 100:.1f}%'></div></div>"
            f"<div class='small'>Carries {share:.0%} of the combined score</div>")


def model_card_empty(message: str) -> None:
    st.markdown(f"<div class='empty'>{esc(message)}</div>", unsafe_allow_html=True)


def voice_card(result: dict, contrib, deployment: dict) -> None:
    pd_likely = result["label"] == "PD likely"
    st.markdown(f"<div class='mhead'><div class='card-badge'>1</div><div><div class='card-title'>Model 1 · Voice</div>"
                "<div class='card-sub'>Sustained 'aaah' · acoustic features + calibrated random forest</div></div>"
                f"<span class='mchip {'pd' if pd_likely else 'hc'}'>{'PD pattern' if pd_likely else 'No PD pattern'}</span></div>",
                unsafe_allow_html=True)
    st.markdown(_stats([("P(Parkinson's)", f"{result['probability_pd']:.0%}"), ("P(Healthy)", f"{result['probability_healthy']:.0%}"),
                        ("Own threshold", f"{result['model']['threshold']:.0%}"), ("Confidence", result["confidence"])]),
                unsafe_allow_html=True)
    if contrib is not None:
        st.markdown(_share(contrib.share), unsafe_allow_html=True)
    reference = deployment.get("reference_medians", {})
    rows = []
    for name, title in KEY_VOICE_FEATURES:
        value, healthy, pd_med = result["features"].get(name), reference.get("healthy", {}).get(name), reference.get("pd", {}).get(name)
        if value is None or healthy is None or pd_med is None or not np.isfinite(value):
            continue
        rows.append([title, f"{value:.3g}", f"{healthy:.3g}", f"{pd_med:.3g}", "PD" if abs(value - pd_med) < abs(value - healthy) else "Healthy"])
    if rows:
        st.markdown("<div class='small'><b>Key indicators</b> (vs the training medians)</div>", unsafe_allow_html=True)
        st.markdown(html_table(["Measurement", "You", "Healthy", "PD", "Closer to"], rows), unsafe_allow_html=True)
    for warning in result["quality"]["warnings"]:
        st.warning(warning)
    for note in result["notes"]:
        st.info(note)


def gait_card(result: dict, contrib) -> None:
    pd_pattern = result["label"] == "Parkinson's"
    st.markdown(f"<div class='mhead'><div class='card-badge'>2</div><div><div class='card-title'>Model 2 · Gait video</div>"
                "<div class='card-sub'>Walking video · MediaPipe leg keypoints + random forest</div></div>"
                f"<span class='mchip {'pd' if pd_pattern else 'hc'}'>{'PD pattern' if pd_pattern else 'No PD pattern'}</span></div>",
                unsafe_allow_html=True)
    framing = result["framing_only_probability_pd"]
    st.markdown(_stats([("P(Parkinson's)", f"{result['probability_pd']:.0%}"), ("P(Healthy)", f"{result['probability_healthy']:.0%}"),
                        ("Framing-only model", "-" if framing is None else f"{framing:.0%} PD"), ("Confidence", result["confidence"])]),
                unsafe_allow_html=True)
    if contrib is not None:
        st.markdown(_share(contrib.share), unsafe_allow_html=True)
    f = result["features"]
    knee = np.nanmean([f.get("knee_angle_range_l", np.nan), f.get("knee_angle_range_r", np.nan)])
    clear = np.nanmean([f.get("foot_clearance_l", np.nan), f.get("foot_clearance_r", np.nan)])
    rows = [["Stride rate", f"{f.get('stride_freq_hz', float('nan')):.2f} /s", "How often a full stride repeats"],
            ["Step regularity", f"{f.get('step_regularity', float('nan')):.2f}", "1 = every step alike"],
            ["Knee bend range", f"{knee:.0f}°", "Larger = more knee movement"],
            ["Foot lift", f"{clear:.2f}", "Leg-lengths; lower = shuffling"],
            ["Time standing still", f"{f.get('rest_fraction', float('nan')):.0%}", "Share of the time the feet are not moving"]]
    st.markdown("<div class='small'><b>Key indicators</b> (not validated for Parkinson's)</div>", unsafe_allow_html=True)
    st.markdown(html_table(["Measurement", "Value", "Meaning"], rows), unsafe_allow_html=True)
    for warning in result["warnings"]:
        st.warning(warning)


def model_error_card(step: str, title: str, message: str, hint: str = "") -> None:
    st.markdown(f"<div class='mhead'><div class='card-badge'>{esc(step)}</div><div><div class='card-title'>{esc(title)}</div></div>"
                "<span class='mchip neutral'>No result</span></div>", unsafe_allow_html=True)
    st.error(message)
    if hint:
        st.caption(hint)


# ------------------------------------------------------------------ explanation of the fusion

def fusion_explainer(result: FusionResult, reliability: dict) -> None:
    rows = []
    for c in result.contributions:
        rel = reliability.get(c.key, {})
        rows.append([c.label.capitalize(), f"{c.probability:.0%}", f"{c.threshold:.0%}", f"{c.aligned:.0%}",
                     f"{rel.get('credited_auc', float('nan')):.2f}", f"{c.weight:.2f}", f"{c.share:.0%}", c.verdict])
    if rows:
        st.markdown(html_table(["Model", "Raw P(PD)", "Own threshold", "Aligned", "Credited AUC", "Weight", "Share", "Verdict"], rows),
                    unsafe_allow_html=True)
    st.markdown("""
**How the combined score is made.** (1) Each model's probability is shifted so that its own decision threshold becomes
50%, which puts the two on one scale. (2) Each model gets a weight of *credited AUC − 0.5* (never below 0.01), where the
credited AUC is the validated, person-wise AUC, counting only skill beyond what a model with no disease information
could reach. (3) The combined score is the weighted average of the aligned probabilities; 50% is the decision line.
(4) Two informative models that agree raise confidence one level, two that disagree lower it, and one model alone is
capped at *Moderate*. A model that cannot be credited with detecting Parkinson's never produces a verdict by itself.
""")
    for key, rel in reliability.items():
        st.caption(f"{'Voice' if key == 'voice' else 'Gait video'}: credited AUC {rel['credited_auc']:.2f}. {rel['basis']}")


# ------------------------------------------------------------------ voice details (previous results page)

def plot_waveform(y: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(y)) / config.SAMPLE_RATE
    times, f0 = features.pitch_contour(y)
    fig, axes = plt.subplots(2, 1, figsize=(8, 3.8), sharex=True)
    axes[0].plot(t, y, linewidth=0.4, color="#1f5d7a")
    axes[0].set_title("Voice waveform (cleaned: 8 kHz, band-passed, denoised, silence trimmed)", fontsize=9)
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(times, f0, color="#b5462c")
    axes[1].set_title("Pitch (F0) contour: a steady line means stable pitch control", fontsize=9)
    axes[1].set_ylabel("Hz")
    axes[1].set_xlabel("Seconds")
    fig.tight_layout()
    return fig


def plot_spectrogram_gradcam(cam: dict, duration: float, threshold: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [0, duration, 0, config.N_MELS]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.6), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.6]})
    axes[0].imshow(cam["log_mel"], origin="lower", aspect="auto", cmap="magma", extent=extent)
    axes[0].set_title("Log-Mel spectrogram of the recording", fontsize=9)
    axes[0].set_ylabel("Mel band (low -> high pitch)")
    axes[1].imshow(cam["log_mel"], origin="lower", aspect="auto", cmap="gray", extent=extent)
    axes[1].imshow(cam["cam"], origin="lower", aspect="auto", cmap="jet", alpha=0.5, extent=extent, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM: where the CNN found PD-like patterns (blue = none, red = strongest)", fontsize=9)
    axes[1].set_ylabel("Mel band")
    probs = cam["window_probabilities"]
    axes[2].plot(cam["window_times"], probs, marker="o", color="#7d1d13")
    axes[2].axhline(threshold, color="grey", linestyle="--", linewidth=1)
    axes[2].fill_between(cam["window_times"], threshold, probs, where=probs >= threshold, color="#e74c3c", alpha=0.25)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("CNN P(PD)")
    axes[2].set_xlabel("Seconds")
    axes[2].set_title("CNN P(PD) for each 1-second window (shaded where above the threshold)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_influence(contributions: list[tuple[str, float]], method: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = contributions[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(items) + 0.8))
    ax.barh([n for n, _ in items], [v for _, v in items], color=["#c0392b" if v > 0 else "#2e7d4f" for _, v in items])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Change in feature-model P(PD)   (red raises, green lowers)")
    ax.set_title(f"Which measurements drove this result ({method})", fontsize=9)
    fig.tight_layout()
    return fig


def voice_details(result: dict, cleaned: np.ndarray, screener) -> None:
    c = st.columns(5)
    c[0].metric("P(PD)", f"{result['probability_pd']:.1%}")
    c[1].metric("P(Healthy)", f"{result['probability_healthy']:.1%}")
    c[2].metric("Confidence", result["confidence"])
    c[3].metric("Voiced audio", f"{result['quality']['voiced_seconds']} s")
    c[4].metric("Latency", f"{result['latency_ms']['total'] / 1000:.2f} s")
    names = {"feature_model": "Feature model", "cnn": "Spectrogram CNN"}
    parts = " · ".join(f"{names[k]}: {v:.1%}" for k, v in result["components"].items())
    extra = "".join(f" Second opinion, not in the final score: {names[k]} {v:.1%}." for k, v in result["secondary_not_in_score"].items())
    windows = result.get("window_probabilities", [])
    spread = (f" Scored in {len(windows)} windows of 1.5 s (median taken); window P(PD) ranged "
              f"{min(windows):.0%} to {max(windows):.0%}.") if windows else ""
    st.caption(f"Final score from: {parts}.{extra}{spread} Decision threshold {result['model']['threshold']:.2f} (chosen to keep "
               "false alarms on healthy people low). Confidence: High if the probability is at least 25 points from the "
               "threshold, Moderate if 10-25 points, Low otherwise.")
    if result["label"] != "PD likely":
        st.caption("'Healthy' means no strong PD pattern was found in this recording. It does not rule out Parkinson's disease.")
    st.markdown("#### Waveform")
    st.pyplot(plot_waveform(cleaned))
    if screener.session is not None:
        st.markdown("#### Log-Mel spectrogram and where the model reacted")
        st.pyplot(plot_spectrogram_gradcam(screener.grad_cam(cleaned), len(cleaned) / config.SAMPLE_RATE, result["model"]["threshold"]))
        role = ("The CNN is part of the final score." if "cnn" in result["components"] else
                "The CNN is shown as an explanation aid; the final score comes from the feature model, which scored better in cross-validation.")
        st.caption("Grad-CAM highlights the time-frequency regions that raised the CNN's PD score. It shows what the model "
                   f"reacted to, not a location of the disease. {role}")
    tab_why, tab_features, tab_model = st.tabs(["Why this result", "Voice measurements", "About the model"])
    with tab_why:
        if screener.feature_bundle is not None:
            with st.spinner("Computing the explanation..."):
                contributions, method = screener.explain_features(result["features"])
            st.pyplot(plot_influence(contributions, method))
            st.caption("Each bar shows how much a voice measurement pushed this recording's score towards PD (red) or Healthy (green).")
    with tab_features:
        reference = screener.deployment.get("reference_medians", {})
        rows = [{"Measurement": features.FEATURE_DESCRIPTIONS.get(n, n), "Feature": n, "Your recording": f"{result['features'][n]:.4g}",
                 "Typical healthy": f"{reference.get('healthy', {}).get(n, float('nan')):.4g}",
                 "Typical PD": f"{reference.get('pd', {}).get(n, float('nan')):.4g}",
                 "Outside training range": "yes" if n in result["out_of_training_range"] else ""} for n in features.FEATURE_NAMES]
        st.markdown(markdown_table(rows))
    with tab_model:
        evaluation = result["model"].get("evaluation", {})
        st.markdown(f"**Deployed model:** {result['model'].get('name')}")
        for title, key in [("Cross-validation (64 people, person-wise folds, clean recordings)", "cv"),
                           ("Held-out test (17 unseen people, clean recordings)", "test")]:
            metrics = evaluation.get(key, {})
            if metrics:
                st.markdown(f"**{title}**")
                cols = st.columns(6)
                for col, metric in zip(cols, ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]):
                    col.metric(metric.replace("_", "-").upper() if metric == "roc_auc" else metric.capitalize(), f"{metrics[metric]:.2f}")
        robustness = screener.deployment.get("robustness", {})
        if robustness:
            st.markdown("**With pitch changes and simulated microphones (same people, perturbed audio)**")
            for title, key in [("Cross-validation", "cv"), ("Held-out test", "test")]:
                perturbed = robustness.get(key, {}).get("perturbed")
                if perturbed:
                    st.caption(f"{title}: AUC {perturbed['auc']:.2f}, sensitivity {perturbed['sensitivity']:.2f}, specificity {perturbed['specificity']:.2f}")
        st.markdown(screener.deployment.get("limitations_markdown", ""))


# ------------------------------------------------------------------ gait details

def plot_window_scores(scores: list[float]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seconds = np.arange(len(scores)) * 1.5
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(seconds, scores, marker="o", color="#7d1d13")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Seconds into the analysed part of the video")
    ax.set_ylabel("P(Parkinson's)")
    fig.tight_layout()
    return fig


def gait_details(result: dict, frames: list, audit: dict | None) -> None:
    v = result["video"]
    st.caption(f"Analysed {v['seconds_analysed']:.0f} s of a {v['width']}x{v['height']} video at {v['fps']:.0f} fps (resampled to "
               f"7 fps); the person was found in {v['detection_rate']:.0%} of frames; {result['windows']} windows of 3 s.")
    if frames:
        st.markdown("#### Pose tracked on the video")
        for row in (frames[:3], frames[3:6]):
            for col, frame in zip(st.columns(3), row):
                col.image(frame, use_container_width=True)
    st.markdown("#### Score over time")
    st.pyplot(plot_window_scores(result["window_probabilities"]))
    with st.expander("All gait measurements (median over windows)"):
        st.markdown(markdown_table([{"feature": k, "value": None if not np.isfinite(val) else round(val, 4)} for k, val in result["features"].items()]))
    if audit:
        st.markdown("**Why this model's score is not credited to Parkinson's**")
        rows = [{"Information given to the model": "Gait + task features (the deployed model)", "Test AUC": f"{audit['gait_model']['test_auc']:.2f}"}]
        names = {"camera_framing_only": "Camera framing only (no gait)", "video_metadata_only": "Video frame rate and size only",
                 "task_features_only": "How the person moves through the scene only"}
        rows += [{"Information given to the model": names[k], "Test AUC": f"{b['test_auc']:.2f}"} for k, b in audit["no_gait_baselines"].items()]
        st.markdown(markdown_table(rows))
        st.caption(audit["finding"])

