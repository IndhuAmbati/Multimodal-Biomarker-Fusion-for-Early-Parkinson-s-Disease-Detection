"""PD Screening: a voice model (Model 1) and a gait-video model (Model 2), combined into one result.

    streamlit run app/streamlit_app.py

Data flow: inputs -> Model 1 (voice) and Model 2 (gait video), each independent -> fusion (reliability-weighted) ->
combined result, shown above the two individual results. Either model can be used alone, and one failing never
affects the other.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdscreen import compat, config, fusion  # noqa: E402
from pdscreen.audio import AudioError  # noqa: E402

compat.pandas_available()  # settle this once, before any library probes for pandas

import tools  # noqa: E402
import views  # noqa: E402

GAIT_AUDIT = ROOT / "models" / "gait" / "audit.json"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]

st.set_page_config(page_title="PD Screening", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
views.inject_css()


# ------------------------------------------------------------------ model loading (cached, and failures are contained)

@st.cache_resource(show_spinner="Loading the voice model (first start only)...")
def load_voice():
    try:
        from pdscreen.inference import Screener

        screener = Screener()
        screener.warm_up()
        return screener, None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


@st.cache_resource(show_spinner="Loading the gait-video model...")
def load_gait():
    try:
        from pdscreen.gait import config as gait_config
        from pdscreen.gait.inference import GaitScreener

        if not gait_config.POSE_MODEL.exists():
            return None, "the MediaPipe pose model is missing (models/gait/pose_landmarker_full.task)"
        return GaitScreener(), None
    except ImportError as error:
        return None, f"a required library is not installed ({error.name or error})"
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


@st.cache_resource(show_spinner=False)
def load_identifier():
    try:
        from pdscreen.voice_id import VoiceIdentifier

        return VoiceIdentifier()
    except Exception:
        return None


@st.cache_data
def held_out_examples() -> dict[str, str]:
    """A few recordings from the held-out people, whom the deployed models never saw."""
    path = config.DATA_PROCESSED / "figshare_features.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        test = [r for r in csv.DictReader(f) if r["test"] == "True"]
    examples = {}
    for label, name in [("1", "has PD"), ("0", "healthy")]:
        for i, row in enumerate([r for r in test if r["label"] == label][:3], start=1):
            examples[f"Unseen person {i} who {name} (age {float(row['age']):.0f}, {row['sex']})"] = row["path"]
    return examples


def read_audit() -> dict | None:
    return json.loads(GAIT_AUDIT.read_text(encoding="utf-8")) if GAIT_AUDIT.exists() else None


# ------------------------------------------------------------------ page header

st.title("🧠 PD Screening")
st.markdown("<div class='page-sub'>Two independent models, voice and gait video, combined into one result.</div>",
            unsafe_allow_html=True)
st.markdown(f"<div class='disclaimer'>⚠️ {config.DISCLAIMER} These tools estimate whether a voice or a way of walking shows "
            "patterns seen in people with Parkinson's disease. They cannot diagnose anything.</div>", unsafe_allow_html=True)

voice, voice_error = load_voice()
gait, gait_error = load_gait()
reliability = fusion.reliability_from_files(voice.deployment if voice else None, GAIT_AUDIT)

with st.sidebar:
    st.header("Voice: how to record")
    st.markdown("1. Sit in a quiet room, 20-30 cm from the microphone.\n"
                "2. Take a breath and say **'aaah'** at a comfortable, steady pitch and loudness.\n"
                "3. Hold it for **5-10 seconds**, then stop the recording.\n"
                "4. Allow microphone access if the browser asks.")
    st.header("Gait video: how to film")
    st.markdown("- One person walking, with the **legs in view**.\n- A steady camera, 10 seconds or more of walking.\n"
                "- The analysis takes roughly half a second per second of video.")
    st.header("Privacy")
    st.caption("Audio and video are processed on this computer. Audio is never saved, and an uploaded video is deleted "
               "straight after analysis.")
    if voice is not None and voice.deployment.get("evaluation", {}).get("test"):
        t = voice.deployment["evaluation"]["test"]
        st.header("Model 1 at a glance")
        st.caption(f"{voice.deployment['name']}\n\nHeld-out test (17 people): AUC {t['roc_auc']:.2f}, "
                   f"sensitivity {t['sensitivity']:.2f}, specificity {t['specificity']:.2f}")

tab_screen, tab_tools, tab_about = st.tabs(["PD Screening", "Voice tools", "How it works"])

# ================================================================== PD SCREENING
with tab_screen:
    views.step_heading("1", "Provide inputs")
    st.caption("Use either input on its own, or both for a combined result.")
    examples = held_out_examples()
    example_bytes = None
    with st.container(key="inputs"):
        col_voice, col_gait = st.columns(2, gap="medium")
        with col_voice, st.container(border=True, key="card_voice"):
            views.card_header("1", "Model 1 · Voice", "Say a steady 'aaah' for 5-10 seconds, or upload a WAV recording.")
            recording = upload = None
            if voice is None:
                st.error(f"The voice model could not be loaded: {voice_error}")
            else:
                tab_rec, tab_up, tab_ex = st.tabs(["Record", "Upload", "Example"])
                with tab_rec:
                    recording = st.audio_input("Record voice", sample_rate=16_000, key="mic", label_visibility="collapsed")
                    st.caption("Allow microphone access if your browser asks. Use the recorder's controls to stop and re-record.")
                with tab_up:
                    upload = st.file_uploader("Upload recording", type=["wav"], key="upload", label_visibility="collapsed")
                with tab_ex:
                    if examples:
                        pick = st.selectbox("Example recording", ["(none)", *examples], key="example")
                        if pick != "(none)":
                            example_bytes = Path(examples[pick]).read_bytes()
                        st.caption("Recordings from people the models never saw during training.")
                    else:
                        st.caption("No example recordings are available.")
            voice_status = st.empty()
        with col_gait, st.container(border=True, key="card_gait"):
            views.card_header("2", "Model 2 · Gait video", "Upload a video of a person walking, with the legs in view (10 s or more).")
            video, seconds, run_gait = None, 30, False
            if gait is None:
                st.warning(f"Gait-video analysis is unavailable: {gait_error}. Voice screening still works.")
            else:
                video = st.file_uploader("Upload a walking video", type=VIDEO_TYPES, key="video", label_visibility="collapsed")
                seconds = st.slider("Analyse the first N seconds", 10, 60, 30, step=5, key="gait_seconds",
                                    help="Longer analysis is slightly more reliable but takes longer.")
                run_gait = st.button("Analyse video", type="primary", key="run_gait", disabled=video is None, width="stretch")
                st.caption("The video is processed on this computer and deleted straight after analysis.")
            gait_status = st.empty()

    # ---------------------------------------------------------- Model 1: voice (runs as soon as there is input)
    sources = {"microphone": recording.getvalue() if recording else None,
               "uploaded file": upload.getvalue() if upload else None, "example": example_bytes}
    active_source = tools.pick_latest("screening", sources) if voice is not None else None
    voice_outcome = None
    if active_source is None:
        st.session_state.pop("voice_outcome", None)
    else:
        audio_bytes = sources[active_source]
        fingerprint = hashlib.sha256(audio_bytes).hexdigest()
        voice_outcome = st.session_state.get("voice_outcome")
        if not voice_outcome or voice_outcome["fp"] != fingerprint:
            try:
                with st.status("Model 1: analysing the voice recording...", expanded=False) as status:
                    st.write("Cleaning the audio and extracting jitter, shimmer, HNR, PPE, RPDE, DFA and MFCCs...")
                    res, cleaned = voice.predict_bytes(audio_bytes)
                    st.write("Scoring with the feature model and the spectrogram CNN...")
                    status.update(label=f"Model 1 done in {res.latency_ms['total'] / 1000:.2f} s", state="complete")
                voice_outcome = {"fp": fingerprint, "source": active_source, "result": res.as_dict(), "cleaned": cleaned, "error": None}
            except AudioError as error:
                voice_outcome = {"fp": fingerprint, "source": active_source, "error": str(error)}
            except Exception as error:
                voice_outcome = {"fp": fingerprint, "source": active_source,
                                 "error": f"Something unexpected went wrong ({type(error).__name__}). Please try again."}
            st.session_state["voice_outcome"] = voice_outcome

    # ---------------------------------------------------------- Model 2: gait video (runs when the button is pressed)
    gait_outcome = st.session_state.get("gait_outcome")
    video_id = (getattr(video, "file_id", None) or f"{video.name}:{video.size}") if video is not None else None
    if video is not None and run_gait:
        slot = st.empty()
        bar = slot.progress(0.0, text="Model 2: starting...")
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix or ".mp4", delete=False) as handle:
                handle.write(video.getvalue())
                path = Path(handle.name)
            res = gait.analyse(path, max_seconds=float(seconds),
                               progress=lambda f: bar.progress(min(f, 1.0), text=f"Model 2: finding the person and measuring their pose... {f:.0%}"))
            bar.progress(1.0, text="Model 2: summarising...")
            gait_outcome = {"video_id": video_id, "seconds": seconds, "name": video.name, "result": res,
                            "frames": gait.skeleton_frames(path, res), "error": None}
        except Exception as error:
            from pdscreen.gait.inference import VideoError

            message = str(error) if isinstance(error, VideoError) else \
                f"Something unexpected went wrong ({type(error).__name__}). Please try another video."
            gait_outcome = {"video_id": video_id, "seconds": seconds, "name": video.name, "error": message}
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
            slot.empty()
        st.session_state["gait_outcome"] = gait_outcome
    gait_current = gait_outcome if (gait_outcome and video_id is not None and gait_outcome["video_id"] == video_id) else None

    # ---------------------------------------------------------- input status chips
    with voice_status.container():
        if voice is None:
            views.state_chip("err", "Voice model unavailable")
        elif voice_outcome is None:
            views.state_chip("wait", "Waiting for a recording")
        elif voice_outcome["error"]:
            views.state_chip("err", "Could not analyse this recording")
        else:
            views.state_chip("ok", f"✓ Analysed ({voice_outcome['source']})")
    with gait_status.container():
        if gait is None:
            views.state_chip("err", "Gait-video model unavailable")
        elif video is None:
            views.state_chip("wait", "No video yet")
        elif gait_current is None:
            views.state_chip("wait", "Ready: click Analyse video")
        elif gait_current["error"]:
            views.state_chip("err", "Could not analyse this video")
        else:
            views.state_chip("ok", f"✓ Analysed ({gait_current['name']})")

    # ---------------------------------------------------------- fusion
    voice_in = gait_in = None
    if voice_outcome:
        if voice_outcome["error"]:
            voice_in = fusion.ModelInput("voice", "voice model", None, error=voice_outcome["error"])
        else:
            r = voice_outcome["result"]
            voice_in = fusion.ModelInput("voice", "voice model", r["probability_pd"], threshold=r["model"]["threshold"],
                                         credited_auc=reliability["voice"]["credited_auc"])
    if gait_current:
        if gait_current["error"]:
            gait_in = fusion.ModelInput("gait", "gait-video model", None, error=gait_current["error"])
        else:
            r = gait_current["result"]
            gait_in = fusion.ModelInput("gait", "gait-video model", r["probability_pd"], threshold=0.5,
                                        credited_auc=reliability["gait"]["credited_auc"], usable=not r["shortcut_risk"],
                                        exclusion_reason="a model that sees only camera framing reaches the same verdict, "
                                                         "so the score may reflect how the video was filmed")
    fused = fusion.fuse(voice_in, gait_in)

    def state_of(key: str) -> str:
        c = views.contribution(fused, key)
        return "off" if c is None else ("used" if c.included else "excluded")

    # ---------------------------------------------------------- results
    views.step_heading("2", "Your screening result")
    views.hero(fused, state_of("voice"), state_of("gait"))

    views.step_heading("3", "How each model contributed")
    with st.container(key="models"):
        col_v, col_g = st.columns(2, gap="medium")
        with col_v, st.container(border=True, key="model_voice"):
            if voice_outcome is None:
                views.model_card_empty("Model 1 · Voice: add a recording above to see its result.")
            elif voice_outcome["error"]:
                views.model_error_card("1", "Model 1 · Voice", f"Could not analyse the recording: {voice_outcome['error']}",
                                       "Try again in a quiet room with at least a few seconds of steady 'aaah'.")
            else:
                views.voice_card(voice_outcome["result"], views.contribution(fused, "voice"), voice.deployment)
        with col_g, st.container(border=True, key="model_gait"):
            if gait is None:
                views.model_error_card("2", "Model 2 · Gait video", "The gait-video model is unavailable on this computer.", str(gait_error))
            elif gait_current is None:
                views.model_card_empty("Model 2 · Gait video: upload a video and click Analyse video to see its result.")
            elif gait_current["error"]:
                views.model_error_card("2", "Model 2 · Gait video", f"Could not analyse the video: {gait_current['error']}",
                                       "Use a video of one person walking with the legs in view.")
            else:
                views.gait_card(gait_current["result"], views.contribution(fused, "gait"))
    if gait_current and not gait_current["error"] and gait_current["seconds"] != seconds:
        st.caption("The analysis length was changed after this video was analysed. Click Analyse video to refresh it.")

    with st.expander("How the combined result was calculated"):
        views.fusion_explainer(fused, reliability)

    if any(o and not o["error"] for o in (voice_outcome, gait_current)):
        st.markdown("#### Details")

        def show_voice():
            views.voice_details(voice_outcome["result"], voice_outcome["cleaned"], voice)

        def show_gait():
            views.gait_details(gait_current["result"], gait_current["frames"], read_audit())

        panels = [("Voice details", voice_outcome, show_voice, "Add a voice recording to see waveform, spectrogram and explanation."),
                  ("Gait-video details", gait_current, show_gait, "Analyse a video to see the tracked pose and the gait measurements.")]
        panels.sort(key=lambda p: not (p[1] and not p[1]["error"]))          # tabs with a result come first (and open first)
        for tab, (_, outcome, show, hint) in zip(st.tabs([p[0] for p in panels]), panels):
            with tab:
                if outcome and not outcome["error"]:
                    show()
                else:
                    st.caption(hint)

# ================================================================== VOICE TOOLS
with tab_tools:
    views.step_heading("•", "Voice tools")
    st.caption("Extra tools that work on any voice and do not take part in the Parkinson's result.")
    with st.container(key="cards"):
        col_id, _spacer = st.columns([1, 1], gap="medium")
        with col_id, st.container(border=True, key="card_identify"):
            views.card_header("3", "Identify Unknown Voice", "Estimate whether a voice is likely male or female.")
            tools.render_identify(load_identifier())

# ================================================================== HOW IT WORKS
with tab_about:
    audit = read_audit()
    st.markdown("""
**Model 1 · Voice.** A sustained "aaah" is cleaned, split into 1.5 s windows, and 50 voice measurements (pitch, jitter,
shimmer, harmonics-to-noise, PPE, RPDE, DFA, MFCCs) go to a calibrated random forest. A spectrogram CNN adds a heatmap
explanation. The decision threshold (0.62) was chosen on out-of-fold people to keep false alarms on healthy people low.

**Model 2 · Gait video.** MediaPipe finds the body keypoints in the video; only the legs are used, measured in leg-lengths
and seconds; features from 3 s windows go to a random forest.

**Fusion.** Each model's probability is put on a common scale and combined with weights equal to its *credited* validated
AUC minus 0.5 (see *How the combined result was calculated* on the result page).
""")
    st.markdown("#### How much each model can be trusted")
    rows = []
    for key, name in (("voice", "Model 1 · Voice"), ("gait", "Model 2 · Gait video")):
        rel = reliability[key]
        rows.append({"Model": name, "Validated AUC": "-" if rel.get("model_auc") is None else f"{rel['model_auc']:.2f}",
                     "Credited AUC": f"{rel['credited_auc']:.2f}", "Fusion weight": f"{fusion.reliability_weight(rel['credited_auc']):.2f}",
                     "Why": rel["basis"]})
    st.markdown(views.markdown_table(rows))
    st.warning("The gait-video model scores a perfect AUC on held-out people, but so do models that are shown no gait at all "
               "(video frame rate and size, camera framing). Its Parkinson's videos and its normal videos were filmed with "
               "different cameras, framings and tasks, so its accuracy cannot be credited to Parkinson's. It therefore carries "
               "almost no weight, and is left out entirely when the same shortcut is detected in your video.")
    if audit:
        names = {"camera_framing_only": "Camera framing only (no gait)", "video_metadata_only": "Video frame rate and size only",
                 "task_features_only": "How the person moves through the scene only"}
        st.markdown(views.markdown_table(
            [{"Information given to the gait model": "Gait + task features (deployed)", "Test AUC": f"{audit['gait_model']['test_auc']:.2f}"}]
            + [{"Information given to the gait model": names[k], "Test AUC": f"{b['test_auc']:.2f}"} for k, b in audit["no_gait_baselines"].items()]))
    st.caption("The two models were trained on different people, so the combination has not been validated on people who "
               "have both a voice recording and a gait video.")
