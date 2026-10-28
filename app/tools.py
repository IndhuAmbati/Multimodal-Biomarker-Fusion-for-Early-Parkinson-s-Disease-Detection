"""Voice tools: the 'Identify Unknown Voice' feature (male / female estimate) and the source picker both inputs share."""

from __future__ import annotations

import hashlib

import streamlit as st

from pdscreen import voice_id
from pdscreen.audio import AudioError

import views


def pick_latest(group: str, sources: dict[str, bytes | None]) -> str | None:
    """The source whose data changed most recently, so a new recording or file always wins."""
    seen = st.session_state.setdefault(f"seen_{group}", {})
    latest = st.session_state.get(f"latest_{group}")
    for source, data in sources.items():
        fingerprint = hashlib.sha256(data).hexdigest() if data else None
        if fingerprint != seen.get(source):
            seen[source] = fingerprint
            if fingerprint:
                latest = source
    st.session_state[f"latest_{group}"] = latest
    return latest if sources.get(latest) else next((s for s, d in sources.items() if d), None)


def show_identity(result: dict) -> None:
    female, male = result["probability_female"], result["probability_male"]
    style = {"Likely female": "female", "Likely male": "male", "Uncertain": "uncertain"}[result["label"]]
    st.markdown(
        f"<div class='id-result {style}'><div class='id-label'>{result['label']}</div>"
        f"<div class='id-sub'>Male {male:.0%} · Female {female:.0%}</div>"
        f"<div class='id-bar'><div class='id-male' style='width:{male * 100:.1f}%'></div>"
        f"<div class='id-female' style='width:{female * 100:.1f}%'></div></div></div>", unsafe_allow_html=True)
    if result["label"] == "Uncertain":
        st.caption("The voice sits between typical male and female patterns. Try a longer, steadier 'aaah'.")
    if result["pitch_hz"]:
        male_low, male_high = voice_id.TYPICAL_PITCH_HZ["male"]
        female_low, female_high = voice_id.TYPICAL_PITCH_HZ["female"]
        st.caption(f"Average pitch **{result['pitch_hz']:.0f} Hz**. Typical adult male voices are "
                   f"{male_low}-{male_high} Hz and female voices {female_low}-{female_high} Hz.")
    for warning in result["quality"]["warnings"]:
        st.warning(warning)
    st.caption(f"Analysed in {result['latency_ms'] / 1000:.2f} s. This describes how a voice sounds, not a person's "
               "gender identity, and can be wrong for children, voice disorders or unusually high or low voices.")


def render_identify(identifier) -> None:
    """The Identify Unknown Voice card body. ``identifier`` is a VoiceIdentifier, or None if it is not available."""
    if identifier is None:
        st.info("The voice-identification model is not available. Run `python scripts/train_voice_id.py`.")
        return
    tab_record, tab_upload = st.tabs(["Record", "Upload"])
    with tab_record:
        id_recording = st.audio_input("Record an unknown voice", sample_rate=16_000, key="id_mic", label_visibility="collapsed")
    with tab_upload:
        id_upload = st.file_uploader("Upload an unknown voice", type=["wav"], key="id_upload", label_visibility="collapsed")
    sources = {"recording": id_recording.getvalue() if id_recording else None,
               "file": id_upload.getvalue() if id_upload else None}
    active = pick_latest("identify", sources)
    if active is None:
        st.caption("Record or upload any voice to estimate whether it sounds male or female. "
                   "Works best with a steady, sustained 'aaah'.")
        return
    fingerprint = hashlib.sha256(sources[active]).hexdigest()
    if st.session_state.get("id_fingerprint") != fingerprint:
        try:
            with st.spinner("Identifying voice..."):
                outcome = {"result": identifier.predict_bytes(sources[active]), "error": None}
        except AudioError as error:
            outcome = {"result": None, "error": str(error)}
        except Exception as error:  # never let a model problem take the page down
            outcome = {"result": None, "error": f"Unexpected problem ({type(error).__name__}). Please try again."}
        st.session_state.update(id_fingerprint=fingerprint, id_outcome=outcome)
    outcome = st.session_state["id_outcome"]
    if outcome["error"]:
        st.error(f"Could not analyse this voice: {outcome['error']}")
    else:
        show_identity(outcome["result"])
