# Demo video script (about 2 minutes)

Record the screen with **Win + Alt + R** (Xbox Game Bar) or OBS Studio. Turn on microphone capture so
the "aaah" is audible in the video.

| Time | Show | Say |
|---|---|---|
| 0:00-0:15 | README results table | "This is a voice-based Parkinson's screening aid. It is trained and tested by person, not by recording, so the same person is never in both training and test data." |
| 0:15-0:30 | Terminal: `streamlit run app/streamlit_app.py` | "The app loads two models: a feature model on jitter, shimmer, HNR, PPE, RPDE and DFA, and a small CNN on log-Mel spectrograms exported to ONNX." |
| 0:30-0:55 | Click the microphone, say "aaah" for 6-8 seconds, stop | "I record a sustained vowel. The app denoises it, trims silence, extracts the features and predicts in under two seconds." |
| 0:55-1:15 | Result card, probability, confidence, latency | "The result is a probability with a confidence level, and the disclaimer: screening aid only, consult a neurologist." |
| 1:15-1:35 | "Why this result" tab | "SHAP shows which voice measurements pushed the score up or down; Grad-CAM shows where the CNN looked in the spectrogram." |
| 1:35-1:50 | Expander "Or upload a WAV file": upload a held-out PD recording from `data/raw/figshare_iyer2023/PD_AH` | "A recording from a held-out person with Parkinson's, never seen in training." |
| 1:50-2:00 | "About the model" tab | "Test-set results and limitations, including the age difference between groups." |

Tip: for the held-out examples, use subject IDs listed in `reports/test_predictions_all.csv`.
