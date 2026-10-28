# Results

All numbers are **person-level** (one prediction per person, the median over that person's 1.5 s windows). Cross-validation (CV) is 5-fold, stratified and grouped by person, on 64 development people. The test set is 17 people held out from the start and used once. 'Perturbed' means the same people's audio after a formant-preserving pitch change of ±2 or ±4 semitones, or a simulated laptop microphone (16 kHz, room reverberation, background noise, different loudness and frequency response).

## 1. Why the first model failed on live recordings

The first deployed model (Random Forest on 50 features of the whole recording) was applied to the 17 held-out people:

| Audio | Healthy people flagged as PD | Healthy people cleared | PD people flagged |
|---|---|---|---|
| Clean dataset recordings | 56% | 44% | 75% |
| Pitch changed ±2/±4 semitones | 36% | 64% | 69% |
| Simulated laptop microphone | 59% | 41% | 62% |

It already flagged most unseen healthy people as PD on clean dataset audio, so the problem was not only the microphone: the model had memorised its training people (training AUC 1.0) and separated unseen people poorly. It also relied heavily on MFCC standard deviations, which carry the recording channel, and on absolute pitch. Uploading a dataset recording the model had trained on looked correct for that reason.

## 2. The rebuilt model

Changes: (1) features from fixed 1.5 s windows, median over windows, so recording length and onset/offset do not matter; (2) the option to drop absolute-pitch features; (3) training on pitch-shifted and microphone-simulated copies of each training person (kept in the same fold as the person); (4) model and feature set chosen on clean AND perturbed out-of-fold performance; (5) the decision threshold chosen on out-of-fold data to limit false alarms on healthy people.

### Out-of-fold comparison (64 development people)

| Model | Features | Training | Clean AUC | Perturbed AUC | Perturbed sensitivity | Perturbed specificity |
|---|---|---|---|---|---|---|
| RF **(deployed)** | all 50 | augmented | 0.654 | 0.659 | 0.75 | 0.51 |
| RF | pitch-invariant 46 | augmented | 0.647 | 0.646 | 0.75 | 0.49 |
| LR | all 50 | plain | 0.687 | 0.594 | 0.47 | 0.68 |
| RF | all 50 | plain | 0.642 | 0.629 | 0.67 | 0.58 |
| LR | pitch-invariant 46 | plain | 0.693 | 0.576 | 0.51 | 0.63 |
| RF | pitch-invariant 46 | plain | 0.653 | 0.610 | 0.64 | 0.58 |
| LR | all 50 | augmented | 0.647 | 0.612 | 0.79 | 0.38 |
| LR | pitch-invariant 46 | augmented | 0.643 | 0.584 | 0.80 | 0.34 |
| LR | clinical, pitch-invariant 20 | augmented | 0.567 | 0.591 | 0.79 | 0.28 |
| LR | pitch-invariant, no MFCC std 33 | plain | 0.567 | 0.509 | 0.49 | 0.57 |
| RF | clinical, pitch-invariant 20 | augmented | 0.527 | 0.485 | 0.83 | 0.12 |
| LR | clinical, pitch-invariant 20 | plain | 0.511 | 0.484 | 0.49 | 0.46 |
| LR | pitch-invariant, no MFCC std 33 | augmented | 0.476 | 0.499 | 0.71 | 0.23 |
| RF | pitch-invariant, no MFCC std 33 | augmented | 0.436 | 0.471 | 0.78 | 0.24 |
| RF | pitch-invariant, no MFCC std 33 | plain | 0.422 | 0.445 | 0.68 | 0.24 |
| RF | clinical, pitch-invariant 20 | plain | 0.431 | 0.426 | 0.84 | 0.12 |

**Deployed:** RF | all 50 | augmented. Decision threshold 0.62 (lowest threshold keeping out-of-fold specificity >= 80%).

### At the deployed threshold

| Set | Audio | AUC | Sensitivity | Specificity | Balanced accuracy |
|---|---|---|---|---|---|
| Cross-validation (64) | clean | 0.65 | 0.34 | 0.81 | 0.58 |
| Cross-validation (64) | perturbed | 0.66 | 0.32 | 0.84 | 0.58 |
| Held-out test (17) | clean | 0.65 | 0.62 | 0.78 | 0.70 |
| Held-out test (17) | perturbed | 0.61 | 0.45 | 0.76 | 0.60 |

Held-out test, clean recordings: AUC 0.65 (95% CI 0.35-0.91), 5 of 8 people with PD flagged, 7 of 9 healthy people cleared. With 17 people the interval is wide.

## 3. Overfitting and underfitting check (deployed model)

- Training AUC (people it was trained on): 0.98
- Cross-validated AUC, clean: 0.65; perturbed: 0.66
- Held-out test AUC, clean: 0.65; perturbed: 0.61
- Gap between training and cross-validated AUC: 0.33. A large gap means the model still fits its training people better than new people (overfitting), which is expected with 64 training people; regularisation (shallow trees or strong L2), augmentation and person-wise selection limit it but cannot remove it.
- Underfitting would show as low training AUC; that is not the case here. The limiting factor is the amount of data, not model capacity.

## 4. Confound check: age

- An age-only logistic regression reaches CV AUC 0.85, because the healthy group is much younger (mean 48 vs 67 years).
- Among people aged 50 or over (30 PD, 14 healthy), the deployed model's out-of-fold AUC is 0.59 (sensitivity 0.33, specificity 0.86).
- Treat the voice model's skill against that baseline: part of its signal may be age.

## 5. Spectrogram CNN (explanation aid, not part of the score)

CV AUC 0.68 ± 0.17 over folds (pooled out-of-fold 0.59); held-out test AUC 0.92. The cross-validated and test numbers disagree because both sets are small. It is used for the Grad-CAM heatmap and shown as a second opinion that does not change the final result.

## 6. Leakage demonstration (UCI Parkinson's, 32 people, about 6 recordings each)

| Model | Accuracy, split by recording | Accuracy, split by person | AUC, split by recording | AUC, split by person |
|---|---|---|---|---|
| Logistic regression | 0.782 | 0.702 | 0.895 | 0.782 |
| SVM (RBF) | 0.876 | 0.743 | 0.933 | 0.668 |
| Random forest | 0.905 | 0.767 | 0.953 | 0.737 |
| XGBoost | 0.899 | 0.755 | 0.948 | 0.752 |

Splitting by recording lets a model recognise people it has already seen, which inflates every metric. All deployed-model results above use person-wise splits.

## 7. Explainability

Top features by mean |SHAP| on the deployed model's calibrated P(PD):

- `mfcc12_std`: mfcc12_std (0.018)
- `mfcc4_mean`: mfcc4_mean (0.017)
- `mfcc13_std`: mfcc13_std (0.009)
- `f0_max_hz`: Highest pitch (Hz) (0.009)
- `f0_mean_hz`: Average pitch (Hz) (0.009)
- `mfcc11_std`: mfcc11_std (0.007)
- `mfcc10_mean`: mfcc10_mean (0.007)
- `spectral_centroid_hz`: Spectral brightness (Hz) (0.005)
- `mfcc3_mean`: mfcc3_mean (0.004)
- `f0_min_hz`: Lowest pitch (Hz) (0.004)

## 8. Confidence levels (out-of-fold, clean recordings)

| Confidence | People | Accuracy |
|---|---|---|
| High | 6 | 0.83 |
| Moderate | 27 | 0.59 |
| Low | 31 | 0.52 |

## Figures

- `figures/calibration_cnn_vs_features.png`
- `figures/calibration_feature_model.png`
- `figures/cnn_learning_curve.png`
- `figures/confusion_cnn_test.png`
- `figures/confusion_deployed_test.png`
- `figures/confusion_feature_model_cv.png`
- `figures/confusion_feature_model_test.png`
- `figures/gradcam_examples.png`
- `figures/roc_cv_cnn_vs_features.png`
- `figures/roc_cv_feature_models.png`
- `figures/roc_robust_model.png`
- `figures/roc_test.png`
- `figures/shap_summary.png`
