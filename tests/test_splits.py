import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold

pytest.importorskip("pandas", exc_type=ImportError)  # the training data helpers use pandas, which some Windows setups block
from pdscreen import config, datasets  # noqa: E402


@pytest.mark.skipif(not (config.FIGSHARE_DIR / "Demographics_age_sex.xlsx").exists(), reason="dataset not downloaded")
def test_holdout_split_is_subject_disjoint_and_balanced():
    manifest = datasets.figshare_manifest()
    test = datasets.holdout_split(manifest)
    train_ids, test_ids = set(manifest.subject_id[~test]), set(manifest.subject_id[test])
    assert not train_ids & test_ids
    assert 15 <= len(test_ids) <= 18
    assert set(manifest.label[test]) == {0, 1}


@pytest.mark.skipif(not (config.UCI_DIR / "parkinsons.data").exists(), reason="dataset not downloaded")
def test_grouped_folds_never_share_a_person():
    """UCI has ~6 recordings per person: grouped folds must keep each person in one fold."""
    frame = datasets.uci_parkinsons()
    groups, y = frame.subject_id.to_numpy(), frame.status.to_numpy()
    assert frame.groupby("subject_id").size().min() > 1
    for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups):
        assert not set(groups[train]) & set(groups[test])
