from pathlib import Path

import pytest

from shield_vio.datasets.splits import PaperSplit, SequenceIdentity, SplitEntry, load_paper_split


def test_frozen_euroc_partitions_are_sequence_disjoint() -> None:
    split = load_paper_split(Path("configs/paper/public_dataset_splits.yaml"))
    partitions = [set(split.partition(name)) for name in ("train", "calibration", "validation", "test", "shifted_test")]
    for i, left in enumerate(partitions):
        for right in partitions[i + 1 :]:
            assert left.isdisjoint(right)


def test_same_physical_sequence_cannot_cross_partitions() -> None:
    identity = SequenceIdentity("euroc", "MH_01_easy")
    with pytest.raises(ValueError, match="appears in both"):
        PaperSplit((SplitEntry(identity, "train"), SplitEntry(identity, "test")))


def test_derived_condition_inherits_parent_sequence_split() -> None:
    split = load_paper_split(Path("configs/paper/public_dataset_splits.yaml"))
    split.assert_run_membership("euroc", "MH_01_easy", "train")
    with pytest.raises(ValueError, match="inherits physical-sequence split"):
        split.assert_run_membership("euroc", "MH_01_easy", "test")


def test_unknown_sequence_is_rejected_instead_of_randomly_split() -> None:
    split = load_paper_split(Path("configs/paper/public_dataset_splits.yaml"))
    with pytest.raises(KeyError, match="unregistered physical sequence"):
        split.split_for("euroc", "made_up")
