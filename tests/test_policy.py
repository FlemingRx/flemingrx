"""The app's recommendation rule, including a check that it matches notebook 04."""
import pathlib

import numpy as np

import policy

DRUGS = list(policy.DRUG_LABELS)
NOTEBOOK_04 = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "04_recommend.py"


def test_picks_the_narrowest_drug_that_clears_the_bar():
    risks = [0.30, 0.25, 0.08, 0.05, 0.04, 0.02, 0.01]
    assert policy.recommend(risks, 0.90) == 2


def test_falls_back_to_the_broadest_drug_when_none_clears_the_bar():
    assert policy.recommend([0.5] * 7, 0.90) == 6


def test_a_drug_exactly_on_the_bar_counts():
    assert policy.recommend([0.25, 0.5], 0.75) == 0


def test_matches_notebook_04():
    """The app must apply the same rule that notebook 04 evaluated."""
    cells = NOTEBOOK_04.read_text(encoding="utf-8").split("# COMMAND ----------")
    building_blocks = next(c for c in cells if "def recommend(P, max_risk)" in c)
    notebook = {"NARROW_DRUGS": policy.NARROW_DRUGS, "BROAD_DRUGS": policy.BROAD_DRUGS}
    exec(building_blocks, notebook)

    P = np.random.default_rng(0).random((2000, len(DRUGS))) ** 3     # mostly low risks, like real ones
    for target in np.round(np.arange(0.50, 1.0, 0.01), 2):
        expected = notebook["recommend"](P, 1 - target)
        ours = [policy.recommend(list(row), target) for row in P]
        assert ours == list(expected), f"different picks at target {target}"


def test_spectrum_groups_match_notebook_04():
    source = NOTEBOOK_04.read_text(encoding="utf-8")
    assert f"NARROW_DRUGS = {policy.NARROW_DRUGS!r}".replace("'", '"') in source
    assert f"BROAD_DRUGS = {policy.BROAD_DRUGS!r}".replace("'", '"') in source


def test_explain_names_the_pick_and_the_drugs_it_skipped():
    risks = [0.30, 0.25, 0.08, 0.05, 0.04, 0.02, 0.01]
    text = policy.explain(risks, DRUGS, 0.90)
    assert "Cefazolin" in text and "Nitrofurantoin" in text and "92%" in text
    assert "No antibiotic reaches" in policy.explain([0.5] * 7, DRUGS, 0.90)
