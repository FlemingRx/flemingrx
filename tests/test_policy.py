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


def test_severity_levels_and_cut_offs_match_notebook_04():
    """The app must sort patients into severity levels exactly as notebook 04 did when it chose the bars."""
    import re

    source = NOTEBOOK_04.read_text(encoding="utf-8")
    number = lambda name: float(re.search(rf"^{name} = ([\d.]+)", source, re.M).group(1))   # noqa: E731
    cut_offs = ["FEVER_F", "FAST_HEART", "FAST_BREATHING", "VERY_FAST_BREATHING", "LOW_NORMAL_SBP",
                "LOW_SBP"]
    for name in cut_offs:
        assert getattr(policy, name) == number(name), name
    levels = re.search(r"^SEVERITY_LEVELS = (\[.*\])", source, re.M).group(1)
    bars = re.search(r"^SEVERITY_BARS = (\[.*\])", source, re.M).group(1)
    assert policy.SEVERITY_LEVELS == eval(levels)
    assert list(policy.DEFAULT_SEVERITY_BARS.values()) == eval(bars)      # the fallback only
    assert policy.severity_bar_param("Systemic signs") == "severity_bar_systemic_signs"

    cells = source.split("# COMMAND ----------")
    building_blocks = next(c for c in cells if "def severity_level(" in c)
    notebook = {"NARROW_DRUGS": policy.NARROW_DRUGS, "BROAD_DRUGS": policy.BROAD_DRUGS,
                "CARBAPENEM_DRUGS": ["meropenem"], **{name: number(name) for name in cut_offs}}
    exec(building_blocks, notebook)

    rng = np.random.default_rng(0)
    n = 3000
    flags = [rng.integers(0, 2, n).astype(float) for _ in range(3)]            # icu, ward_ip, ward_er
    vitals = [rng.uniform(96, 104, n), rng.uniform(50, 130, n), rng.uniform(10, 30, n),
              rng.uniform(70, 150, n)]
    for column in vitals:                                 # some vital signs are not recorded
        column[rng.random(n) < 0.3] = np.nan
    expected = notebook["severity_level"](*flags, *vitals)
    clean = lambda value: None if np.isnan(value) else float(value)            # noqa: E731
    ours = [policy.severity_level(flags[0][i], flags[1][i], flags[2][i],
                                  *[clean(column[i]) for column in vitals]) for i in range(n)]
    assert ours == list(expected)
    assert set(ours) == {0, 1, 2}
