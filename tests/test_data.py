"""Turning patient details into model features."""
import math

import numpy as np
import pandas as pd

import data

FEATURES = ["is_inpatient", "age_min", "gender", "ward_ip", "ward_op", "adi_score",
            "abx_class_fluoroquinolone_90d", "days_since_last_antibiotic", "prior_org_escherichia",
            "prior_resist_ciprofloxacin", "n_prior_resistant_drugs", "proc_urethral_catheter",
            "nursing_home"]


class FakeModel:
    def predict(self, frame):
        assert list(frame.columns) == FEATURES and (frame.dtypes == "float64").all()
        return pd.DataFrame({"p_resistant_nitrofurantoin": [0.2], "p_resistant_meropenem": [0.01]})


def test_blank_patient_has_no_history_and_unknown_demographics():
    patient = data.blank_patient(FEATURES)
    assert list(patient) == FEATURES
    assert patient["prior_resist_ciprofloxacin"] == 0 and patient["ward_ip"] == 0
    assert patient["days_since_last_antibiotic"] == data.NO_RECENT_ANTIBIOTIC
    assert math.isnan(patient["age_min"]) and math.isnan(patient["adi_score"])


def test_options_lists_the_yes_no_features_with_readable_labels():
    assert data.options(FEATURES, "abx_class_", "_90d") == {"Fluoroquinolone": "abx_class_fluoroquinolone_90d"}
    assert data.options(FEATURES, "proc_") == {"Urethral catheter": "proc_urethral_catheter"}


def test_every_care_setting_only_sets_visit_features():
    for flags in data.SETTINGS.values():
        assert all(name == "is_inpatient" or name.startswith("ward_") for name in flags)


def test_predict_sends_float_features_in_order_and_returns_drug_slugs():
    out = data.predict(FakeModel(), FEATURES, data.blank_patient(FEATURES))
    assert list(out.index) == ["nitrofurantoin", "meropenem"] and out["meropenem"] == 0.01


def test_sex_labels_map_to_the_dataset_codes():
    assert data.SEX_CODES == {"Female": 0, "Male": 1}


class FakeDrugModel:
    def __init__(self, risk):
        self.risk = risk

    def predict_proba(self, X):
        assert X.shape == (1, len(FEATURES))
        return np.array([[1 - self.risk, self.risk]])


class PackagedModel:
    """Stands in for the registered model: predict must never be called (see data.predict)."""

    def unwrap_python_model(self):
        class Inner:
            models = {"nitrofurantoin": FakeDrugModel(0.3), "meropenem": FakeDrugModel(0.02)}
            features = FEATURES
            drugs = ["nitrofurantoin", "meropenem"]
        return Inner()

    def predict(self, frame):
        raise AssertionError("the saved predict code must not run")


def test_predict_scores_the_packaged_per_drug_models_directly():
    out = data.predict(PackagedModel(), FEATURES, data.blank_patient(FEATURES))
    assert out.to_dict() == {"nitrofurantoin": 0.3, "meropenem": 0.02}


NEW_FEATURES = ["age_min", "gender", "ward_ip", "ward_op", "ward_er", "ward_icu", "is_inpatient",
                "nursing_home", "days_since_last_antibiotic", "abx_class_fluoroquinolone_90d",
                "abx_class_beta_lactam_90d", "prior_org_escherichia", "prior_resist_ciprofloxacin",
                "n_prior_resistant_drugs", "proc_urethral_catheter",
                "days_since_drug_ciprofloxacin", "days_since_subtype_fluoroquinolone",
                "days_since_subtype_penicillin", "days_since_resist_ciprofloxacin",
                "days_since_resist_nitrofurantoin", "n_antibiotic_days_365d", "n_prior_cultures_365d",
                "comorb_diabetes", "n_comorbidities"]
RECORD = {"age_band": "75-84", "sex": "Male", "care_setting": "Hospital ward", "nursing_home": False,
          "resistant_to": ["ciprofloxacin"], "other_resistant_count": 0,
          "recent_antibiotic_classes": ["fluoroquinolone", "beta_lactam"],
          "days_since_last_antibiotic": 14, "prior_organisms": ["escherichia"],
          "recent_procedures": ["urethral_catheter"]}


def test_the_newer_models_inputs_are_filled_the_way_notebook_02_fills_them():
    blank = data.blank_patient(NEW_FEATURES)
    assert blank["days_since_drug_ciprofloxacin"] == blank["days_since_resist_nitrofurantoin"] == data.NEVER
    assert blank["n_antibiotic_days_365d"] == blank["comorb_diabetes"] == blank["n_comorbidities"] == 0

    patient = data.patient_features(NEW_FEATURES, RECORD)
    assert list(patient) == NEW_FEATURES and not any(math.isnan(v) for v in patient.values())
    assert patient["days_since_resist_ciprofloxacin"] == data.RESISTANCE_DAYS_AGO
    assert patient["days_since_resist_nitrofurantoin"] == data.NEVER          # no such resistance
    assert patient["days_since_subtype_fluoroquinolone"] == 14               # last antibiotic 14 days ago
    assert patient["days_since_subtype_penicillin"] == data.NEVER            # no matching subtype
    assert patient["days_since_drug_ciprofloxacin"] == data.NEVER            # the drug itself isn't recorded
    assert patient["n_antibiotic_days_365d"] == 1 and patient["n_prior_cultures_365d"] == 1

    none = data.patient_features(NEW_FEATURES, {**RECORD, "resistant_to": [], "prior_organisms": [],
                                               "recent_antibiotic_classes": [],
                                               "days_since_last_antibiotic": data.NO_RECENT_ANTIBIOTIC})
    assert none["n_antibiotic_days_365d"] == 0 and none["n_prior_cultures_365d"] == 0


def test_severity_comes_from_care_setting_and_vital_signs():
    import policy

    level = lambda **changes: policy.SEVERITY_LEVELS[data.severity_of({**RECORD, **changes})]   # noqa: E731
    assert level(care_setting="Outpatient clinic") == "Mild"                 # no vital signs recorded
    assert level(care_setting="Emergency department") == "Mild"
    assert level() == "Systemic signs"                                       # admitted to a ward
    assert level(care_setting="Outpatient clinic", temperature_f=100.9) == "Systemic signs"
    assert level(care_setting="Outpatient clinic", heart_rate=95, respiratory_rate=21) == "Systemic signs"
    assert level(care_setting="Intensive care unit") == "Severe"
    assert level(care_setting="Outpatient clinic", systolic_bp=88) == "Severe"
    assert level(respiratory_rate=22, systolic_bp=96) == "Severe"


def test_bars_fall_back_to_the_notebook_defaults_when_mlflow_cannot_be_read(monkeypatch):
    import mlflow
    import policy

    def unreachable(*args, **kwargs):
        raise ConnectionError("no Databricks here")

    monkeypatch.setattr(mlflow, "MlflowClient", unreachable)
    bars, source = data.severity_bars()
    assert bars == policy.DEFAULT_SEVERITY_BARS and "defaults" in source and "ConnectionError" in source


def test_bars_are_read_from_the_newest_mlflow_run_that_logged_them(monkeypatch):
    import types

    import mlflow

    run = lambda name, params: types.SimpleNamespace(                        # noqa: E731
        info=types.SimpleNamespace(run_name=name, experiment_id="7"), data=types.SimpleNamespace(params=params))

    class Client:
        def __init__(self, **kwargs): pass
        def search_model_versions(self, query): return [types.SimpleNamespace(version="3", run_id="a"),
                                                        types.SimpleNamespace(version="12", run_id="b")]
        def get_run(self, run_id): assert run_id == "b"; return run("training", {})
        def search_runs(self, experiments, order_by, max_results):
            return [run("training", {"n_features": "105"}),
                    run("recommendation_policy", {"severity_bar_mild": "0.8", "severity_bar_severe": "0.97",
                                                  "severity_bar_systemic_signs": "0.9"}),
                    run("older_policy", {"severity_bar_mild": "0.5", "severity_bar_severe": "0.5",
                                         "severity_bar_systemic_signs": "0.5"})]

    monkeypatch.setattr(mlflow, "MlflowClient", Client)
    bars, source = data.severity_bars()
    assert bars == {"Mild": 0.8, "Systemic signs": 0.9, "Severe": 0.97}
    assert "recommendation_policy" in source and "version 12" in source
