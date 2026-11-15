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
