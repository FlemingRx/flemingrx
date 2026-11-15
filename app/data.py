"""Loading the model and tables, and turning patient details into the features the model expects.

Everything downstream works from one thing: a `patient` dict of {feature name: number}.
`patient_features` builds it from a patient's row in the database (db.py). Reading patients from
a hospital's records instead means filling that table from them; the model call, the
recommendation rule and the screens don't change.

On Databricks the model comes from Unity Catalog and the tables from a SQL warehouse. For local
development both can be replaced with files (see app/README.md):
  MODEL_URI       a saved MLflow model folder instead of the registered model
  LOCAL_DATA_DIR  a folder holding <table>.parquet instead of the warehouse
If neither is set and the repo has a local copy in data/app_local/ (git-ignored), that is used.
"""

import math
import os
import pathlib

import pandas as pd

CATALOG = os.getenv("FLEMINGRX_CATALOG", "workspace")
SCHEMA = os.getenv("FLEMINGRX_SCHEMA", "armd")
MODEL_NAME = os.getenv("MODEL_NAME", f"{CATALOG}.{SCHEMA}.urine_resistance_model")

NO_RECENT_ANTIBIOTIC = 366   # notebook 02's value for "none in the last year"

# A local copy of the model and tables, for running the app on a laptop without Databricks.
# It lives outside this folder, so it is never there in a deployed Databricks App.
_LOCAL_COPY = pathlib.Path(__file__).resolve().parents[1] / "data" / "app_local"

# What each care setting means for the visit features (the combinations found in the data).
SETTINGS = {
    "Outpatient clinic": {"is_inpatient": 0, "ward_op": 1},
    "Emergency department": {"is_inpatient": 1, "ward_ip": 1, "ward_er": 1},
    "Hospital ward": {"is_inpatient": 1, "ward_ip": 1},
    "Intensive care unit": {"is_inpatient": 1, "ward_ip": 1, "ward_icu": 1},
}

# The dataset codes gender as 0/1 and deliberately doesn't say which is which. We read code 0 as
# female: it is about 80% of urine cultures but only about half of blood and respiratory cultures,
# which matches urinary infections being far more common in women. This is an inference, not
# something the dataset documents.
SEX_CODES = {"Female": 0, "Male": 1}

# The dataset gives age in bands and keeps the lower end.
AGE_BANDS = {"18-24": 18, "25-34": 25, "35-44": 35, "45-54": 45, "55-64": 55, "65-74": 65,
             "75-84": 75, "85-89": 85, "90 or older": 90}

_ZERO_PREFIXES = ("abx_class_", "prior_org_", "prior_resist_", "proc_", "ward_")
_ZERO_NAMES = {"nursing_home", "n_prior_resistant_drugs", "is_inpatient"}


# ----------------------------------------------------------------------------- loading

def local_model_uri():
    """The model folder to load instead of the registered model, or None to use Unity Catalog."""
    if os.getenv("MODEL_URI"):
        return os.getenv("MODEL_URI")
    return str(_LOCAL_COPY / "model") if (_LOCAL_COPY / "model" / "MLmodel").exists() else None


def local_data_dir():
    """The folder of <table>.parquet files to read instead of the warehouse, or None."""
    if os.getenv("LOCAL_DATA_DIR"):
        return os.getenv("LOCAL_DATA_DIR")
    return str(_LOCAL_COPY / "tables") if (_LOCAL_COPY / "tables").is_dir() else None


def model_source() -> str:
    """Where the model comes from, for showing in the app."""
    uri = local_model_uri()
    return f"local copy ({uri})" if uri else f"Unity Catalog ({MODEL_NAME})"


def load_model():
    """The registered model: patient features in, one `p_resistant_<drug>` column per antibiotic out."""
    import mlflow

    uri = local_model_uri()
    if not uri:
        mlflow.set_registry_uri("databricks-uc")
        versions = mlflow.MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
        latest = max(int(v.version) for v in versions)
        uri = f"models:/{MODEL_NAME}/{latest}"
    return mlflow.pyfunc.load_model(uri)


def model_features(model) -> list:
    """The feature columns the model was trained on, in order."""
    return list(model.metadata.get_input_schema().input_names())


def load_table(name: str) -> pd.DataFrame:
    """A whole table from the project's schema (they are all small)."""
    local_dir = local_data_dir()
    if local_dir:
        return pd.read_parquet(pathlib.Path(local_dir) / f"{name}.parquet")

    import certifi
    from databricks import sql
    from databricks.sdk.core import Config

    warehouse = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not set, so there is no SQL warehouse to ask.")
    config = Config()
    with sql.connect(
        server_hostname=config.host,
        http_path=f"/sql/1.0/warehouses/{warehouse}",
        credentials_provider=lambda: config.authenticate,
        # Python installed from python.org on a Mac has no trusted certificates of its own, and the
        # connection is then refused. certifi's bundle is always there (it comes with requests).
        _tls_trusted_ca_file=certifi.where(),
        # Give up after a few tries instead of the default 30, so a network problem shows up as
        # a message within seconds rather than a page that hangs for minutes.
        _retry_stop_after_attempts_count=4,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {CATALOG}.{SCHEMA}.{name}")
            return cursor.fetchall_arrow().to_pandas()


# ----------------------------------------------------------------------------- features

def blank_patient(features) -> dict:
    """A patient with no history: every flag off, no recent antibiotic, everything else unknown."""
    patient = {}
    for name in features:
        if name.startswith(_ZERO_PREFIXES) or name in _ZERO_NAMES:
            patient[name] = 0.0
        elif name == "days_since_last_antibiotic":
            patient[name] = float(NO_RECENT_ANTIBIOTIC)
        else:
            patient[name] = math.nan        # age, sex, deprivation index: unknown until filled in
    return patient


def patient_features(features, record: dict) -> dict:
    """The model's input for one patient row from the database (see db.patients for the columns)."""
    patient = blank_patient(features)
    patient["age_min"] = float(AGE_BANDS[record["age_band"]])
    patient["gender"] = float(SEX_CODES[record["sex"]])
    patient.update({name: float(value) for name, value in SETTINGS[record["care_setting"]].items()})
    patient["nursing_home"] = float(record["nursing_home"])
    patient["n_prior_resistant_drugs"] = float(len(record["resistant_to"])
                                               + record["other_resistant_count"])
    patient["days_since_last_antibiotic"] = float(record["days_since_last_antibiotic"])
    flags = ([f"prior_resist_{drug}" for drug in record["resistant_to"]]
             + [f"abx_class_{name}_90d" for name in record["recent_antibiotic_classes"]]
             + [f"prior_org_{name}" for name in record["prior_organisms"]]
             + [f"proc_{name}" for name in record["recent_procedures"]])
    for name in flags:
        if name in patient:      # the model has no feature for anything else, so it can't use it
            patient[name] = 1.0
    return {name: patient[name] for name in features}


def options(features, prefix: str, suffix: str = "") -> dict:
    """{label shown in the form: feature name} for the yes/no features that start with `prefix`."""
    names = [f for f in features if f.startswith(prefix) and f.endswith(suffix)]
    return {pretty(f[len(prefix):len(f) - len(suffix)]): f for f in names}


def pretty(text: str) -> str:
    special = {"cons": "Coagulase-negative staph", "cvc": "Central venous catheter",
               "mechvent": "Mechanical ventilation"}
    return special.get(text, text.replace("_", " ").capitalize())


def _per_drug_models(model):
    """The object notebook 03 packaged (per-drug models, feature order, drug order), or None."""
    try:
        inner = model.unwrap_python_model()
    except Exception:       # not a model packaged the way notebook 03 does it
        return None
    return inner if all(hasattr(inner, name) for name in ("models", "features", "drugs")) else None


def predict(model, features, patient: dict) -> pd.Series:
    """Predicted chance of resistance per drug slug, narrowest drug first."""
    row = pd.DataFrame([{name: patient.get(name, math.nan) for name in features}]).astype("float64")

    # The registered model wraps 7 scikit-learn models in a small class from notebook 03. MLflow
    # saves that class as compiled code for the exact Python version of the notebook, and running
    # it under a different Python version crashes the whole process. Its predict method only
    # scores each drug's model in turn, so we do that here and never run the saved code. Only the
    # scikit-learn version then has to match the one the model was trained with.
    packaged = _per_drug_models(model)
    if packaged is not None:
        X = row[packaged.features].to_numpy()
        return pd.Series({drug: float(packaged.models[drug].predict_proba(X)[0, 1])
                          for drug in packaged.drugs})

    out = model.predict(row).iloc[0]
    return out.rename(lambda column: column.removeprefix("p_resistant_"))
