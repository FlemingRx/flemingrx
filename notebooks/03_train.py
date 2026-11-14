# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Train: predict resistance for each antibiotic
# MAGIC
# MAGIC **What this notebook does:** trains models that take a patient's information (the features from
# MAGIC notebook 02) and estimate, for each of the 7 antibiotics, the **chance the infection is resistant**.
# MAGIC
# MAGIC **We compare four approaches, from simplest to most advanced:**
# MAGIC
# MAGIC | Model | What it does | Why it's here |
# MAGIC |---|---|---|
# MAGIC | **Antibiogram** | Gives every patient the hospital's overall resistance rate for that drug | What hospitals use today. The model must beat this |
# MAGIC | **Prior-resistance rule** | Two rates: one for patients resistant to this drug in the past year, one for everyone else | The obvious rule a careful doctor would apply. Shows whether ML adds anything beyond it |
# MAGIC | **Logistic regression** | A weighted sum of all features | Simple and explainable |
# MAGIC | **Gradient boosting** | Hundreds of small decision trees added together | Usually the most accurate on tables like ours; your paper found tree-based models did best |
# MAGIC
# MAGIC **One model per antibiotic.** Each drug gets its own model, because what predicts nitrofurantoin
# MAGIC resistance isn't the same as what predicts ceftriaxone resistance.
# MAGIC
# MAGIC **How the data is split (by date, never shuffled):**
# MAGIC - **Step A, choose the approach:** train on 2008-2020, compare on **2021**.
# MAGIC - **Step B, final score:** retrain the chosen approach on 2008-2021 and score it once on
# MAGIC   **2022 onwards**, the test years nobody has looked at. This is the number for the pitch.
# MAGIC
# MAGIC **How we score models** (on cultures the model never saw):
# MAGIC - **AUROC** (0.5-1.0): pick one resistant and one non-resistant culture at random; how often does
# MAGIC   the model give the resistant one the higher risk? 0.5 = coin flip, 1.0 = perfect.
# MAGIC - **AUPRC** (0-1): how well the model finds the resistant cultures. A random guess scores the
# MAGIC   resistance rate (e.g. 0.21 for a drug with 21% resistance), so compare against that.
# MAGIC - **Brier score** (lower is better): the average squared gap between the predicted chance and
# MAGIC   what happened. It rewards probabilities that are *honest*, which matters because notebook 04
# MAGIC   uses them as "90% likely to work".
# MAGIC
# MAGIC **Output:**
# MAGIC - Every run logged in **MLflow** (scores, settings, the models).
# MAGIC - The final model registered in **Unity Catalog** as `workspace.armd.urine_resistance_model`.
# MAGIC - Test-year predictions saved to `workspace.armd.gold_test_predictions` for notebook 04.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "armd"


def tbl(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"


VALIDATION_YEAR = 2021   # Step A: train on years before this, compare on this year
SEED = 42                # makes the gradient-boosting results repeatable

# Labs and vital signs are left OUT by default, for two reasons:
#  1. The dataset doesn't say whether the 14-day lab window can include days after the culture was
#     ordered, so they might leak future information.
#  2. Most outpatients have none (58% of rows have no labs), and the demo is simpler without them.
# Section 4 measures how much they would add, so the team can make an informed call.
USE_LABS_VITALS = False

MODEL_NAME = tbl("urine_resistance_model")
EXPERIMENT_NAME = "flemingrx_urine_resistance"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The building blocks
# MAGIC These functions do all the modelling. They only use numpy and scikit-learn (no Spark), so the
# MAGIC same code can be tested on a laptop.

# COMMAND ----------

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASELINES = ["antibiogram", "prior_resistance_rule"]
ML_MODELS = ["logistic_regression", "gradient_boosting"]

# Columns that must never be model inputs
NOT_FEATURES = {
    "order_proc_id_coded", "anon_id", "order_time", "ordering_mode", "split",
    "antibiotic", "antibiotic_slug", "spectrum_rank",
    "resistant",                 # the answer
    "n_organisms_tested",        # comes from the lab result -> would leak the answer
    "order_year",                # the model must work on future years it has never seen
    "prior_resist_same_drug",    # duplicate of prior_resist_<drug> once we have one model per drug
}


def select_features(columns, use_labs_vitals):
    return [c for c in columns
            if c not in NOT_FEATURES and (use_labs_vitals or not c.startswith(("lab_", "vital_")))]


def make_model(kind):
    if kind == "logistic_regression":
        # fill missing values with the median (and flag that they were missing), put every feature
        # on the same scale, then fit the weighted sum
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        )
    if kind == "gradient_boosting":
        # handles missing values by itself; stops adding trees once a held-out 10% stops improving
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, random_state=SEED,
        )
    raise ValueError(kind)


def fit_and_predict(kind, X_train, y_train, X_eval, prior_col):
    """Train one approach and return (fitted model or None, predicted chance of resistance)."""
    if kind == "antibiogram":
        return None, np.full(len(X_eval), y_train.mean())
    if kind == "prior_resistance_rule":
        had = X_train[:, prior_col] == 1
        rate_had = y_train[had].mean() if had.any() else y_train.mean()
        rate_not = y_train[~had].mean()
        return None, np.where(X_eval[:, prior_col] == 1, rate_had, rate_not)
    model = make_model(kind).fit(X_train, y_train)
    return model, model.predict_proba(X_eval)[:, 1]


def evaluate(y, p):
    scores = {"cultures": int(len(y)), "pct_resistant": round(100 * float(y.mean()), 1)}
    if len(np.unique(y)) < 2:   # can't rank if every culture has the same answer
        return {**scores, "auroc": float("nan"), "auprc": float("nan"),
                "brier": round(float(brier_score_loss(y, p)), 4)}
    return {**scores,
            "auroc": round(float(roc_auc_score(y, p)), 3),
            "auprc": round(float(average_precision_score(y, p)), 3),
            "brier": round(float(brier_score_loss(y, p)), 4)}


def compare_models(X, y, drug, features, drugs_in_order, train_mask, eval_mask, kinds):
    """Train every approach in `kinds` for every drug. Returns (score rows, predictions)."""
    rows, predictions = [], {}
    for d in drugs_in_order:
        train = (drug == d) & train_mask
        test = (drug == d) & eval_mask
        prior_col = features.index(f"prior_resist_{d}")
        for kind in kinds:
            model, p = fit_and_predict(kind, X[train], y[train], X[test], prior_col)
            rows.append({"antibiotic": d, "model": kind, **evaluate(y[test], p)})
            predictions[(d, kind)] = {"rows": np.flatnonzero(test), "p": p, "model": model}
    return rows, predictions


def average_by_model(rows):
    """Mean score per approach across the drugs (meropenem is skipped where it can't be scored)."""
    out = []
    for kind in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == kind]
        out.append({"model": kind,
                    "mean_auroc": round(float(np.nanmean([r["auroc"] for r in mine])), 3),
                    "mean_auprc": round(float(np.nanmean([r["auprc"] for r in mine])), 3),
                    "mean_brier": round(float(np.mean([r["brier"] for r in mine])), 4)})
    return out


def calibration_table(y, p, bins=10):
    """Group cultures by predicted chance and compare with how often they were really resistant."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return [{"bin": int(b), "cultures": int((which == b).sum()),
             "mean_predicted_pct": round(100 * float(p[which == b].mean()), 1),
             "actual_resistant_pct": round(100 * float(y[which == b].mean()), 1)}
            for b in range(len(edges) - 1) if (which == b).any()]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load the training table
# MAGIC About 505,000 rows: small enough to work with in memory. Features become one numeric table `X`,
# MAGIC with missing values as NaN.

# COMMAND ----------

import pandas as pd

pdf = spark.table(tbl("gold_urine_features")).toPandas()

FEATURES = select_features(pdf.columns, USE_LABS_VITALS)
FEATURES_WITH_LABS = select_features(pdf.columns, True)

X = pdf[FEATURES].astype("float64").to_numpy()
X_with_labs = pdf[FEATURES_WITH_LABS].astype("float64").to_numpy()
y = pdf["resistant"].astype(int).to_numpy()
drug = pdf["antibiotic_slug"].to_numpy()
year = pdf["order_year"].astype(int).to_numpy()
is_test = (pdf["split"] == "test").to_numpy()

drug_names = (pdf[["antibiotic_slug", "antibiotic", "spectrum_rank"]].drop_duplicates()
              .sort_values("spectrum_rank"))
DRUGS = list(drug_names["antibiotic_slug"])
DRUG_LABEL = dict(zip(drug_names["antibiotic_slug"], drug_names["antibiotic"]))

print(f"{len(pdf):,} rows, {len(FEATURES)} features (labs/vitals {'in' if USE_LABS_VITALS else 'out'})")
print("Drugs:", ", ".join(DRUG_LABEL[d] for d in DRUGS))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Step A: choose the approach (train 2008-2020, compare on 2021)
# MAGIC The test years (2022+) are not touched here.

# COMMAND ----------

step_a_train = (~is_test) & (year < VALIDATION_YEAR)
step_a_eval = (~is_test) & (year == VALIDATION_YEAR)

val_rows, _ = compare_models(X, y, drug, FEATURES, DRUGS, step_a_train, step_a_eval,
                             BASELINES + ML_MODELS)
display(pd.DataFrame(val_rows))
display(pd.DataFrame(average_by_model(val_rows)))

BEST_MODEL = max((r for r in average_by_model(val_rows) if r["model"] in ML_MODELS),
                 key=lambda r: r["mean_auroc"])["model"]
print(f"Chosen approach (highest average AUROC on 2021): {BEST_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Do labs and vital signs help?
# MAGIC Same comparison for gradient boosting, with and without labs and vitals. If the gain is tiny,
# MAGIC leaving them out costs nothing and avoids the leakage question. If it's large, discuss it as a
# MAGIC team before switching `USE_LABS_VITALS` on: a big jump from lab values can itself be a sign of
# MAGIC leakage.

# COMMAND ----------

with_rows, _ = compare_models(X_with_labs, y, drug, FEATURES_WITH_LABS, DRUGS, step_a_train,
                              step_a_eval, ["gradient_boosting"])
without = {r["antibiotic"]: r["auroc"] for r in val_rows if r["model"] == "gradient_boosting"}
display(pd.DataFrame([
    {"antibiotic": r["antibiotic"], "auroc_without_labs": without[r["antibiotic"]],
     "auroc_with_labs": r["auroc"], "gain": round(r["auroc"] - without[r["antibiotic"]], 3)}
    for r in with_rows
]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step B: final score on the test years (train 2008-2021, test 2022+)
# MAGIC Every approach is retrained on all training years and scored once on the test years. These are
# MAGIC the numbers to report. Everything is logged to MLflow.

# COMMAND ----------

import mlflow

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/{EXPERIMENT_NAME}")

test_rows, test_predictions = compare_models(X, y, drug, FEATURES, DRUGS, ~is_test, is_test,
                                             BASELINES + ML_MODELS)
test_table = pd.DataFrame(test_rows)
summary_table = pd.DataFrame(average_by_model(test_rows))

with mlflow.start_run(run_name=f"final_{BEST_MODEL}") as run:
    mlflow.log_params({
        "chosen_model": BEST_MODEL, "use_labs_vitals": USE_LABS_VITALS,
        "n_features": len(FEATURES), "train_years": f"2008-{VALIDATION_YEAR}", "test_years": "2022+",
    })
    for r in test_rows:
        for metric in ("auroc", "auprc", "brier"):
            if not np.isnan(r[metric]):
                mlflow.log_metric(f"{r['model']}.{r['antibiotic']}.{metric}", r[metric])
    for r in average_by_model(test_rows):
        mlflow.log_metric(f"{r['model']}.mean_auroc", r["mean_auroc"])
    mlflow.log_table(test_table, artifact_file="test_scores.json")
    mlflow.log_table(pd.DataFrame(val_rows), artifact_file="validation_2021_scores.json")
    RUN_ID = run.info.run_id

display(test_table)
display(summary_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Are the probabilities honest? (calibration)
# MAGIC Cultures are grouped into 10 bins by the chosen model's predicted chance of resistance. In each
# MAGIC bin, **mean_predicted_pct** should be close to **actual_resistant_pct**. If the model says 30%,
# MAGIC about 30% of those cultures should really be resistant. Notebook 04 relies on this.

# COMMAND ----------

calibration_rows = []
for d in DRUGS:
    pred = test_predictions[(d, BEST_MODEL)]
    for row in calibration_table(y[pred["rows"]], pred["p"]):
        calibration_rows.append({"antibiotic": d, **row})
display(pd.DataFrame(calibration_rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save the test-year predictions for notebook 04
# MAGIC One row per (test culture, antibiotic): the true answer, the chosen model's predicted chance of
# MAGIC resistance, and the antibiogram's (the baseline notebook 04 compares against).

# COMMAND ----------

pred_frames = []
for d in DRUGS:
    model_pred = test_predictions[(d, BEST_MODEL)]
    base_pred = test_predictions[(d, "antibiogram")]
    part = pdf.iloc[model_pred["rows"]][
        ["order_proc_id_coded", "anon_id", "order_year", "antibiotic", "antibiotic_slug",
         "spectrum_rank", "resistant"]].copy()
    part["p_resistant"] = model_pred["p"]
    part["p_resistant_antibiogram"] = base_pred["p"]
    pred_frames.append(part)

predictions_pdf = pd.concat(pred_frames, ignore_index=True)
(
    spark.createDataFrame(predictions_pdf)
    .write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(tbl("gold_test_predictions"))
)
print(f"Saved {len(predictions_pdf):,} rows to {tbl('gold_test_predictions')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Package the 7 models as one model and register it in Unity Catalog
# MAGIC The demo app needs one thing it can call: *patient features in, 7 chances of resistance out*.
# MAGIC We wrap the 7 per-drug models into a single MLflow model and register it, so the app (and the
# MAGIC judges) can find it under **Catalog -> workspace -> armd -> Models**.
# MAGIC
# MAGIC These are the models trained on 2008-2021 and scored above, so the reported test numbers
# MAGIC describe exactly the model that is registered.

# COMMAND ----------

from mlflow.models import infer_signature


class UrineResistanceModel(mlflow.pyfunc.PythonModel):
    """Patient features in -> one column per antibiotic with the predicted chance of resistance."""

    def __init__(self, models, features, drugs):
        self.models = models          # {drug_slug: fitted scikit-learn model}
        self.features = features      # column order the models were trained on
        self.drugs = drugs            # narrowest -> broadest spectrum

    def predict(self, context, model_input, params=None):
        X_in = model_input[self.features].astype("float64").to_numpy()
        return pd.DataFrame(
            {f"p_resistant_{d}": self.models[d].predict_proba(X_in)[:, 1] for d in self.drugs},
            index=model_input.index,
        )


packaged = UrineResistanceModel(
    models={d: test_predictions[(d, BEST_MODEL)]["model"] for d in DRUGS},
    features=FEATURES,
    drugs=DRUGS,
)

example_input = pdf[FEATURES].astype("float64").head(5)
signature = infer_signature(example_input, packaged.predict(None, example_input))

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_id=RUN_ID):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=packaged,
        input_example=example_input,
        signature=signature,
        registered_model_name=MODEL_NAME,
    )
print(f"Registered {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Check the registered model works
# MAGIC Load it back from Unity Catalog exactly as the demo app will, and score three test patients.

# COMMAND ----------

from mlflow import MlflowClient

latest = max(int(v.version) for v in MlflowClient().search_model_versions(f"name='{MODEL_NAME}'"))
loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest}")
sample = pdf[is_test][FEATURES].head(3)
display(loaded.predict(sample).round(3))
print(f"Loaded {MODEL_NAME} version {latest}")
