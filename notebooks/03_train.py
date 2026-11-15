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
# MAGIC | **Antibiogram** | Gives every patient the hospital's overall resistance rate for that drug, measured over the most recent training year | What hospitals use today (they rebuild it every year). The model must beat this |
# MAGIC | **Prior-resistance rule** | Two rates: one for patients resistant to this drug in the past year, one for everyone else | The obvious rule a careful doctor would apply. Shows whether ML adds anything beyond it |
# MAGIC | **Logistic regression** | A weighted sum of all features | Simple and explainable |
# MAGIC | **Gradient boosting** | Hundreds of small decision trees added together | Usually the most accurate on tables like ours; your paper found tree-based models did best |
# MAGIC
# MAGIC **One model per antibiotic.** Each drug gets its own model, because what predicts nitrofurantoin
# MAGIC resistance isn't the same as what predicts ceftriaxone resistance.
# MAGIC
# MAGIC **How the data is split (by date, never shuffled):**
# MAGIC - **Step A, choose the approach:** train on 2008-2020, compare on **2021**. This is also where
# MAGIC   gradient boosting's settings are tuned, and where we check that labs and the newer features
# MAGIC   are worth keeping.
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
# MAGIC - Every input ranked by importance, saved to `workspace.armd.gold_feature_importance`.
# MAGIC - 2021 validation predictions saved to `workspace.armd.gold_validation_predictions`, so notebook
# MAGIC   04 can choose its settings without looking at the test years.

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

# The antibiogram baseline only uses the most recent year(s) of its training data. Real hospital
# antibiograms are rebuilt every year, and resistance drifts, so an average over every year since
# 2008 would be a stale (and unrealistically weak or strong) baseline.
ANTIBIOGRAM_YEARS = 1

# Labs and vital signs are left OUT by default, for two reasons:
#  1. The dataset doesn't say whether the 14-day lab window can include days after the culture was
#     ordered, so they might leak future information.
#  2. Most outpatients have none (58% of rows have no labs), and the demo is simpler without them.
# Section 4 measures how much they would add, so the team can make an informed call.
USE_LABS_VITALS = False

# Gradient-boosting settings to try in Step A. The best one on 2021 is used from then on.
# The first is the setting used before tuning was added, so tuning can only match or beat it.
#  - learning_rate: how big a step each new tree takes (smaller = slower but often more accurate)
#  - max_leaf_nodes: how detailed each tree can be
#  - min_samples_leaf: the fewest cultures a tree may base a decision on (higher = less overfitting)
#  - l2_regularization: a brake on extreme predictions
GB_CANDIDATES = [
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 50,  "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 50,  "l2_regularization": 1.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 3.0},
    {"learning_rate": 0.10, "max_leaf_nodes": 31, "min_samples_leaf": 50,  "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 200, "l2_regularization": 10.0},
]
GB_PARAMS = dict(GB_CANDIDATES[0])   # replaced by the best candidate in section 4

# Features added in the "model improvements" round. Section 4 measures what they add by training
# once without them.
NEW_FEATURE_PREFIXES = ("days_since_drug_", "days_since_subtype_", "days_since_resist_",
                        "n_antibiotic_days_", "n_prior_cultures_", "comorb_", "n_comorbidities")

# Section 6b ranks the inputs by importance. It scores the test cultures a few hundred times, which
# takes several minutes; set RUN_IMPORTANCE to False to skip it.
RUN_IMPORTANCE = True
IMPORTANCE_REPEATS = 3            # shuffles per input (more = steadier numbers, slower)
IMPORTANCE_MIN_RESISTANT = 200    # skip drugs with fewer resistant test cultures: too noisy to rank on
# Which group each input belongs to: (group name, how the input's name starts)
FEATURE_GROUPS = [
    ("Past resistance", ("prior_resist_", "days_since_resist_", "n_prior_resistant_drugs")),
    ("Antibiotic exposure", ("abx_class_", "days_since_drug_", "days_since_subtype_",
                             "days_since_last_antibiotic", "n_antibiotic_days_")),
    ("Past organisms", ("prior_org_", "n_prior_cultures_")),
    ("Where the culture was ordered", ("ward_", "is_inpatient")),
    ("Demographics", ("age_min", "gender", "adi_")),
    ("Procedures", ("proc_",)),
    ("Nursing home", ("nursing_home",)),
    ("Comorbidities", ("comorb_", "n_comorbidities")),
    ("Labs and vital signs", ("lab_", "vital_")),
]

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
        # handles missing values by itself; stops adding trees once a held-out 10% stops improving.
        # GB_PARAMS holds the tunable settings (see GB_CANDIDATES).
        return HistGradientBoostingClassifier(
            **GB_PARAMS, max_iter=1000, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, random_state=SEED,
        )
    raise ValueError(kind)


def fit_and_predict(kind, X_train, y_train, year_train, X_eval, prior_col):
    """Train one approach and return (fitted model or None, predicted chance of resistance)."""
    if kind == "antibiogram":
        # the resistance rate over the last ANTIBIOGRAM_YEARS year(s) of the training data
        recent = year_train > year_train.max() - ANTIBIOGRAM_YEARS
        return None, np.full(len(X_eval), y_train[recent].mean())
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


def compare_models(X, y, drug, year, features, drugs_in_order, train_mask, eval_mask, kinds):
    """Train every approach in `kinds` for every drug. Returns (score rows, predictions)."""
    rows, predictions = [], {}
    for d in drugs_in_order:
        train = (drug == d) & train_mask
        test = (drug == d) & eval_mask
        prior_col = features.index(f"prior_resist_{d}")
        for kind in kinds:
            model, p = fit_and_predict(kind, X[train], y[train], year[train], X[test], prior_col)
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


def feature_group(name):
    """The group in FEATURE_GROUPS that an input belongs to."""
    return next((group for group, starts in FEATURE_GROUPS if name.startswith(starts)), "Other")


def auroc_drop(model, X_eval, y_eval, columns, repeats, rng):
    """How much AUROC falls when `columns` are shuffled between patients (averaged over `repeats`).

    Shuffling an input keeps its values but breaks its link to the patient, so the model can no
    longer use it. A big drop means the model relies on it. Several columns are shuffled together
    (each patient gets another patient's whole set), which measures a group of inputs at once.
    """
    base = roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1])
    drops = []
    for _ in range(repeats):
        shuffled = X_eval.copy()
        shuffled[:, columns] = X_eval[rng.permutation(len(X_eval))][:, columns]
        drops.append(base - roc_auc_score(y_eval, model.predict_proba(shuffled)[:, 1]))
    return float(np.mean(drops))


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
# MAGIC About 509,000 rows: small enough to work with in memory. Features become one numeric table `X`,
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
# MAGIC
# MAGIC **First, tune gradient boosting.** Each setting in `GB_CANDIDATES` is trained on 2008-2020 and
# MAGIC scored on 2021; the one with the highest average AUROC is kept for the rest of the notebook.
# MAGIC Because this only looks at 2021, the test years still give an honest final score.

# COMMAND ----------

step_a_train = (~is_test) & (year < VALIDATION_YEAR)
step_a_eval = (~is_test) & (year == VALIDATION_YEAR)

tuning_rows = []
for i, params in enumerate(GB_CANDIDATES):
    GB_PARAMS = dict(params)
    rows, _ = compare_models(X, y, drug, year, FEATURES, DRUGS, step_a_train, step_a_eval,
                             ["gradient_boosting"])
    scores = average_by_model(rows)[0]
    tuning_rows.append({"setting": i, **params, "mean_auroc": scores["mean_auroc"],
                        "mean_brier": scores["mean_brier"]})

best_setting = max(tuning_rows, key=lambda r: r["mean_auroc"])["setting"]   # ties: earliest wins
GB_PARAMS = dict(GB_CANDIDATES[best_setting])
display(pd.DataFrame(tuning_rows))
print(f"Chosen gradient-boosting setting {best_setting}: {GB_PARAMS}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Then compare the four approaches** on 2021, with gradient boosting at its chosen setting.

# COMMAND ----------

val_rows, val_predictions = compare_models(X, y, drug, year, FEATURES, DRUGS, step_a_train,
                                           step_a_eval, BASELINES + ML_MODELS)
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

with_rows, _ = compare_models(X_with_labs, y, drug, year, FEATURES_WITH_LABS, DRUGS, step_a_train,
                              step_a_eval, ["gradient_boosting"])
without = {r["antibiotic"]: r["auroc"] for r in val_rows if r["model"] == "gradient_boosting"}
display(pd.DataFrame([
    {"antibiotic": r["antibiotic"], "auroc_without_labs": without[r["antibiotic"]],
     "auroc_with_labs": r["auroc"], "gain": round(r["auroc"] - without[r["antibiotic"]], 3)}
    for r in with_rows
]))
del X_with_labs   # free the memory; only needed for this comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ### Do the new features help?
# MAGIC Same comparison again: gradient boosting with and without the features added in the
# MAGIC improvements round (specific drugs, antibiotic subtypes, "days since" timings, yearly counts and
# MAGIC comorbidities; see `NEW_FEATURE_PREFIXES`). A positive gain on most drugs means they earn their
# MAGIC place.

# COMMAND ----------

OLD_FEATURES = [c for c in FEATURES if not c.startswith(NEW_FEATURE_PREFIXES)]
old_idx = [FEATURES.index(c) for c in OLD_FEATURES]
old_rows, _ = compare_models(X[:, old_idx], y, drug, year, OLD_FEATURES, DRUGS, step_a_train,
                             step_a_eval, ["gradient_boosting"])
with_new = {r["antibiotic"]: r["auroc"] for r in val_rows if r["model"] == "gradient_boosting"}
print(f"{len(FEATURES) - len(OLD_FEATURES)} new features ({len(OLD_FEATURES)} before, {len(FEATURES)} now)")
display(pd.DataFrame([
    {"antibiotic": r["antibiotic"], "auroc_old_features": r["auroc"],
     "auroc_with_new_features": with_new[r["antibiotic"]],
     "gain": round(with_new[r["antibiotic"]] - r["auroc"], 3)}
    for r in old_rows
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

test_rows, test_predictions = compare_models(X, y, drug, year, FEATURES, DRUGS, ~is_test, is_test,
                                             BASELINES + ML_MODELS)
test_table = pd.DataFrame(test_rows)
summary_table = pd.DataFrame(average_by_model(test_rows))

with mlflow.start_run(run_name=f"final_{BEST_MODEL}") as run:
    mlflow.log_params({
        "chosen_model": BEST_MODEL, "use_labs_vitals": USE_LABS_VITALS,
        "n_features": len(FEATURES), "train_years": f"2008-{VALIDATION_YEAR}", "test_years": "2022+",
        "antibiogram_years": ANTIBIOGRAM_YEARS,
        **{f"gb_{k}": v for k, v in GB_PARAMS.items()},
    })
    mlflow.log_table(pd.DataFrame(tuning_rows), artifact_file="gb_tuning_2021.json")
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
# MAGIC ## 6b. Which inputs matter most?
# MAGIC For each input, we shuffle its values between the test patients and measure how much the chosen
# MAGIC model's AUROC falls. A bigger fall means the model relies on that input more. This is called
# MAGIC permutation importance.
# MAGIC
# MAGIC **How to read the tables:**
# MAGIC - **Groups first.** Inputs that say nearly the same thing share the credit: "days since
# MAGIC   nitrofurantoin" and "nitrofuran class in the last 90 days" overlap, so each looks smaller alone
# MAGIC   than the information is worth. The group table shuffles a whole group together, which is the
# MAGIC   fairer summary.
# MAGIC - **A fall near 0** means removing that one input changes almost nothing. It does not mean the
# MAGIC   fact is medically irrelevant.
# MAGIC - Figures are averaged over the antibiotics with enough resistant test cultures to rank on
# MAGIC   (`IMPORTANCE_MIN_RESISTANT`); meropenem has too few.
# MAGIC
# MAGIC The full ranking is saved to `workspace.armd.gold_feature_importance`. The demo app can use it
# MAGIC to say *why* a patient's prediction is high.

# COMMAND ----------

if RUN_IMPORTANCE:
    rng = np.random.default_rng(SEED)
    scored_drugs = [d for d in DRUGS
                    if y[test_predictions[(d, BEST_MODEL)]["rows"]].sum() >= IMPORTANCE_MIN_RESISTANT]
    groups = list(dict.fromkeys(feature_group(f) for f in FEATURES))
    feature_drops = {f: {} for f in FEATURES}
    group_drops = {g: {} for g in groups}
    for d in scored_drugs:
        pred = test_predictions[(d, BEST_MODEL)]
        X_eval, y_eval, model = X[pred["rows"]], y[pred["rows"]], pred["model"]
        for j, f in enumerate(FEATURES):
            feature_drops[f][d] = auroc_drop(model, X_eval, y_eval, [j], IMPORTANCE_REPEATS, rng)
        for g in groups:
            members = [j for j, f in enumerate(FEATURES) if feature_group(f) == g]
            group_drops[g][d] = auroc_drop(model, X_eval, y_eval, members, IMPORTANCE_REPEATS, rng)

    group_importance = pd.DataFrame([
        {"group": g, "inputs": sum(feature_group(f) == g for f in FEATURES),
         "mean_auroc_drop": round(float(np.mean(list(drops.values()))), 4)}
        for g, drops in group_drops.items()
    ]).sort_values("mean_auroc_drop", ascending=False, ignore_index=True)

    feature_importance = pd.DataFrame([
        {"feature": f, "group": feature_group(f),
         "mean_auroc_drop": round(float(np.mean(list(drops.values()))), 4),
         "matters_most_for": max(drops, key=drops.get),
         **{f"drop_{d}": round(drops[d], 4) for d in scored_drugs}}
        for f, drops in feature_drops.items()
    ]).sort_values("mean_auroc_drop", ascending=False, ignore_index=True)
    feature_importance.insert(0, "rank", feature_importance.index + 1)

    (
        spark.createDataFrame(feature_importance)
        .write.mode("overwrite").option("overwriteSchema", True)
        .saveAsTable(tbl("gold_feature_importance"))
    )
    with mlflow.start_run(run_id=RUN_ID):
        mlflow.log_table(feature_importance, artifact_file="feature_importance.json")
        mlflow.log_table(group_importance, artifact_file="feature_group_importance.json")

    print(f"Ranked {len(FEATURES)} inputs on {len(scored_drugs)} antibiotics: {', '.join(scored_drugs)}")
    display(group_importance)
    display(feature_importance.head(25))
else:
    print("RUN_IMPORTANCE is False: skipped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save the predictions for notebook 04
# MAGIC One row per (culture, antibiotic): the true answer, the chosen model's predicted chance of
# MAGIC resistance, and the antibiogram's (the baseline notebook 04 compares against).
# MAGIC
# MAGIC Two tables are saved:
# MAGIC - **`gold_test_predictions`**: the 2022+ test cultures, scored by the models trained on 2008-2021.
# MAGIC - **`gold_validation_predictions`**: the 2021 cultures, scored by the Step A models trained on
# MAGIC   2008-2020. Notebook 04 chooses its settings on these, so the test years are only used to
# MAGIC   report the result, never to tune it.

# COMMAND ----------

def prediction_frame(predictions):
    """The chosen model's and the antibiogram's predictions for one set of cultures."""
    frames = []
    for d in DRUGS:
        model_pred = predictions[(d, BEST_MODEL)]
        base_pred = predictions[(d, "antibiogram")]
        part = pdf.iloc[model_pred["rows"]][
            ["order_proc_id_coded", "anon_id", "order_year", "antibiotic", "antibiotic_slug",
             "spectrum_rank", "resistant"]].copy()
        part["p_resistant"] = model_pred["p"]
        part["p_resistant_antibiogram"] = base_pred["p"]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


for table, predictions in [("gold_test_predictions", test_predictions),
                           ("gold_validation_predictions", val_predictions)]:
    predictions_pdf = prediction_frame(predictions)
    (
        spark.createDataFrame(predictions_pdf)
        .write.mode("overwrite").option("overwriteSchema", True)
        .saveAsTable(tbl(table))
    )
    print(f"Saved {len(predictions_pdf):,} rows to {tbl(table)}")

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
# MAGIC Whatever calls the model must send every feature as a decimal number (float64).

# COMMAND ----------

from mlflow import MlflowClient

latest = max(int(v.version) for v in MlflowClient().search_model_versions(f"name='{MODEL_NAME}'"))
loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest}")
# the model expects every feature as a decimal number (float64), as in training; MLflow rejects
# whole-number columns like age_min = 75 instead of 75.0, so convert first
sample = pdf[is_test][FEATURES].astype("float64").head(3)
display(loaded.predict(sample).round(3))
print(f"Loaded {MODEL_NAME} version {latest}")
