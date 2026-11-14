# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Recommend: turn predictions into an antibiotic choice
# MAGIC
# MAGIC **The rule:** go through the antibiotics from **narrowest to broadest spectrum** and recommend the
# MAGIC first one the patient is likely enough to respond to, e.g. at least a 90% predicted chance of
# MAGIC working. If none qualifies, recommend the broadest (meropenem).
# MAGIC
# MAGIC Narrow-spectrum drugs kill fewer of the body's harmless bacteria, so using them whenever they'll
# MAGIC work puts less pressure on bacteria to become resistant. That's antibiotic stewardship.
# MAGIC
# MAGIC **What we compare (on the 2022+ test cultures, which no model saw during training):**
# MAGIC - **Antibiogram policy (today):** the same rule, using the hospital's overall success rate for
# MAGIC   each drug. Every patient gets the same drug.
# MAGIC - **Model policy (ours):** the same rule, using the model's personal prediction for each patient.
# MAGIC
# MAGIC **Two outcomes matter, and they pull against each other:**
# MAGIC - **Failure rate:** how often the recommended drug turned out not to work (the lab found the
# MAGIC   bacteria resistant). Lower is safer.
# MAGIC - **Broad-spectrum use:** how often piperacillin/tazobactam or meropenem was recommended. Lower is
# MAGIC   better for resistance.
# MAGIC
# MAGIC **The headline:** at the **same failure rate** as the antibiogram, how much less broad-spectrum
# MAGIC antibiotic does the model need?
# MAGIC
# MAGIC **Input:** `workspace.armd.gold_test_predictions` from notebook 03.
# MAGIC **Output:** tables and a chart for the slides, saved to `workspace.armd.gold_policy_tradeoff`
# MAGIC and logged to MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "armd"


def tbl(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"


# The required chance that the recommended drug works. 0.90 = "at least 90% likely to work".
TARGET_SUCCESS = 0.90

# Spectrum groups used for reporting (the drugs are ranked narrowest -> broadest in notebook 02)
NARROW_DRUGS = ["nitrofurantoin", "trimethoprim_sulfamethoxazole", "cefazolin"]
BROAD_DRUGS = ["piperacillin_tazobactam", "meropenem"]

EXPERIMENT_NAME = "flemingrx_urine_resistance"   # same MLflow experiment as notebook 03

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The building blocks
# MAGIC Plain numpy functions, so the same code can be tested on a laptop.
# MAGIC
# MAGIC The predictions are arranged as a grid: one row per culture, one column per drug (narrowest
# MAGIC first). `P` holds the predicted chance of **resistance**, `R` holds what really happened
# MAGIC (1 = resistant).

# COMMAND ----------

import numpy as np


def recommend(P, max_risk):
    """Column index of the narrowest drug with predicted resistance <= max_risk (broadest if none)."""
    ok = P <= max_risk
    return np.where(ok.any(axis=1), ok.argmax(axis=1), P.shape[1] - 1)


def policy_outcomes(choice, R, drugs):
    """Failure rate and spectrum mix of a set of recommendations."""
    failed = R[np.arange(len(choice)), choice]
    picked = np.array(drugs)[choice]
    return {
        "failure_pct": round(100 * float(failed.mean()), 2),
        "narrow_pct": round(100 * float(np.isin(picked, NARROW_DRUGS).mean()), 1),
        "broad_pct": round(100 * float(np.isin(picked, BROAD_DRUGS).mean()), 1),
    }


def drug_mix(choice, drugs):
    """Percent of cultures recommended each drug."""
    return {d: round(100 * float((choice == i).mean()), 1) for i, d in enumerate(drugs)}


def tradeoff_curve(P, R, drugs, successes):
    """Outcomes of the rule at each required chance of success."""
    return [{"target_success": round(float(s), 3), **policy_outcomes(recommend(P, 1 - s), R, drugs)}
            for s in successes]


def matched_model_policy(P_model, R, drugs, max_failure_pct):
    """Among model thresholds whose failure rate is no worse than max_failure_pct, the one that uses
    the least broad-spectrum antibiotic. Returns (required success, outcomes) or None."""
    best = None
    for s in np.arange(0.50, 1.0, 0.001):
        out = policy_outcomes(recommend(P_model, 1 - s), R, drugs)
        if out["failure_pct"] <= max_failure_pct:
            key = (out["broad_pct"], -out["narrow_pct"])
            if best is None or key < best[0]:
                best = (key, round(float(s), 3), out)
    return None if best is None else (best[1], best[2])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load the test-year predictions
# MAGIC We keep the cultures where the lab tested **all 7 drugs** (about 89% of them), so every
# MAGIC recommendation can be checked against a real lab result. The rest mostly grew organisms for
# MAGIC which some drugs aren't routinely tested.

# COMMAND ----------

import pandas as pd

preds = spark.table(tbl("gold_test_predictions")).toPandas()

drug_order = preds[["antibiotic_slug", "antibiotic", "spectrum_rank"]].drop_duplicates().sort_values("spectrum_rank")
DRUGS = list(drug_order["antibiotic_slug"])
DRUG_LABEL = dict(zip(drug_order["antibiotic_slug"], drug_order["antibiotic"]))


def grid(column):
    return preds.pivot(index="order_proc_id_coded", columns="antibiotic_slug", values=column)[DRUGS]


complete = grid("resistant").dropna().index          # cultures with all 7 results
P_model = grid("p_resistant").loc[complete].to_numpy(dtype=float)
P_antibiogram = grid("p_resistant_antibiogram").loc[complete].to_numpy(dtype=float)
R = grid("resistant").loc[complete].to_numpy(dtype=int)

print(f"{len(complete):,} of {preds['order_proc_id_coded'].nunique():,} test cultures have all 7 drugs tested")
display(pd.DataFrame({
    "antibiotic": [DRUG_LABEL[d] for d in DRUGS],
    "antibiogram_success_pct": (100 * (1 - P_antibiogram[0])).round(1),
    "actual_resistant_pct_in_test": (100 * R.mean(axis=0)).round(1),
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Both policies at the chosen target (90% chance of working)
# MAGIC The antibiogram gives everyone the same drug: the narrowest one whose overall success rate clears
# MAGIC the bar. The model can give low-risk patients a narrow drug and save broad drugs for high-risk ones.

# COMMAND ----------

choice_antibiogram = recommend(P_antibiogram, 1 - TARGET_SUCCESS)
choice_model = recommend(P_model, 1 - TARGET_SUCCESS)

display(pd.DataFrame([
    {"policy": "antibiogram (today)", **policy_outcomes(choice_antibiogram, R, DRUGS)},
    {"policy": "model (ours)", **policy_outcomes(choice_model, R, DRUGS)},
]))

display(pd.DataFrame({
    "antibiotic": [DRUG_LABEL[d] for d in DRUGS],
    "antibiogram_pct_recommended": list(drug_mix(choice_antibiogram, DRUGS).values()),
    "model_pct_recommended": list(drug_mix(choice_model, DRUGS).values()),
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The headline: same safety, less broad-spectrum use
# MAGIC Section 4 compares the policies at the same *target*, but they can end up with different failure
# MAGIC rates. A fairer comparison: find the model setting that fails **no more often** than the
# MAGIC antibiogram, then compare how much broad-spectrum antibiotic each one uses.
# MAGIC
# MAGIC **Be precise about what this shows:** it compares the same decision rule fed with
# MAGIC population-level numbers (the antibiogram) versus personal predictions. Real doctors already
# MAGIC tailor choices somewhat, and this dataset doesn't record what they actually prescribed, so
# MAGIC this is not a head-to-head against clinicians.

# COMMAND ----------

baseline = policy_outcomes(choice_antibiogram, R, DRUGS)
matched = matched_model_policy(P_model, R, DRUGS, baseline["failure_pct"])

if matched is None:
    print("No model setting matches the antibiogram's failure rate.")
else:
    matched_success, matched_out = matched
    choice_matched = recommend(P_model, 1 - matched_success)
    display(pd.DataFrame([
        {"policy": f"antibiogram at {TARGET_SUCCESS:.0%} target", **baseline},
        {"policy": f"model at {matched_success:.1%} target (matched failure rate)", **matched_out},
    ]))
    display(pd.DataFrame({
        "antibiotic": [DRUG_LABEL[d] for d in DRUGS],
        "antibiogram_pct_recommended": list(drug_mix(choice_antibiogram, DRUGS).values()),
        "model_pct_recommended": list(drug_mix(choice_matched, DRUGS).values()),
    }))
    HEADLINE = (
        f"With no more treatment failures than today's antibiogram ({matched_out['failure_pct']}% vs "
        f"{baseline['failure_pct']}%), the model recommends a broad-spectrum antibiotic for "
        f"{matched_out['broad_pct']}% of patients instead of {baseline['broad_pct']}%, and a "
        f"narrow-spectrum one for {matched_out['narrow_pct']}% instead of {baseline['narrow_pct']}%."
    )
    print(HEADLINE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. The full trade-off (chart for the slides)
# MAGIC Each point is one setting of the rule, from "at least 70% likely to work" to "at least 99%".
# MAGIC **Down and to the left is better:** fewer failures *and* less broad-spectrum use. Wherever the
# MAGIC model's line sits below the antibiogram's, the model gets the same safety with fewer broad drugs.

# COMMAND ----------

import matplotlib.pyplot as plt

SUCCESS_GRID = np.round(np.arange(0.70, 0.995, 0.005), 3)
curve_model = tradeoff_curve(P_model, R, DRUGS, SUCCESS_GRID)
curve_antibiogram = tradeoff_curve(P_antibiogram, R, DRUGS, SUCCESS_GRID)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([r["broad_pct"] for r in curve_antibiogram], [r["failure_pct"] for r in curve_antibiogram],
        "o--", color="#888888", label="Antibiogram (today)")
ax.plot([r["broad_pct"] for r in curve_model], [r["failure_pct"] for r in curve_model],
        "-", color="#1f6feb", linewidth=2.5, label="Model (ours)")
if matched is not None:
    ax.scatter([baseline["broad_pct"], matched_out["broad_pct"]],
               [baseline["failure_pct"], matched_out["failure_pct"]],
               s=120, c=["#888888", "#1f6feb"], zorder=5, edgecolors="black")
ax.set_xlabel("Patients given a broad-spectrum antibiotic (%)")
ax.set_ylabel("Recommended antibiotic failed (%)")
ax.set_title("Same safety, fewer broad-spectrum antibiotics")
ax.grid(alpha=0.3)
ax.legend()
plt.show()

tradeoff = pd.DataFrame(
    [{"policy": "model", **r} for r in curve_model]
    + [{"policy": "antibiogram", **r} for r in curve_antibiogram]
)
display(tradeoff[tradeoff["target_success"].isin([0.80, 0.85, 0.90, 0.95])])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Example patients
# MAGIC What a doctor would see for a few test cultures: each drug's predicted chance of working, the
# MAGIC recommendation at the 90% target, and whether that drug really worked.

# COMMAND ----------

rng = np.random.default_rng(0)
examples = rng.choice(len(complete), size=6, replace=False)
rows = []
for i in examples:
    row = {DRUG_LABEL[d]: f"{100 * (1 - P_model[i, j]):.0f}%" for j, d in enumerate(DRUGS)}
    picked = choice_model[i]
    row["recommended"] = DRUG_LABEL[DRUGS[picked]]
    row["it_worked"] = "yes" if R[i, picked] == 0 else "NO"
    rows.append(row)
display(pd.DataFrame(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save and log the results
# MAGIC The trade-off table goes to `workspace.armd.gold_policy_tradeoff` (handy for the demo app and
# MAGIC slides), and the headline numbers and chart are logged to the same MLflow experiment as notebook 03.

# COMMAND ----------

import mlflow

(
    spark.createDataFrame(tradeoff)
    .write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(tbl("gold_policy_tradeoff"))
)

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/{EXPERIMENT_NAME}")
with mlflow.start_run(run_name="recommendation_policy"):
    mlflow.log_params({"target_success": TARGET_SUCCESS, "test_cultures": len(complete),
                       "narrow_drugs": ",".join(NARROW_DRUGS), "broad_drugs": ",".join(BROAD_DRUGS)})
    for name, out in [("antibiogram", baseline), ("model_at_target", policy_outcomes(choice_model, R, DRUGS))]:
        for k, v in out.items():
            mlflow.log_metric(f"{name}.{k}", v)
    if matched is not None:
        mlflow.log_param("matched_target_success", matched_success)
        for k, v in matched_out.items():
            mlflow.log_metric(f"model_matched.{k}", v)
        mlflow.set_tag("headline", HEADLINE)
    mlflow.log_figure(fig, "tradeoff_chart.png")
    mlflow.log_table(tradeoff, artifact_file="tradeoff.json")

print(f"Saved {tbl('gold_policy_tradeoff')} and logged the run to MLflow")
