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
# MAGIC   each drug over the most recent year. Every patient gets the same drug.
# MAGIC - **Model policy (ours):** the same rule, using the model's personal prediction for each patient.
# MAGIC
# MAGIC **Two outcomes matter, and they pull against each other:**
# MAGIC - **Failure rate:** how often the recommended drug turned out not to work (the lab found the
# MAGIC   bacteria resistant). Lower is safer.
# MAGIC - **Broad-spectrum use:** how often piperacillin/tazobactam or meropenem was recommended. Lower is
# MAGIC   better for resistance.
# MAGIC
# MAGIC We also report **carbapenem use** (meropenem) on its own. It's part of "broad", but it's the
# MAGIC last-resort drug, so a policy that cuts piperacillin/tazobactam by switching some patients to
# MAGIC meropenem should be shown plainly rather than hidden inside one "broad" number.
# MAGIC
# MAGIC **The headline:** at the **same failure rate** as the antibiogram, how much less broad-spectrum
# MAGIC antibiotic does the model need?
# MAGIC
# MAGIC **No tuning on the test years.** The model setting that matches the antibiogram's failure rate is
# MAGIC chosen on the **2021 validation cultures**, then applied unchanged to 2022+. Choosing it on the
# MAGIC test years themselves would make the headline look better than it really is.
# MAGIC
# MAGIC **Input:** `workspace.armd.gold_test_predictions` and `workspace.armd.gold_validation_predictions`
# MAGIC from notebook 03.
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
# Carbapenems are the last-resort drugs stewardship teams most want to protect, so their share is
# reported on its own as well as inside "broad".
CARBAPENEM_DRUGS = ["meropenem"]

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
        "carbapenem_pct": round(100 * float(np.isin(picked, CARBAPENEM_DRUGS).mean()), 1),
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
# MAGIC ## 3. Load the predictions
# MAGIC Two sets: the **2022+ test cultures** (for every reported number) and the **2021 validation
# MAGIC cultures** (only for choosing the model setting in section 5).
# MAGIC
# MAGIC In both we keep the cultures where the lab tested **all 7 drugs** (about 92% of the test ones), so
# MAGIC every recommendation can be checked against a real lab result. The rest mostly grew organisms for
# MAGIC which some drugs aren't routinely tested; section 6 looks at what leaving them out hides.

# COMMAND ----------

import pandas as pd

test_preds = spark.table(tbl("gold_test_predictions")).toPandas()
val_preds = spark.table(tbl("gold_validation_predictions")).toPandas()

drug_order = (test_preds[["antibiotic_slug", "antibiotic", "spectrum_rank"]].drop_duplicates()
              .sort_values("spectrum_rank"))
DRUGS = list(drug_order["antibiotic_slug"])
DRUG_LABEL = dict(zip(drug_order["antibiotic_slug"], drug_order["antibiotic"]))


def grids(preds, name):
    """(P_model, P_antibiogram, R) for the cultures in `preds` that have all 7 drugs tested."""
    def grid(column):
        return preds.pivot(index="order_proc_id_coded", columns="antibiotic_slug", values=column)[DRUGS]

    complete = grid("resistant").dropna().index          # cultures with all 7 results
    print(f"{name}: {len(complete):,} of {preds['order_proc_id_coded'].nunique():,} cultures "
          f"have all {len(DRUGS)} drugs tested")
    return (grid("p_resistant").loc[complete].to_numpy(dtype=float),
            grid("p_resistant_antibiogram").loc[complete].to_numpy(dtype=float),
            grid("resistant").loc[complete].to_numpy(dtype=int))


P_model, P_antibiogram, R = grids(test_preds, "test")
P_model_val, _, R_val = grids(val_preds, "validation")

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
# MAGIC rates. A fairer comparison matches them on failure rate, in two steps:
# MAGIC
# MAGIC 1. **Choose the setting on 2021.** The antibiogram recommends one drug to everyone. On the
# MAGIC    validation cultures, find the model setting that fails **no more often** than giving everyone
# MAGIC    that drug, while using the least broad-spectrum antibiotic.
# MAGIC 2. **Report on 2022+.** Apply that setting, unchanged, to the test cultures and compare both
# MAGIC    policies there.
# MAGIC
# MAGIC Because the setting isn't tuned on the test years, the two failure rates on 2022+ will be close
# MAGIC but not identical. Report both numbers as they come out.
# MAGIC
# MAGIC **Be precise about what this shows:** it compares the same decision rule fed with
# MAGIC population-level numbers (the antibiogram) versus personal predictions. Real doctors already
# MAGIC tailor choices somewhat, and this dataset doesn't record what they actually prescribed, so
# MAGIC this is not a head-to-head against clinicians.

# COMMAND ----------

baseline = policy_outcomes(choice_antibiogram, R, DRUGS)

# Step 1: choose the model setting on the validation cultures. The yardstick is the drug the
# antibiogram policy recommends on the test years (it gives everyone the same one), so both steps
# compare against the same drug even if the antibiogram's pick would have differed a year earlier.
antibiogram_drug = int(choice_antibiogram[0])
baseline_val = policy_outcomes(np.full(len(R_val), antibiogram_drug), R_val, DRUGS)
matched = matched_model_policy(P_model_val, R_val, DRUGS, baseline_val["failure_pct"])

if matched is None:
    print("No model setting matches the antibiogram's failure rate on the validation cultures.")
else:
    matched_success, matched_val_out = matched
    print("Step 1, choosing the setting on the validation cultures:")
    display(pd.DataFrame([
        {"policy": f"antibiogram's drug for everyone ({DRUG_LABEL[DRUGS[antibiogram_drug]]})", **baseline_val},
        {"policy": f"model at {matched_success:.1%} target (matched failure rate)", **matched_val_out},
    ]))

    # Step 2: apply that setting, unchanged, to the test cultures
    choice_matched = recommend(P_model, 1 - matched_success)
    matched_out = policy_outcomes(choice_matched, R, DRUGS)
    print("Step 2, the result on the test cultures:")
    display(pd.DataFrame([
        {"policy": f"antibiogram at {TARGET_SUCCESS:.0%} target", **baseline},
        {"policy": f"model at {matched_success:.1%} target (chosen on validation)", **matched_out},
    ]))
    display(pd.DataFrame({
        "antibiotic": [DRUG_LABEL[d] for d in DRUGS],
        "antibiogram_pct_recommended": list(drug_mix(choice_antibiogram, DRUGS).values()),
        "model_pct_recommended": list(drug_mix(choice_matched, DRUGS).values()),
    }))
    HEADLINE = (
        f"With a setting chosen on earlier data to match today's antibiogram, the model's "
        f"recommendation failed for {matched_out['failure_pct']}% of test patients versus "
        f"{baseline['failure_pct']}% for the antibiogram, while recommending a broad-spectrum "
        f"antibiotic for {matched_out['broad_pct']}% of patients instead of {baseline['broad_pct']}%, "
        f"and a narrow-spectrum one for {matched_out['narrow_pct']}% instead of {baseline['narrow_pct']}%. "
        f"Within broad-spectrum, it recommended a carbapenem (meropenem) for "
        f"{matched_out['carbapenem_pct']}% of patients versus {baseline['carbapenem_pct']}% for the antibiogram."
    )
    print(HEADLINE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Caveat: the cultures that were left out
# MAGIC Everything above only scores cultures where the lab tested **all 7 drugs**. The ones left out are
# MAGIC not a random sample: a result is usually missing because of *which organism grew*, and those
# MAGIC organisms tend to be the resistant ones. A doctor can't leave these patients out, because the
# MAGIC organism isn't known when the antibiotic is chosen. So the failure rates above describe the
# MAGIC easier-to-score patients, and the tables below show how big the gap is.
# MAGIC
# MAGIC - **a) By drug:** how often each drug has no result among the left-out cultures, and how
# MAGIC   resistant they are where it was tested, next to the cultures we kept.
# MAGIC - **b) By organism:** what grew in the left-out cultures.
# MAGIC - **c) The antibiogram on every culture:** the antibiogram gives everyone one drug, so we can
# MAGIC   bound its failure rate over *all* test cultures by counting a missing result once as "worked"
# MAGIC   and once as "failed". The truth is somewhere in between.
# MAGIC
# MAGIC The model can't be bounded the same way yet: notebook 03 only predicts for (culture, drug) pairs
# MAGIC that have a lab result, so the rule can't walk all 7 drugs for these cultures.
# MAGIC
# MAGIC Not shown here: about 1% of positive urine cultures have a result for *none* of the 7 drugs
# MAGIC (mostly Group B strep), so they never reach the training table at all.

# COMMAND ----------

from pyspark.sql import functions as F

results = test_preds.pivot(index="order_proc_id_coded", columns="antibiotic_slug", values="resistant")[DRUGS]
kept = results[results.notna().all(axis=1)]
left_out = results[results.isna().any(axis=1)]
print(f"{len(left_out):,} of {len(results):,} test cultures ({len(left_out) / len(results):.1%}) are left out")

# a) by drug
display(pd.DataFrame({
    "antibiotic": [DRUG_LABEL[d] for d in DRUGS],
    "left_out_no_result_pct": (100 * left_out.isna().mean()).round(1).to_numpy(),
    "left_out_resistant_pct_where_tested": (100 * left_out.mean()).round(1).to_numpy(),
    "kept_resistant_pct": (100 * kept.mean()).round(1).to_numpy(),
}))

# b) by organism (a culture that grew two organisms is listed as "A + B")
organisms = (
    spark.table(tbl("bronze_cohort"))
    .where(F.col("organism").isNotNull() & F.col("order_proc_id_coded").isin(list(left_out.index)))
    .select("order_proc_id_coded", "organism")
    .distinct()
    .toPandas()
)
grew = organisms.groupby("order_proc_id_coded")["organism"].apply(lambda names: " + ".join(sorted(names)))
top_organisms = grew.value_counts().head(10)
display(pd.DataFrame({
    "organism": top_organisms.index,
    "left_out_cultures": top_organisms.to_numpy(),
    "pct_of_left_out": (100 * top_organisms / len(left_out)).round(1).to_numpy(),
}))

# c) the antibiogram's failure rate over every test culture
antibiogram_result = results[DRUGS[antibiogram_drug]]          # 1 = resistant, NaN = no result
LEFT_OUT = {
    "left_out_cultures": int(len(left_out)),
    "left_out_pct": round(100 * len(left_out) / len(results), 1),
    "antibiogram_drug_no_result": int(antibiogram_result.isna().sum()),
    "antibiogram_failure_pct_best_case": round(100 * float((antibiogram_result == 1).mean()), 2),
    "antibiogram_failure_pct_worst_case": round(
        100 * float(((antibiogram_result == 1) | antibiogram_result.isna()).mean()), 2),
}
display(pd.DataFrame([
    {"cultures scored": "only those with all 7 results (used above)", "cultures": len(kept),
     "antibiogram_failure_pct": baseline["failure_pct"]},
    {"cultures scored": "all, a missing result counted as worked", "cultures": len(results),
     "antibiogram_failure_pct": LEFT_OUT["antibiogram_failure_pct_best_case"]},
    {"cultures scored": "all, a missing result counted as failed", "cultures": len(results),
     "antibiogram_failure_pct": LEFT_OUT["antibiogram_failure_pct_worst_case"]},
]))
print(f"{DRUG_LABEL[DRUGS[antibiogram_drug]]} (the antibiogram's drug) has no result for "
      f"{LEFT_OUT['antibiogram_drug_no_result']:,} test cultures.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. The full trade-off (chart for the slides)
# MAGIC Each point is one setting of the rule, from "at least 70% likely to work" to "at least 99%".
# MAGIC **Down and to the left is better:** fewer failures *and* less broad-spectrum use. Wherever the
# MAGIC model's line sits below the antibiogram's, the model gets the same safety with fewer broad drugs.
# MAGIC The two large dots are the policies compared in section 5.

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
# MAGIC ## 8. Example patients
# MAGIC What a doctor would see for a few test cultures: each drug's predicted chance of working, the
# MAGIC recommendation at the 90% target, and whether that drug really worked.

# COMMAND ----------

rng = np.random.default_rng(0)
examples = rng.choice(len(R), size=6, replace=False)
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
# MAGIC ## 9. Save and log the results
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
    mlflow.log_params({"target_success": TARGET_SUCCESS, "test_cultures": len(R),
                       "validation_cultures": len(R_val),
                       "narrow_drugs": ",".join(NARROW_DRUGS), "broad_drugs": ",".join(BROAD_DRUGS),
                       "carbapenem_drugs": ",".join(CARBAPENEM_DRUGS)})
    for name, out in [("antibiogram", baseline), ("model_at_target", policy_outcomes(choice_model, R, DRUGS)),
                      ("antibiogram_validation", baseline_val)]:
        for k, v in out.items():
            mlflow.log_metric(f"{name}.{k}", v)
    if matched is not None:
        # the setting was chosen on the validation cultures; model_matched.* are its test results
        mlflow.log_param("matched_target_success", matched_success)
        mlflow.log_param("antibiogram_drug", DRUGS[antibiogram_drug])
        for k, v in matched_out.items():
            mlflow.log_metric(f"model_matched.{k}", v)
        for k, v in matched_val_out.items():
            mlflow.log_metric(f"model_matched_validation.{k}", v)
        mlflow.set_tag("headline", HEADLINE)
    for k, v in LEFT_OUT.items():       # the caveat from section 6
        mlflow.log_metric(f"left_out.{k}", v)
    mlflow.log_figure(fig, "tradeoff_chart.png")
    mlflow.log_table(tradeoff, artifact_file="tradeoff.json")

print(f"Saved {tbl('gold_policy_tradeoff')} and logged the run to MLflow")
