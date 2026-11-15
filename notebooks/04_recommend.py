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
# MAGIC **A bar that depends on severity (section 5b).** One bar for everyone is too strict for mild
# MAGIC infections and too loose for severe ones. Section 5b gives each culture a severity level from
# MAGIC its care setting and vital signs, and uses a different bar for each level.
# MAGIC
# MAGIC **Input:** `workspace.armd.gold_test_predictions` and `workspace.armd.gold_validation_predictions`
# MAGIC from notebook 03; `gold_urine_features` and `bronze_vitals` for the severity level.
# MAGIC **Output:** tables and a chart for the slides, saved to `workspace.armd.gold_policy_tradeoff`
# MAGIC and `workspace.armd.gold_severity_policy`, and logged to MLflow.

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

# --- Severity (section 5b) ---
# How sick the patient looks decides how much risk of failure is acceptable, so each severity
# level gets its own bar. The level comes from where the culture was ordered and the vital signs
# recorded in the 48 hours BEFORE the order (the ARMD paper states that window).
SEVERITY_LEVELS = ["Mild", "Systemic signs", "Severe"]
# Required chance of working for each level, in the same order. These are defaults for a clinician
# to confirm. They follow US guideline thresholds: for a simple bladder infection, a drug is
# acceptable while local resistance is under 20%; for a kidney infection, under 10%.
SEVERITY_BARS = [0.85, 0.90, 0.95]
# Bars tried for the mild level on the 2021 validation cultures, to show what each would cost
MILD_BARS_TO_COMPARE = [0.80, 0.825, 0.85, 0.875, 0.90]

# Vital-sign cut-offs (standard sepsis-screening values; temperature is in Fahrenheit)
FEVER_F = 100.4             # upper-quartile temperature at or above this = fever
FAST_HEART = 90             # median heart rate above this
FAST_BREATHING = 20         # median breaths per minute above this
VERY_FAST_BREATHING = 22    # with LOW_NORMAL_SBP: two warning signs at once -> severe
LOW_NORMAL_SBP = 100        # lower-quartile systolic blood pressure at or below this
LOW_SBP = 90                # lower-quartile systolic blood pressure below this = low blood pressure

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


def severity_level(icu, ward_ip, ward_er, temp_q75, heart, breathing, sbp_q25):
    """Position in SEVERITY_LEVELS for each culture (0 = mild, 1 = systemic signs, 2 = severe).

    Inputs are arrays, one value per culture: care-setting flags (1 = yes) and vital signs from the
    48 hours before the order. A vital sign that wasn't recorded (NaN) counts as not abnormal.
      severe:   intensive care, or low blood pressure, or very fast breathing with low-normal pressure
      systemic: fever, or two of (fever, fast heart rate, fast breathing), or admitted to a ward
      mild:     everyone else
    """
    with np.errstate(invalid="ignore"):          # comparisons with NaN are simply False
        fever = temp_q75 >= FEVER_F
        fast_heart = heart > FAST_HEART
        fast_breathing = breathing > FAST_BREATHING
        severe = ((icu == 1) | (sbp_q25 < LOW_SBP)
                  | ((breathing >= VERY_FAST_BREATHING) & (sbp_q25 <= LOW_NORMAL_SBP)))
        signs = fever.astype(int) + fast_heart.astype(int) + fast_breathing.astype(int)
        systemic = fever | (signs >= 2) | ((ward_ip == 1) & ~(ward_er == 1))
    return np.where(severe, 2, np.where(systemic, 1, 0))


def recommend_by_severity(P, levels, bars):
    """The same rule as `recommend`, but each culture's bar is the one for its severity level."""
    max_risk = 1 - np.asarray(bars)[levels]
    return recommend(P, max_risk[:, None])


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
# MAGIC ## 5b. A bar that depends on how sick the patient is
# MAGIC One bar for everyone is too strict for mild infections and too loose for severe ones. If the
# MAGIC first drug fails in a simple bladder infection, the patient has a few more days of symptoms and
# MAGIC switches drug when the culture is back. In a patient with sepsis, a failed first drug can be
# MAGIC fatal. Guidelines accept more risk in the first case to spare broad-spectrum drugs.
# MAGIC
# MAGIC So each culture gets a **severity level**, and each level has its own bar (`SEVERITY_BARS`):
# MAGIC
# MAGIC | Level | Who | Default bar |
# MAGIC |---|---|---|
# MAGIC | **Severe** | Intensive care, or low blood pressure, or very fast breathing with low-normal blood pressure | 95% |
# MAGIC | **Systemic signs** | Fever, or two of (fever, fast heart rate, fast breathing), or admitted to a hospital ward | 90% |
# MAGIC | **Mild** | Everyone else: outpatients and emergency patients with normal or unrecorded vital signs | 85% |
# MAGIC
# MAGIC The vital signs are the ones recorded in the **48 hours before** the culture was ordered, so
# MAGIC nothing here looks into the future. The model's predictions don't change; only the bar does.
# MAGIC
# MAGIC **Read these results with three cautions:**
# MAGIC - **"Mild" is a rough stand-in.** The dataset has no symptoms, so the rule can't tell a bladder
# MAGIC   infection from a kidney infection in a patient with normal vital signs. In the app, the level
# MAGIC   is a default that the clinician can change.
# MAGIC - **The care setting does most of the work.** Fever and low blood pressure are uncommon in this
# MAGIC   data, so vital signs change the level for only a few percent of cultures.
# MAGIC - **A lower bar means more failures, on purpose.** For mild patients, compare against giving
# MAGIC   everyone the narrowest drug (what guidelines suggest), not against giving everyone a
# MAGIC   broad-spectrum drug that no doctor would use for a simple infection.

# COMMAND ----------

from pyspark.sql import functions as F

# Care setting and vital signs for every culture (vital signs: the 48 hours before the order)
culture_settings = (
    spark.table(tbl("gold_urine_features"))
    .select("order_proc_id_coded", "ward_ip", "ward_er", "ward_icu")
    .distinct()
)
vital_signs = spark.table(tbl("bronze_vitals")).groupBy("order_proc_id_coded").agg(
    F.max(F.col("q75_temp").cast("double")).alias("temp_q75"),
    F.max(F.col("median_heartrate").cast("double")).alias("heart"),
    F.max(F.col("median_resprate").cast("double")).alias("breathing"),
    F.max(F.col("q25_sysbp").cast("double")).alias("sbp_q25"),
)
severity_inputs = culture_settings.join(vital_signs, "order_proc_id_coded", "left")

severity_table = severity_inputs.toPandas().set_index("order_proc_id_coded")
SEVERITY_COLUMNS = ["ward_icu", "ward_ip", "ward_er", "temp_q75", "heart", "breathing", "sbp_q25"]
CARE_SETTINGS = ["Outpatient", "Emergency department", "Hospital ward", "Intensive care"]


def levels_for(preds):
    """(severity level, care setting) for each culture that grids() kept, in the same order."""
    kept = (preds.pivot(index="order_proc_id_coded", columns="antibiotic_slug", values="resistant")[DRUGS]
            .dropna().index)
    rows = severity_table.reindex(kept)
    levels = severity_level(*[rows[c].to_numpy(dtype=float) for c in SEVERITY_COLUMNS])
    setting = np.select(
        [(rows["ward_icu"] == 1).to_numpy(), (rows["ward_er"] == 1).to_numpy(), (rows["ward_ip"] == 1).to_numpy()],
        ["Intensive care", "Emergency department", "Hospital ward"], "Outpatient")
    return levels, setting


levels, setting = levels_for(test_preds)
levels_val, _ = levels_for(val_preds)

display(pd.DataFrame([
    {"severity": name, "bar": f"{SEVERITY_BARS[i]:.0%}", "test_cultures": int((levels == i).sum()),
     "pct_of_test": round(100 * float((levels == i).mean()), 1),
     **{place: int(((levels == i) & (setting == place)).sum()) for place in CARE_SETTINGS}}
    for i, name in enumerate(SEVERITY_LEVELS)
]))

# COMMAND ----------

# MAGIC %md
# MAGIC **Choosing the mild bar (on the 2021 validation cultures, not the test years).** Each row is one
# MAGIC option for the mild patients. The first two give everyone the same drug: the narrowest one, and
# MAGIC the antibiogram's pick. The rest are the model at different bars. A lower bar gives more
# MAGIC narrow-spectrum drugs and more failures. `SEVERITY_BARS` holds the bar the team settled on.

# COMMAND ----------

def everyone(drug_position, mask):
    """Recommendations that give every culture in `mask` the same drug."""
    return np.full(int(mask.sum()), drug_position)


mild_val = levels_val == 0
display(pd.DataFrame(
    [{"policy": f"{DRUG_LABEL[DRUGS[0]]} for everyone (narrowest drug)",
      **policy_outcomes(everyone(0, mild_val), R_val[mild_val], DRUGS)},
     {"policy": f"{DRUG_LABEL[DRUGS[antibiogram_drug]]} for everyone (antibiogram's drug)",
      **policy_outcomes(everyone(antibiogram_drug, mild_val), R_val[mild_val], DRUGS)}]
    + [{"policy": f"model at {bar:.1%} bar",
        **policy_outcomes(recommend(P_model_val[mild_val], 1 - bar), R_val[mild_val], DRUGS)}
       for bar in MILD_BARS_TO_COMPARE]
))

# COMMAND ----------

# MAGIC %md
# MAGIC **The result on the test cultures.** For each severity level, and for all patients together:
# MAGIC the narrowest drug for everyone, the antibiogram's drug for everyone, the model with one bar for
# MAGIC everyone, and the model with the bar set by severity. The second table shows which drugs the
# MAGIC severity-based policy recommends at each level.

# COMMAND ----------

choice_severity = recommend_by_severity(P_model, levels, SEVERITY_BARS)

severity_rows = []
for i, name in list(enumerate(SEVERITY_LEVELS)) + [(None, "All patients")]:
    mask = np.ones(len(R), dtype=bool) if i is None else levels == i
    if not mask.any():
        continue
    bar = "set by severity" if i is None else f"{SEVERITY_BARS[i]:.0%}"
    for policy, choice in [
        (f"{DRUG_LABEL[DRUGS[0]]} for everyone (narrowest drug)", everyone(0, mask)),
        (f"{DRUG_LABEL[DRUGS[antibiogram_drug]]} for everyone (antibiogram's drug)",
         everyone(antibiogram_drug, mask)),
        (f"model, {TARGET_SUCCESS:.0%} bar for everyone", choice_model[mask]),
        (f"model, bar {bar}", choice_severity[mask]),
    ]:
        severity_rows.append({"severity": name, "cultures": int(mask.sum()), "policy": policy,
                              **policy_outcomes(choice, R[mask], DRUGS)})
severity_results = pd.DataFrame(severity_rows)
display(severity_results)

display(pd.DataFrame({
    "antibiotic": [DRUG_LABEL[d] for d in DRUGS],
    **{f"{name.lower().replace(' ', '_')}_pct": list(drug_mix(choice_severity[levels == i], DRUGS).values())
       for i, name in enumerate(SEVERITY_LEVELS)},
}))

mild = levels == 0
mild_model = policy_outcomes(choice_severity[mild], R[mild], DRUGS)
mild_narrowest = policy_outcomes(everyone(0, mild), R[mild], DRUGS)
SEVERITY_HEADLINE = (
    f"For the {100 * float(mild.mean()):.0f}% of test patients with a mild presentation, the model at "
    f"a {SEVERITY_BARS[0]:.0%} bar recommended a narrow-spectrum antibiotic for "
    f"{mild_model['narrow_pct']}% and its recommendation failed for {mild_model['failure_pct']}%, "
    f"versus {mild_narrowest['failure_pct']}% if every one of them were given "
    f"{DRUG_LABEL[DRUGS[0]].lower()}."
)
print(SEVERITY_HEADLINE)

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
# MAGIC The trade-off table goes to `workspace.armd.gold_policy_tradeoff` and the severity results to
# MAGIC `workspace.armd.gold_severity_policy` (handy for the demo app and slides). The headline numbers
# MAGIC and chart are logged to the same MLflow experiment as notebook 03.

# COMMAND ----------

import mlflow

(
    spark.createDataFrame(tradeoff)
    .write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(tbl("gold_policy_tradeoff"))
)
(
    spark.createDataFrame(severity_results)
    .write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(tbl("gold_severity_policy"))
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
    # section 5b: the bar set by severity
    mlflow.log_params({f"severity_bar_{name.lower().replace(' ', '_')}": bar
                       for name, bar in zip(SEVERITY_LEVELS, SEVERITY_BARS)})
    for row in severity_rows:
        if row["policy"].startswith("model, bar"):
            level = row["severity"].lower().replace(" ", "_")
            mlflow.log_metric(f"severity.{level}.cultures", row["cultures"])
            for k in ("failure_pct", "narrow_pct", "broad_pct", "carbapenem_pct"):
                mlflow.log_metric(f"severity.{level}.{k}", row[k])
    mlflow.set_tag("severity_headline", SEVERITY_HEADLINE)
    mlflow.log_table(severity_results, artifact_file="severity_policy.json")
    mlflow.log_figure(fig, "tradeoff_chart.png")
    mlflow.log_table(tradeoff, artifact_file="tradeoff.json")

print(f"Saved {tbl('gold_policy_tradeoff')} and {tbl('gold_severity_policy')}, and logged the run to MLflow")
