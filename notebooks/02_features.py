# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Features: build the training table
# MAGIC
# MAGIC **The question the model will answer:** a doctor has just ordered a urine culture. Before the
# MAGIC lab result comes back, how likely is it that the bacteria will be **resistant** to each common
# MAGIC antibiotic?
# MAGIC
# MAGIC **What this notebook builds:** one row per *(urine culture, antibiotic)* pair.
# MAGIC - **Label** (`resistant`): 1 if the bacteria turned out resistant to that antibiotic, 0 if
# MAGIC   susceptible.
# MAGIC - **Features**: only what was known **when the culture was ordered**: age, sex, ward,
# MAGIC   antibiotics taken recently (by class, subtype and specific drug, and how many days ago), past
# MAGIC   resistant infections (and how long ago), past organisms, recent procedures, nursing-home
# MAGIC   contact, recent labs and vital signs. Long-term diagnoses (comorbidities) are optional and off by
# MAGIC   default (see `USE_COMORBIDITIES`).
# MAGIC - **Split**: older cultures are for training, the most recent years are held back for testing.
# MAGIC
# MAGIC ### The one rule that matters most: no data leakage
# MAGIC A feature must never contain information from after the culture was ordered (for example,
# MAGIC which organism grew). If it does, the model looks brilliant in testing and is useless in real
# MAGIC life. Every history feature below only uses events **at least 1 day before** the culture
# MAGIC (3 days for past lab results, which take that long to come back).
# MAGIC
# MAGIC **Input:** the `bronze_*` tables from notebook 01 (and `bronze_comorbidity` from 01b, optional).
# MAGIC **Output:** `workspace.armd.gold_urine_features`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings
# MAGIC These are the choices you may want to change. Everything else follows from them.

# COMMAND ----------

import re
from functools import reduce
from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "armd"


def tbl(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"


# Antibiotics the model will score, roughly narrowest to broadest spectrum.
# Notebook 04 recommends the first one in this list that is likely enough to work.
TARGET_ANTIBIOTICS = [
    "Nitrofurantoin",
    "Trimethoprim/Sulfamethoxazole",
    "Cefazolin",
    "Ciprofloxacin",
    "Ceftriaxone",
    "Piperacillin/Tazobactam",
    "Meropenem",
]

MIN_YEAR = 2008          # data before 2008 is very sparse (a few thousand cultures a year)
TEST_START_YEAR = 2022   # train on 2008-2021, test on 2022 onwards

RECENT_DAYS = 90         # window for "recent" antibiotics and procedures
HISTORY_DAYS = 365       # window for past organisms and past resistance
LOOKBACK_DAYS = 730      # how far back the "days since ..." features look (2 years)

# Individual antibiotics worth tracking on their own. The class flags lump very different drugs
# together (cephalexin and ceftriaxone are both "Beta Lactam"), but resistance follows the specific
# drug. Each entry: feature name -> (text the medication name may contain, text that rules it out).
# Matching is case-insensitive. Ear/eye drops and skin creams are ruled out: they don't drive
# resistance in the urinary tract the way tablets and IV drugs do.
NOT_SYSTEMIC = ["dexamethasone", "silver sulfadiazine"]
SPECIFIC_ANTIBIOTICS = {
    "nitrofurantoin":          (["nitrofurantoin", "macrobid", "macrodantin"], []),
    "trimethoprim":            (["sulfamethoxazole", "bactrim", "trimethoprim"], []),
    "cephalexin":              (["cephalexin", "keflex", "cefadroxil"], []),
    "cefazolin":               (["cefazolin"], []),
    "oral_cephalosporin_gen3": (["cefpodoxime", "cefdinir"], []),
    "ceftriaxone":             (["ceftriaxone"], []),
    "cefepime":                (["cefepime"], []),
    "ciprofloxacin":           (["cipro"], []),
    "levofloxacin":            (["levofloxacin", "levaquin"], []),
    "amoxicillin_clavulanate": (["clavulan", "augmentin"], []),
    "amoxicillin_ampicillin":  (["amoxicillin", "ampicillin"], ["clavulan"]),
    "piperacillin_tazobactam": (["piperacillin"], []),
    "carbapenem":              (["meropenem", "ertapenem"], []),
}

# Comorbidities (diagnoses such as diabetes or kidney disease) need the extra table from notebook
# 01b. If that table doesn't exist, this notebook skips them and still runs.
# OFF by default: on the 2022+ test years they raised gradient boosting's mean AUROC by only
# ~0.001 (0.736 -> 0.737), while adding 61 inputs the demo app would have to fill. The medication
# and resistance history already carries that information. Set to True to bring them back.
USE_COMORBIDITIES = False
TOP_COMORBIDITIES = 60   # keep the N most common diagnoses in the training years as yes/no flags
# Diagnosis groups that are never used as features. "Invalid PDX" is a billing artifact (an invalid
# principal diagnosis code), not a medical condition.
EXCLUDE_COMORBIDITIES = ["Invalid PDX"]

# Lab results that count as "the drug will not work reliably" -> resistant = 1.
# Intermediate results are counted as resistant because a doctor wouldn't rely on that drug.
NOT_SUSCEPTIBLE = ["resistant", "intermediate", "non susceptible", "susceptible - dose dependent"]
SUSCEPTIBLE = ["susceptible"]
# Anything else ("Inconclusive", "See Comment", missing...) is dropped.

# Intrinsic resistance that the dataset's own implied-susceptibility rules leave out. Labs don't
# test these combinations because the organism is always resistant, so without a rule the result
# is simply missing. Each entry: (text the organism name must contain, drugs it always resists).
# The dataset's rules already mark Pseudomonas aeruginosa as resistant to cefazolin and
# ceftriaxone; these two complete the picture. Have a clinician confirm any rule added here.
EXTRA_INTRINSIC_RESISTANCE = [
    ("PSEUDOMONAS AERUGINOSA", ["Nitrofurantoin", "Trimethoprim/Sulfamethoxazole"]),
]


def slug(text: str) -> str:
    """'Piperacillin/Tazobactam' -> 'piperacillin_tazobactam' (safe for column names)."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def days(col_name: str):
    """The 'days before the culture' columns were loaded as text; make them numbers."""
    return F.col(col_name).cast("int")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pick the cultures: positive urine cultures from 2008 onwards
# MAGIC The cohort table has one row per *(culture, organism, antibiotic tested)*, so we collapse it to
# MAGIC one row per culture order. Negative cultures are dropped: if nothing grew, there is no
# MAGIC resistance to predict.

# COMMAND ----------

orders = (
    spark.table(tbl("bronze_cohort"))
    .where((F.col("culture_description") == "URINE") & (F.col("was_positive") == "1"))
    .groupBy("order_proc_id_coded")
    .agg(
        F.first("anon_id", ignorenulls=True).alias("anon_id"),
        F.first("order_time_jittered_utc", ignorenulls=True).alias("order_time"),
        F.first("ordering_mode", ignorenulls=True).alias("ordering_mode"),
    )
    .withColumn("order_year", F.substring("order_time", 1, 4).cast("int"))
    .where(F.col("order_year") >= MIN_YEAR)
    .withColumn("split", F.when(F.col("order_year") >= TEST_START_YEAR, "test").otherwise("train"))
    .withColumn("is_inpatient", (F.col("ordering_mode") == "Inpatient").cast("int"))
)

print(f"Positive urine cultures since {MIN_YEAR}: {orders.count():,}")
display(orders.groupBy("split").count().orderBy("split"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build the labels: was the infection resistant to each antibiotic?
# MAGIC
# MAGIC Results come from two tables plus a short list of our own rules, and we need all three:
# MAGIC - **cohort**: the results the lab actually reported.
# MAGIC - **implied susceptibility**: extra results the lab didn't report but that follow from standard
# MAGIC   rules (for example, an organism that is naturally resistant to a drug). This table is *not* a
# MAGIC   full copy of the lab results: it has only ~8,000 trimethoprim/sulfamethoxazole rows where the
# MAGIC   cohort has ~103,000, so using it alone would almost wipe out that drug.
# MAGIC - **extra intrinsic-resistance rules** (`EXTRA_INTRINSIC_RESISTANCE` in the settings): cases the
# MAGIC   dataset's rules miss. Pseudomonas aeruginosa is always resistant to nitrofurantoin and
# MAGIC   trimethoprim/sulfamethoxazole, so the lab never tests them and no result exists. Filling them
# MAGIC   in as resistant gives about 3,800 more labelled rows, and means a culture that grew
# MAGIC   Pseudomonas alongside another organism correctly counts as resistant to those two drugs.
# MAGIC
# MAGIC Some cultures grow more than one organism. The drug only works if it works on **all** of them,
# MAGIC so the culture counts as resistant if **any** organism was resistant.

# COMMAND ----------

abx_lookup = spark.createDataFrame(
    [(name.lower(), name, slug(name), rank) for rank, name in enumerate(TARGET_ANTIBIOTICS)],
    ["abx_lower", "antibiotic", "antibiotic_slug", "spectrum_rank"],
)

lab_results = spark.table(tbl("bronze_cohort")).select(
    "order_proc_id_coded",
    "organism",
    F.lower(F.trim("antibiotic")).alias("abx_lower"),
    F.lower(F.trim("susceptibility")).alias("result"),
)
implied_results = spark.table(tbl("bronze_implied_susceptibility")).select(
    "order_proc_id_coded",
    "organism",
    F.lower(F.trim("antibiotic")).alias("abx_lower"),
    F.lower(F.trim(F.coalesce("susceptibility", "implied_susceptibility"))).alias("result"),
)

# One "resistant" row per (culture, organism, drug) covered by an extra intrinsic-resistance rule
organisms_grown = (
    spark.table(tbl("bronze_cohort"))
    .where(F.col("organism").isNotNull())
    .select("order_proc_id_coded", "organism")
    .distinct()
)
rule_results = reduce(
    lambda a, b: a.unionByName(b),
    [
        organisms_grown.where(F.upper(F.col("organism")).like(f"%{pattern.upper()}%")).select(
            "order_proc_id_coded", "organism",
            F.lit(drug.lower()).alias("abx_lower"), F.lit("resistant").alias("result"),
        )
        for pattern, drugs in EXTRA_INTRINSIC_RESISTANCE
        for drug in drugs
    ],
)

labels = (
    lab_results.unionByName(implied_results).unionByName(rule_results)
    .join(F.broadcast(abx_lookup.select("abx_lower")), "abx_lower")         # keep target drugs only
    .join(orders.select("order_proc_id_coded"), "order_proc_id_coded")      # keep our cultures only
    .withColumn(
        "resistant",
        F.when(F.col("result").isin(NOT_SUSCEPTIBLE), 1).when(F.col("result").isin(SUSCEPTIBLE), 0),
    )
    .where(F.col("resistant").isNotNull())
    .groupBy("order_proc_id_coded", "abx_lower")
    .agg(
        F.max("resistant").alias("resistant"),              # resistant if ANY organism was
        F.countDistinct("organism").alias("n_organisms_tested"),
    )
    .join(F.broadcast(abx_lookup), "abx_lower")
    .drop("abx_lower")
)

display(
    labels.groupBy("spectrum_rank", "antibiotic")
    .agg(F.count("*").alias("cultures"), F.round(100 * F.avg("resistant"), 1).alias("pct_resistant"))
    .orderBy("spectrum_rank")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Features about the patient and the visit
# MAGIC These tables have exactly one row per culture order.
# MAGIC - **Age** comes in bins ("25-34 years"); we keep the lower end of the bin (25). "above 90" -> 90.
# MAGIC - **Gender** is 0/1 in the source, and the dataset deliberately doesn't say which is which.
# MAGIC - **Ward**: inpatient, outpatient, emergency room, ICU.
# MAGIC - **ADI**: Area Deprivation Index, a neighbourhood-level measure of socioeconomic disadvantage.

# COMMAND ----------

demographics = spark.table(tbl("bronze_demographics")).select(
    "order_proc_id_coded",
    F.regexp_extract("age", r"(\d+)", 1).cast("int").alias("age_min"),
    F.col("gender").cast("int").alias("gender"),
)

ward = spark.table(tbl("bronze_ward_info")).select(
    "order_proc_id_coded",
    *[F.col(f"hosp_ward_{w}").cast("int").alias(f"ward_{w}") for w in ["ip", "op", "er", "icu"]],
)

adi = spark.table(tbl("bronze_adi_scores")).select(
    "order_proc_id_coded",
    F.col("adi_score").cast("double").alias("adi_score"),
    F.col("adi_state_rank").cast("double").alias("adi_state_rank"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Recent lab results and vital signs
# MAGIC The source has several rows per culture for these, but the **median** columns are the same on
# MAGIC every row, so we keep one value per culture. Missing values stay missing (many patients,
# MAGIC especially outpatients, have no recent labs). The models in notebook 03 handle that.
# MAGIC
# MAGIC ⚠️ **To double-check as a team:** the dataset README says labs come from a 14-day window "from
# MAGIC the culture order" but doesn't say whether that window can include days after the order. Same-day
# MAGIC values were removed by the dataset authors. Notebook 03 will compare models with and without
# MAGIC labs and vitals so we can see whether they matter.

# COMMAND ----------

LAB_MEDIANS = ["wbc", "neutrophils", "lymphocytes", "hgb", "plt", "na", "hco3", "bun", "cr",
               "lactate", "procalcitonin"]
VITAL_MEDIANS = ["heartrate", "resprate", "temp", "sysbp", "diasbp"]

labs = (
    spark.table(tbl("bronze_labs"))
    .groupBy("order_proc_id_coded")
    .agg(*[F.max(F.col(f"median_{x}").cast("double")).alias(f"lab_{x}") for x in LAB_MEDIANS])
    # hemoglobin is stored at 1000x the usual g/dL value (median 12000 -> 12.0 g/dL)
    .withColumn("lab_hgb", F.col("lab_hgb") / 1000)
)

vitals = (
    spark.table(tbl("bronze_vitals"))
    .groupBy("order_proc_id_coded")
    .agg(*[F.max(F.col(f"median_{x}").cast("double")).alias(f"vital_{x}") for x in VITAL_MEDIANS])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Patient history: yes/no flags for past events
# MAGIC Each of these turns a list of past events into 0/1 columns, e.g. `abx_class_fluoroquinolone_90d
# MAGIC = 1` means the patient took a fluoroquinolone in the 90 days before the culture.
# MAGIC
# MAGIC **Leakage guard:** every flag only counts events at least 1 day before the culture
# MAGIC (`min_days=1`). Past *resistance* needs 3 days, because lab results take that long to come back.

# COMMAND ----------

def history_flags(table, value_col, day_col, values, prefix, max_days, min_days=1, suffix=""):
    """One 0/1 column per value: did this event happen between min_days and max_days before the culture?"""
    df = spark.table(tbl(table)).withColumn("_days", days(day_col))
    df = df.where((F.col("_days") >= min_days) & (F.col("_days") <= max_days))
    return df.groupBy("order_proc_id_coded").agg(*[
        F.max(F.when(F.col(value_col) == v, 1).otherwise(0)).alias(f"{prefix}_{slug(v)}{suffix}")
        for v in values
    ])


ANTIBIOTIC_CLASSES = [
    "Beta Lactam", "Fluoroquinolone", "Macrolide Lincosamide", "Combination Antibiotic",
    "Nitroimidazole", "Nitrofuran", "Glycopeptide", "Tetracycline", "Aminoglycoside", "Ansamycin",
    "Oxazolidinone", "Antitubercular", "Sulfonamide", "Urinary Antiseptic", "Fosfomycin",
    "Monobactam", "Folate Synthesis Inhibitor",
]
PRIOR_ORGANISMS = [
    "Escherichia", "Staphylococcus", "Pseudomonas", "Enterococcus", "Klebsiella", "Streptococcus",
    "Proteus", "CONS", "Stenotrophomonas", "Enterobacter", "Citrobacter", "Candida", "Serratia",
    "Morganella", "Acinetobacter", "Providencia",
]
PROCEDURES = ["urethral_catheter", "cvc", "dialysis", "mechvent", "parenteral_nutrition",
              "surgical_procedure"]

# Antibiotics taken in the last 90 days, by class
abx_exposure = history_flags("bronze_antibiotic_class_exposure", "antibiotic_class",
                             "time_to_culturetime", ANTIBIOTIC_CLASSES, "abx_class",
                             RECENT_DAYS, suffix=f"_{RECENT_DAYS}d")

# Days since the most recent antibiotic of any kind (within a year)
abx_recency = (
    spark.table(tbl("bronze_antibiotic_class_exposure"))
    .withColumn("_days", days("time_to_culturetime"))
    .where((F.col("_days") >= 1) & (F.col("_days") <= HISTORY_DAYS))
    .groupBy("order_proc_id_coded")
    .agg(F.min("_days").alias("days_since_last_antibiotic"))
)

# Organisms found in this patient's earlier cultures (last year)
prior_organisms = history_flags("bronze_prior_infecting_organism", "prior_organism",
                                "prior_infecting_organism_days_to_culture", PRIOR_ORGANISMS,
                                "prior_org", HISTORY_DAYS)

# Earlier resistance to each target antibiotic (last year) - usually the strongest signal.
# The dataset's own resistance-history table skips trimethoprim/sulfamethoxazole and
# piperacillin/tazobactam, so we work it out from the patient's earlier cultures instead.
# An earlier culture only counts if it was ordered at least PRIOR_RESULT_MIN_DAYS before: its
# susceptibility results take ~2-3 days, so a doctor couldn't have seen anything newer.
PRIOR_RESULT_MIN_DAYS = 3


def order_timestamp(col_name):
    # "2023-12-23 22:29:00+00:00" -> timestamp (the first 19 characters are the date and time)
    return F.to_timestamp(F.substring(col_name, 1, 19))


past_resistant_results = (
    spark.table(tbl("bronze_cohort"))
    .where(F.lower(F.trim("susceptibility")).isin(NOT_SUSCEPTIBLE))
    .select(
        "anon_id",
        order_timestamp("order_time_jittered_utc").alias("past_time"),
        F.lower(F.trim("antibiotic")).alias("past_abx"),
    )
    .distinct()
)

prior_resistance = (
    orders.select("order_proc_id_coded", "anon_id", order_timestamp("order_time").alias("order_ts"))
    .join(past_resistant_results, "anon_id")
    .withColumn("_days", F.datediff("order_ts", "past_time"))
    .where((F.col("_days") >= PRIOR_RESULT_MIN_DAYS) & (F.col("_days") <= HISTORY_DAYS))
    .groupBy("order_proc_id_coded")
    .agg(
        *[F.max(F.when(F.col("past_abx") == name.lower(), 1).otherwise(0)).alias(f"prior_resist_{slug(name)}")
          for name in TARGET_ANTIBIOTICS],
        F.countDistinct("past_abx").alias("n_prior_resistant_drugs"),
    )
)

# Procedures in the last 90 days (catheters matter a lot for urinary infections)
procedures = history_flags("bronze_prior_procedures", "procedure_description",
                           "procedure_time_to_culturetime", PROCEDURES, "proc", RECENT_DAYS)

# Any nursing-home visit on record. This one allows day 0: living in or coming from a nursing
# home is known at the time of the order.
nursing_home = (
    spark.table(tbl("bronze_nursing_home_visits"))
    .select("order_proc_id_coded").distinct()
    .withColumn("nursing_home", F.lit(1))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6b. Patient history: how long ago, and which specific drug
# MAGIC The flags above only say *whether* something happened in a fixed window. These say **how many
# MAGIC days ago** it last happened (up to 2 years; "never" is stored as 731). One number covers every
# MAGIC window at once: gradient boosting picks the cut-offs that matter (7 days, 30, 180...) by itself,
# MAGIC instead of us guessing one.
# MAGIC
# MAGIC - `days_since_drug_<name>`: the last dose of a **specific** antibiotic (see
# MAGIC   `SPECIFIC_ANTIBIOTICS` in the settings), e.g. ciprofloxacin rather than "any fluoroquinolone".
# MAGIC - `days_since_subtype_<name>`: the last dose of each antibiotic **subtype** (e.g. 1st- vs
# MAGIC   3rd-generation cephalosporin, carbapenem), which is finer than the class.
# MAGIC - `days_since_resist_<drug>`: the last culture that was resistant to each target drug.
# MAGIC - `n_antibiotic_days_365d` / `n_prior_cultures_365d`: how many different days in the last year
# MAGIC   the patient was on an antibiotic, or had an infecting organism found.
# MAGIC
# MAGIC Same leakage guards as before: at least 1 day before the culture, 3 days for past results.

# COMMAND ----------

NEVER = LOOKBACK_DAYS + 1


def days_since(df, groups, prefix, min_days=1, max_days=LOOKBACK_DAYS):
    """One column per group: days since the most recent matching row between min_days and
    max_days before the culture. `df` needs order_proc_id_coded and _days; `groups` maps
    name -> condition. No matching row -> missing (filled with NEVER in section 7)."""
    df = df.where((F.col("_days") >= min_days) & (F.col("_days") <= max_days))
    return df.groupBy("order_proc_id_coded").agg(*[
        F.min(F.when(condition, F.col("_days"))).alias(f"{prefix}_{name}")
        for name, condition in groups.items()
    ])


def name_matches(col, include, exclude):
    """Case-insensitive: the name contains any of `include` and none of `exclude`."""
    lower = F.lower(F.col(col))
    hit = reduce(lambda a, b: a | b, [lower.like(f"%{t}%") for t in include])
    for t in exclude + NOT_SYSTEMIC:
        hit = hit & ~lower.like(f"%{t}%")
    return hit


abx_events = spark.table(tbl("bronze_antibiotic_class_exposure")).withColumn(
    "_days", days("time_to_culturetime"))

specific_drug_recency = days_since(
    abx_events,
    {name: name_matches("medication_name", inc, exc) for name, (inc, exc) in SPECIFIC_ANTIBIOTICS.items()},
    "days_since_drug",
)

subtype_events = spark.table(tbl("bronze_antibiotic_subtype_exposure")).withColumn(
    "_days", days("medication_time_to_culturetime"))
SUBTYPES = sorted(r[0] for r in subtype_events.select("antibiotic_subtype").distinct().collect()
                  if r[0] is not None)
subtype_recency = days_since(
    subtype_events,
    {slug(s): F.col("antibiotic_subtype") == s for s in SUBTYPES},
    "days_since_subtype",
)

past_resistance_events = (
    orders.select("order_proc_id_coded", "anon_id", order_timestamp("order_time").alias("order_ts"))
    .join(past_resistant_results, "anon_id")
    .withColumn("_days", F.datediff("order_ts", "past_time"))
)
resistance_recency = days_since(
    past_resistance_events,
    {slug(name): F.col("past_abx") == name.lower() for name in TARGET_ANTIBIOTICS},
    "days_since_resist",
    min_days=PRIOR_RESULT_MIN_DAYS,
)

# How much antibiotic, and how many past infections, in the last year
abx_burden = (
    abx_events.where((F.col("_days") >= 1) & (F.col("_days") <= HISTORY_DAYS))
    .groupBy("order_proc_id_coded")
    .agg(F.countDistinct("_days").alias("n_antibiotic_days_365d"))
)
culture_burden = (
    spark.table(tbl("bronze_prior_infecting_organism"))
    .withColumn("_days", days("prior_infecting_organism_days_to_culture"))
    .where((F.col("_days") >= 1) & (F.col("_days") <= HISTORY_DAYS))
    .groupBy("order_proc_id_coded")
    .agg(F.countDistinct("_days").alias("n_prior_cultures_365d"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6c. Comorbidities (off by default; needs notebook 01b)
# MAGIC Skipped unless `USE_COMORBIDITIES = True`. They were tested and added almost nothing (test AUROC
# MAGIC 0.736 -> 0.737), so the final model leaves them out. The code stays so the team can re-check.
# MAGIC
# MAGIC Long-term conditions such as diabetes, kidney disease or a urinary-tract disorder. They come from
# MAGIC `bronze_comorbidity`, one row per (culture, diagnosis) with the day the diagnosis started.
# MAGIC
# MAGIC - Only diagnoses that started **at least 1 day before** the culture count (a diagnosis written
# MAGIC   during the same visit may have been made because of this infection).
# MAGIC - There are hundreds of diagnosis groups, most of them rare. We keep the `TOP_COMORBIDITIES`
# MAGIC   most common ones **in the training years** as `comorb_<name>` flags, plus `n_comorbidities`
# MAGIC   (how many different diagnoses the patient has). Choosing them from training years only keeps
# MAGIC   the test years untouched.
# MAGIC - The table is ~20 GB, so we first save just our urine cultures' rows to a small helper table
# MAGIC   (`silver_urine_comorbidity`) and work from that.

# COMMAND ----------

def table_exists(name):
    try:
        spark.table(tbl(name)).columns       # fails if the table isn't there
        return True
    except Exception:
        return False


comorbidity_tables = []
if USE_COMORBIDITIES and table_exists("bronze_comorbidity"):
    (
        spark.table(tbl("bronze_comorbidity"))
        .join(F.broadcast(orders.select("order_proc_id_coded", "split")), "order_proc_id_coded")
        .select(
            "order_proc_id_coded", "split",
            F.trim("comorbidity_component").alias("component"),
            days("comorbidity_component_start_days_culture").alias("_days"),
        )
        .where(F.col("component").isNotNull() & F.col("_days").isNotNull())
        .write.mode("overwrite").option("overwriteSchema", True)
        .saveAsTable(tbl("silver_urine_comorbidity"))
    )
    comorbidity = spark.table(tbl("silver_urine_comorbidity"))

    # Like every other ARMD file, positive days should mean "before the culture". Stop if not.
    share_negative = comorbidity.select(F.avg((F.col("_days") < 0).cast("int"))).first()[0]
    assert share_negative is not None, "bronze_comorbidity has no rows for our urine cultures"
    print(f"Diagnosis start days below 0 (after the culture): {share_negative:.1%}")
    assert share_negative < 0.5, (
        "Most start days are negative, so this file may count days the other way round. "
        "Check the columns in notebook 01b before going on.")

    before = comorbidity.where((F.col("_days") >= 1) & ~F.col("component").isin(EXCLUDE_COMORBIDITIES))
    top = [r["component"] for r in (
        before.where(F.col("split") == "train")
        .groupBy("component").agg(F.countDistinct("order_proc_id_coded").alias("cultures"))
        .orderBy(F.desc("cultures"), "component")
        .limit(TOP_COMORBIDITIES)
        .collect()
    )]
    # short, unique column names: comorb_<first 50 characters of the name>
    columns = {}
    for name in top:
        base = f"comorb_{slug(name)[:50]}"
        col_name, k = base, 2
        while col_name in columns.values():
            col_name, k = f"{base}_{k}", k + 1
        columns[name] = col_name
    print(f"Comorbidity flags ({len(top)}):", ", ".join(top))

    comorbidity_tables = [
        before.groupBy("order_proc_id_coded").agg(
            *[F.max(F.when(F.col("component") == name, 1).otherwise(0)).alias(col_name)
              for name, col_name in columns.items()],
            F.countDistinct("component").alias("n_comorbidities"),
        )
    ]
else:
    print("No bronze_comorbidity table (or USE_COMORBIDITIES is False): skipping comorbidities.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Put it all together
# MAGIC Every feature table is joined onto the culture list. A patient with no history row simply had no
# MAGIC such event, so missing history flags and counts become 0, and missing "days since" values
# MAGIC become "never" (731). Labs, vitals and ADI stay missing when absent.
# MAGIC
# MAGIC We also add `prior_resist_same_drug`: was this patient's infection resistant to *this row's*
# MAGIC antibiotic in the last year?

# COMMAND ----------

feature_tables = [demographics, ward, adi, labs, vitals, abx_exposure, abx_recency,
                  prior_organisms, prior_resistance, procedures, nursing_home,
                  specific_drug_recency, subtype_recency, resistance_recency, abx_burden,
                  culture_burden, *comorbidity_tables]

order_features = reduce(
    lambda acc, df: acc.join(df, "order_proc_id_coded", "left"),
    feature_tables,
    orders,
)

zero_fill = [c for c in order_features.columns
             if c.startswith(("abx_class_", "prior_org_", "prior_resist_", "proc_", "comorb_", "n_"))
             or c == "nursing_home"]
never_fill = [c for c in order_features.columns
              if c.startswith(("days_since_drug_", "days_since_subtype_", "days_since_resist_"))]
order_features = order_features.select(*[
    F.coalesce(F.col(c), F.lit(0)).alias(c) if c in zero_fill
    else F.coalesce(F.col(c), F.lit(NEVER)).alias(c) if c in never_fill
    else F.coalesce(F.col(c), F.lit(HISTORY_DAYS + 1)).alias(c) if c == "days_since_last_antibiotic"
    else F.col(c)
    for c in order_features.columns
])

same_drug = reduce(
    lambda acc, name: acc.when(F.col("antibiotic_slug") == slug(name), F.col(f"prior_resist_{slug(name)}")),
    TARGET_ANTIBIOTICS[1:],
    F.when(F.col("antibiotic_slug") == slug(TARGET_ANTIBIOTICS[0]),
           F.col(f"prior_resist_{slug(TARGET_ANTIBIOTICS[0])}")),
)

gold = (
    labels.join(order_features, "order_proc_id_coded", "inner")
    .withColumn("prior_resist_same_drug", same_drug)
)

(
    gold.write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(tbl("gold_urine_features"))
)

gold = spark.table(tbl("gold_urine_features"))
print(f"Saved {tbl('gold_urine_features')}: {gold.count():,} rows, {len(gold.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sanity checks
# MAGIC Look at these before moving on to notebook 03.
# MAGIC
# MAGIC **a) One row per culture and antibiotic.** If this fails, a join duplicated rows.

# COMMAND ----------

total = gold.count()
distinct_pairs = gold.select("order_proc_id_coded", "antibiotic").distinct().count()
assert total == distinct_pairs, f"Duplicate rows: {total:,} rows but {distinct_pairs:,} unique pairs"
print(f"OK: {total:,} rows, each a unique (culture, antibiotic) pair")

# COMMAND ----------

# MAGIC %md
# MAGIC **b) Resistance rates by antibiotic and split.** A full run of this notebook gave these overall
# MAGIC rates: nitrofurantoin ~23%, trimethoprim/sulfamethoxazole ~27%, cefazolin ~21%,
# MAGIC ciprofloxacin ~21%, ceftriaxone ~15%, piperacillin/tazobactam ~4.5%, meropenem ~0.5%. Each
# MAGIC drug should be within a point or two of these; a value far off means something is wrong.
# MAGIC
# MAGIC The test years may be a little higher or lower than training. That drift over time is real
# MAGIC and is exactly why we test on later years.

# COMMAND ----------

display(
    gold.groupBy("spectrum_rank", "antibiotic", "split")
    .agg(F.count("*").alias("rows"), F.round(100 * F.avg("resistant"), 1).alias("pct_resistant"))
    .orderBy("spectrum_rank", "split")
)

# COMMAND ----------

# MAGIC %md
# MAGIC **c) How often each feature is missing or switched on.** Labs and vitals will be missing for
# MAGIC many outpatients; that's expected. A history flag that is *never* 1 is useless and can be dropped.

# COMMAND ----------

feature_cols = [c for c in gold.columns if c not in {
    "order_proc_id_coded", "anon_id", "order_time", "ordering_mode", "order_year", "split",
    "antibiotic", "antibiotic_slug", "spectrum_rank", "resistant", "n_organisms_tested"}]

summary = gold.select(
    *[F.round(100 * F.avg(F.col(c).isNull().cast("int")), 1).alias(c) for c in feature_cols]
).first().asDict()
means = gold.select(*[F.round(F.avg(c), 3).alias(c) for c in feature_cols]).first().asDict()

display(spark.createDataFrame(
    [(c, summary[c], means[c]) for c in feature_cols],
    "feature string, pct_missing double, mean double",
))

# COMMAND ----------

# MAGIC %md
# MAGIC **d) Quick signal check.** Patients who were resistant to a drug in the past year should be
# MAGIC resistant again far more often. If `prior_resist_same_drug = 1` doesn't show a much higher rate,
# MAGIC the history join is broken.

# COMMAND ----------

display(
    gold.groupBy("antibiotic", "prior_resist_same_drug")
    .agg(F.count("*").alias("rows"), F.round(100 * F.avg("resistant"), 1).alias("pct_resistant"))
    .orderBy("antibiotic", "prior_resist_same_drug")
)

# COMMAND ----------

# MAGIC %md
# MAGIC **e) Patients in both splits.** The split is by date, so a patient can have older cultures in
# MAGIC training and newer ones in testing. That matches real use (the hospital has seen the patient
# MAGIC before), but it's worth knowing how common it is.

# COMMAND ----------

train_patients = gold.where("split = 'train'").select("anon_id").distinct()
test_patients = gold.where("split = 'test'").select("anon_id").distinct()
overlap = test_patients.join(train_patients, "anon_id").count()
print(f"Test patients also seen in training: {overlap:,} of {test_patients.count():,}")
