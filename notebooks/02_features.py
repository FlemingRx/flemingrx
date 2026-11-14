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
# MAGIC   antibiotics taken recently, past resistant infections, past organisms, recent procedures,
# MAGIC   nursing-home contact, recent labs and vital signs.
# MAGIC - **Split**: older cultures are for training, the most recent years are held back for testing.
# MAGIC
# MAGIC ### The one rule that matters most: no data leakage
# MAGIC A feature must never contain information from after the culture was ordered (for example,
# MAGIC which organism grew). If it does, the model looks brilliant in testing and is useless in real
# MAGIC life. Every history feature below only uses events **at least 1 day before** the culture
# MAGIC (3 days for past lab results, which take that long to come back).
# MAGIC
# MAGIC **Input:** the `bronze_*` tables from notebook 01.
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

# Lab results that count as "the drug will not work reliably" -> resistant = 1.
# Intermediate results are counted as resistant because a doctor wouldn't rely on that drug.
NOT_SUSCEPTIBLE = ["resistant", "intermediate", "non susceptible", "susceptible - dose dependent"]
SUSCEPTIBLE = ["susceptible"]
# Anything else ("Inconclusive", "See Comment", missing...) is dropped.


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
# MAGIC Results come from two tables, and we need both:
# MAGIC - **cohort**: the results the lab actually reported.
# MAGIC - **implied susceptibility**: extra results the lab didn't report but that follow from standard
# MAGIC   rules (for example, an organism that is naturally resistant to a drug). This table is *not* a
# MAGIC   full copy of the lab results: it has only ~8,000 trimethoprim/sulfamethoxazole rows where the
# MAGIC   cohort has ~103,000, so using it alone would almost wipe out that drug.
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

labels = (
    lab_results.unionByName(implied_results)
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
# MAGIC ## 7. Put it all together
# MAGIC Every feature table is joined onto the culture list. A patient with no history row simply had no
# MAGIC such event, so missing history flags become 0. Labs, vitals and ADI stay missing when absent.
# MAGIC
# MAGIC We also add `prior_resist_same_drug`: was this patient's infection resistant to *this row's*
# MAGIC antibiotic in the last year?

# COMMAND ----------

feature_tables = [demographics, ward, adi, labs, vitals, abx_exposure, abx_recency,
                  prior_organisms, prior_resistance, procedures, nursing_home]

order_features = reduce(
    lambda acc, df: acc.join(df, "order_proc_id_coded", "left"),
    feature_tables,
    orders,
)

zero_fill = [c for c in order_features.columns
             if c.startswith(("abx_class_", "prior_org_", "prior_resist_", "proc_"))
             or c in ("nursing_home", "n_prior_resistant_drugs")]
order_features = order_features.select(*[
    F.coalesce(F.col(c), F.lit(0)).alias(c) if c in zero_fill
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
# MAGIC **b) Resistance rates by antibiotic and split.** When we profiled the raw data, urine resistance
# MAGIC was roughly: nitrofurantoin 20%, trimethoprim/sulfamethoxazole 24%, cefazolin 18%,
# MAGIC ciprofloxacin 20%, ceftriaxone 11%, piperacillin/tazobactam 4%, meropenem under 1%. Numbers here
# MAGIC will differ a bit (implied results are included), but a value far off means something is wrong.
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
