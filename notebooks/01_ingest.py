# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest: load the ARMD CSVs into Delta tables
# MAGIC
# MAGIC **What this notebook does:** reads the raw CSV files from the dataset and saves each one as a
# MAGIC Delta table, which is much faster to query than CSV. No analysis happens here.
# MAGIC
# MAGIC **Run this once.** Re-reading 4 GB of CSV uses a lot of the Free Edition daily compute quota.
# MAGIC Every later notebook reads the tables this one creates.
# MAGIC
# MAGIC **Before running:**
# MAGIC 1. Unzip `DRYAD DATA.zip` on your laptop.
# MAGIC 2. In Databricks, open **Catalog**, then `workspace`, and run cell 2 below once to create the
# MAGIC    `armd` schema and the `raw` volume (or create them by hand).
# MAGIC 3. Open the `raw` volume and **Upload to this volume** all 15 CSV files.
# MAGIC
# MAGIC **Output:** 15 tables named `workspace.armd.bronze_<name>`. "Bronze" is the usual name for a
# MAGIC raw copy of the source data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings
# MAGIC Change these only if you named your catalog, schema or volume differently.

# COMMAND ----------

CATALOG = "workspace"   # Free Edition's default catalog
SCHEMA = "armd"         # our project's schema (like a folder of tables)
VOLUME = "raw"          # where the uploaded CSVs live

RAW_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the schema and volume (safe to re-run)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Upload the CSVs to: {RAW_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Check that every file was uploaded
# MAGIC The numbers are the data rows in each file (not counting the header), counted on the original
# MAGIC download. Step 5 compares the loaded tables against them.

# COMMAND ----------

# file name -> (table name, expected row count)
FILES = {
    "microbiology_cultures_cohort.csv":                      ("cohort",                    2_241_050),
    "microbiology_cultures_implied_susceptibility.csv":      ("implied_susceptibility",    1_978_731),
    "implied_susceptibility_rules.csv":                      ("implied_susceptibility_rules", 50),
    "microbiology_cultures_demographics.csv":                ("demographics",                751_075),
    "microbiology_cultures_ward_info.csv":                   ("ward_info",                   751_075),
    "microbiology_cultures_adi_scores.csv":                  ("adi_scores",                  751_075),
    "microbiology_cultures_labs.csv":                        ("labs",                      1_408_677),
    "microbiology_cultures_vitals.csv":                      ("vitals",                    4_591_374),
    "microbiology_cultures_nursing_home_visits.csv":         ("nursing_home_visits",           7_628),
    "microbiology_cultures_priorprocedures.csv":             ("prior_procedures",          1_664_615),
    "microbiology_culture_prior_infecting_organism.csv":     ("prior_infecting_organism",  1_083_739),
    "microbiology_cultures_microbial_resistance.csv":        ("microbial_resistance",      2_161_648),
    "microbiology_cultures_antibiotic_class_exposure.csv":   ("antibiotic_class_exposure", 5_402_486),
    "microbiology_cultures_antibiotic_subtype_exposure.csv": ("antibiotic_subtype_exposure", 5_402_486),
    "microbiology_cultures_prior_med.csv":                   ("prior_med",                 9_823_458),
}

uploaded = {f.name for f in dbutils.fs.ls(RAW_DIR)}
missing = [name for name in FILES if name not in uploaded]
if missing:
    raise FileNotFoundError(f"Not uploaded yet: {missing}")
print(f"All {len(FILES)} files found in {RAW_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load each CSV and save it as a Delta table
# MAGIC
# MAGIC Three clean-ups happen here, because the raw files are a bit inconsistent:
# MAGIC - **Missing values** are written as the text `Null` (sometimes `null`). We turn them into real
# MAGIC   empty values so Spark treats them as missing.
# MAGIC - **Column names** are lower-cased (the files mix `Q75_wbc` and `cultureTime`), and one typo in
# MAGIC   the source (`culutre`) is fixed.
# MAGIC - **Everything is loaded as text.** Guessing types would mean reading 4 GB twice. Notebook 02
# MAGIC   converts only the columns it uses into numbers.
# MAGIC
# MAGIC Some organism and medication names contain commas inside quotes; the `quote`/`escape` options
# MAGIC make Spark read those correctly.

# COMMAND ----------

from pyspark.sql import functions as F

NULL_TOKENS = ["null", ""]


def clean_column_name(name: str) -> str:
    return name.strip().lower().replace("culutre", "culture")


def load_csv(path: str):
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", False)
        .csv(path)
    )
    df = df.toDF(*[clean_column_name(c) for c in df.columns])
    # turn "Null" / "null" / "" into real missing values in every column
    return df.select([
        F.when(F.lower(F.trim(F.col(c))).isin(NULL_TOKENS), None).otherwise(F.col(c)).alias(c)
        for c in df.columns
    ])


for file_name, (table, _expected) in FILES.items():
    full_table = f"{CATALOG}.{SCHEMA}.bronze_{table}"
    print(f"Loading {file_name} -> {full_table}")
    (
        load_csv(f"{RAW_DIR}/{file_name}")
        .write.mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(full_table)
    )

print("Done.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Check the row counts
# MAGIC Every table should say `OK`. A mismatch usually means a file was only partly uploaded:
# MAGIC re-upload that file and re-run step 4.

# COMMAND ----------

rows = []
for file_name, (table, expected) in FILES.items():
    actual = spark.table(f"{CATALOG}.{SCHEMA}.bronze_{table}").count()
    rows.append((table, expected, actual, "OK" if actual == expected else "MISMATCH"))

display(spark.createDataFrame(rows, ["table", "expected_rows", "actual_rows", "status"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Peek at the main table
# MAGIC One row per (culture, organism, antibiotic tested). Negative cultures have no organism.

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.bronze_cohort").limit(20))
