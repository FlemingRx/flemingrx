# Databricks notebook source
# MAGIC %md
# MAGIC # 01b · Ingest the comorbidity file (optional)
# MAGIC
# MAGIC **What this notebook does:** loads the patients' long-term diagnoses (diabetes, kidney disease,
# MAGIC urinary-tract disorders...) into `workspace.armd.bronze_comorbidity`. Notebook 02 turns them into
# MAGIC features. If you skip this notebook, 02 still runs, just without comorbidities.
# MAGIC
# MAGIC **Why it's separate from 01:** the file is ~19.7 GB, too big for the Databricks upload page
# MAGIC (5 GB per file). So it goes through one extra step on your laptop first.
# MAGIC
# MAGIC **Before running:**
# MAGIC 1. Download `microbiology_cultures_comorbidity.csv` on its own from the ARMD Dryad page (it isn't
# MAGIC    in the zip download).
# MAGIC 2. On your laptop, run `local/convert_comorbidity.py` (see the repo). It rewrites the CSV as about
# MAGIC    1 GB Parquet files without changing any data, and prints the row count. Write that number
# MAGIC    down.
# MAGIC 3. In Databricks, open **Catalog**, then `workspace`, `armd`, `raw`. Create a folder called
# MAGIC    `comorbidity` inside the volume, open it, and **Upload to this volume** every `.parquet` file.
# MAGIC 4. Put the row count from step 2 into `EXPECTED_ROWS` below, then run this notebook.
# MAGIC
# MAGIC **Output:** `workspace.armd.bronze_comorbidity`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "armd"
VOLUME = "raw"

COMORBIDITY_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/comorbidity"
TABLE = f"{CATALOG}.{SCHEMA}.bronze_comorbidity"

EXPECTED_ROWS = None   # the row count convert_comorbidity.py printed, e.g. 123_456_789

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Check the files were uploaded

# COMMAND ----------

parts = [f for f in dbutils.fs.ls(COMORBIDITY_DIR) if f.name.endswith(".parquet")]
if not parts:
    raise FileNotFoundError(f"No .parquet files in {COMORBIDITY_DIR} yet")
print(f"{len(parts)} files, {sum(f.size for f in parts) / 1e9:.2f} GB in total")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load them and save as a Delta table
# MAGIC The same clean-ups as notebook 01: lower-case column names (and fix the `culutre` typo), and make
# MAGIC sure the text "null" counts as missing. Everything stays text; notebook 02 converts the columns
# MAGIC it needs.

# COMMAND ----------

NULL_TOKENS = ["null", ""]


def clean_column_name(name: str) -> str:
    return name.strip().lower().replace("culutre", "culture")


df = spark.read.parquet(COMORBIDITY_DIR)
df = df.toDF(*[clean_column_name(c) for c in df.columns])
df = df.select([
    F.when(F.lower(F.trim(F.col(c))).isin(NULL_TOKENS), None).otherwise(F.col(c)).alias(c)
    for c in df.columns
])
df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(TABLE)

table = spark.table(TABLE)
actual = table.count()
print(f"Saved {TABLE}: {actual:,} rows; columns: {', '.join(table.columns)}")
if EXPECTED_ROWS is None:
    print("EXPECTED_ROWS isn't set, so the row count wasn't checked.")
else:
    assert actual == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} rows but loaded {actual:,}: re-upload the files"
    print("Row count OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Check the columns notebook 02 needs
# MAGIC Notebook 02 uses `order_proc_id_coded`, `comorbidity_component` and
# MAGIC `comorbidity_component_start_days_culture`. If a name differs, tell the team before running 02.
# MAGIC
# MAGIC **Which way do the days count?** In every other ARMD file a positive number means "this many days
# MAGIC *before* the culture". Here most start days should be positive too (diagnoses usually come
# MAGIC before the infection). Notebook 02 stops with an error if most of them are negative.

# COMMAND ----------

needed = ["order_proc_id_coded", "comorbidity_component", "comorbidity_component_start_days_culture"]
missing = [c for c in needed if c not in table.columns]
assert not missing, f"Columns notebook 02 needs are missing: {missing}. Found: {table.columns}"

start = F.col("comorbidity_component_start_days_culture").cast("int")
display(table.select(
    F.count("*").alias("rows"),
    F.countDistinct("order_proc_id_coded").alias("cultures"),
    F.countDistinct("comorbidity_component").alias("different_diagnoses"),
    F.round(100 * F.avg((start < 0).cast("int")), 1).alias("pct_start_days_negative"),
    F.min(start).alias("min_start_days"),
    F.expr("percentile_approx(cast(comorbidity_component_start_days_culture as int), 0.5)").alias("median_start_days"),
    F.max(start).alias("max_start_days"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The most common diagnoses
# MAGIC Notebook 02 keeps the most common ones (60 by default) as yes/no features.

# COMMAND ----------

display(
    table.groupBy("comorbidity_component")
    .agg(F.countDistinct("order_proc_id_coded").alias("cultures"))
    .orderBy(F.desc("cultures"))
    .limit(30)
)
